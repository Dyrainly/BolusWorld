import json
import sys
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fetch_showo import SHA256, SIZE, digest


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x.float())).to(x.dtype)


class SourceImageConditioningAdapter(nn.Sequential):
    def __init__(self, channels, patch_size, hidden):
        super().__init__(nn.Conv2d(channels, 64, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(64, hidden, patch_size, stride=patch_size))
        nn.init.normal_(self[-1].weight, std=0.001)
        nn.init.zeros_(self[-1].bias)


class BolusWorld(nn.Module):
    def __init__(self, assets, rank=8, dtype=torch.bfloat16):
        super().__init__()
        sys.path.insert(0, str(assets / "upstream"))
        from models import Showo2Qwen2_5, WanVAE
        from models.misc import get_text_tokenizer, interpolate_pos_encoding
        from models.modeling_siglip import SiglipModel
        from transformers import SiglipConfig

        weights = assets / "showo/pytorch_model.bin"
        if not weights.is_file() or weights.stat().st_size != SIZE or digest(weights) != SHA256:
            raise ValueError(f"Official Show-o2 checkpoint is missing or failed SHA-256 verification: {weights}")
        config = json.loads((assets / "showo/config.json").read_text())
        config.update(llm_model_path=str(assets / "qwen"), load_from_showo=True)
        siglip = SiglipConfig.from_pretrained(assets / "siglip", local_files_only=True)

        with torch.device("meta"), patch.object(SiglipModel, "from_pretrained", side_effect=lambda _: SiglipModel(siglip)):
            model = Showo2Qwen2_5(**{k: v for k, v in config.items() if not k.startswith("_")})
        state = torch.load(weights, mmap=True, weights_only=True, map_location="cpu")
        model.load_state_dict(state, strict=True, assign=True)
        del state

        for module in model.modules():
            if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq"):
                freq, scale = module.rope_init_fn(module.config, torch.device("cpu"), **module.rope_kwargs)
                module.inv_freq = module.original_inv_freq = freq
                module.attention_scaling = scale
            if "image_position_ids" in module._buffers:
                module.image_position_ids = torch.arange(config["image_latent_height"] * config["image_latent_width"])[None]
        for module in model.modules():
            for name, buf in module.named_buffers(recurse=False):
                if buf.is_meta and name == "position_ids":
                    module.register_buffer(name, torch.arange(buf.numel()).reshape(buf.shape), persistent=False)
        self.backbone = model.requires_grad_(False).to(device="cuda", dtype=dtype)
        self.backbone.showo.config.use_cache = False
        self.backbone.showo.gradient_checkpointing_enable({"use_reentrant": False})
        count = 0
        for parent in list(model.modules()):
            if parent.__class__.__name__ not in {"Qwen2SdpaAttention", "Qwen2Attention", "SdpaAttention"}:
                continue
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                layer = getattr(parent, name)
                setattr(parent, name, LoRALinear(layer, rank).to("cuda"))
                count += 1
        if not count:
            raise RuntimeError("No Show-o2 attention projections matched")
        self.dtype = dtype
        self.patch_size = config["patch_size"]
        self.source_adapter = SourceImageConditioningAdapter(config["image_latent_dim"], self.patch_size,
                                                             config["hidden_size"]).cuda()
        self.phase_embedding = nn.Embedding(3, config["hidden_size"]).cuda()
        nn.init.normal_(self.phase_embedding.weight, std=.02)
        tokenizer, ids = get_text_tokenizer(str(assets / "qwen"), return_showo_token_ids=True)
        if len(tokenizer) != config["llm_vocab_size"]:
            raise ValueError("Tokenizer vocabulary does not match the Show-o2 checkpoint")
        self.register_buffer("prefix", torch.tensor([ids["bos_id"], ids["boi_id"]], device="cuda"), persistent=False)
        self.interpolate_pos = interpolate_pos_encoding
        self.vae = WanVAE(vae_pth=str(assets / "wan/Wan2.1_VAE.pth"), dtype=dtype, device="cuda")
        self.vae.model.to(dtype=dtype)

    @torch.no_grad()
    def encode(self, image):
        rgb = (image * 2 - 1).expand(-1, 3, -1, -1).unsqueeze(2)
        return self.vae.model.sample(rgb.to(self.dtype), self.vae.scale, deterministic=True).float().squeeze(2)

    def conditioning(self, source, phase):
        spatial = self.source_adapter(source.float()).flatten(2).transpose(1, 2)
        return spatial + self.phase_embedding(phase)[:, None, :]

    def forward(self, z, t, conditioning):
        model = self.backbone
        z = z.to(self.dtype)
        batch, channels, height, width = z.shape
        h, w = height // self.patch_size, width // self.patch_size
        with torch.no_grad():
            und = model.image_embedder_und(z)
            pos = self.interpolate_pos(model.config.clip_latent_dim, model.position_embedding, h, w, 1)
            und = model.und_trans(und + pos).last_hidden_state
            visual = model.fusion_proj(torch.cat([und, model.image_embedder_gen(z)], dim=-1))
            time = model.time_embed(t, self.dtype)
            time_token = model.time_embed_proj(time) if hasattr(model, "time_embed_proj") else time
            prefix = model.showo.model.embed_tokens(self.prefix).expand(batch, -1, -1)
        hidden = torch.cat([prefix, time_token[:, None], visual + conditioning.to(self.dtype)], dim=1)
        length = hidden.shape[1]
        positions = torch.tensor([2, length - 2], device=z.device).reshape(1, 1, 2).expand(batch, -1, -1)

        allowed = torch.ones((length, length), device=z.device, dtype=torch.bool).tril()
        allowed[2:, 2:] = True
        mask = torch.zeros_like(allowed, dtype=z.dtype).masked_fill(~allowed, torch.finfo(z.dtype).min)[None, None]
        hidden = model.showo.model(inputs_embeds=hidden, attention_mask=mask.expand(batch, -1, -1, -1),
                                  use_cache=False).last_hidden_state
        if hasattr(model, "diff_proj"):
            hidden = model.diff_proj(hidden)
        rope = torch.arange(length, device=z.device)[None]
        for layer in model.diffusion_head_a:
            kwargs = dict(adaln_input=time, attention_mask=mask, position_ids=rope, modality_positions=positions)
            if self.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, use_reentrant=False, **kwargs)[0]
            else:
                hidden = layer(hidden, **kwargs)[0]
        velocity = model.diffusion_head_b(hidden, time, positions)[:, 3:]
        return model.unpatchify(velocity, h, w).transpose(1, 2).reshape(batch, channels, height, width)

    @torch.no_grad()
    def generate(self, source, phase, steps, generator):
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        padded = F.pad(source, (16, 16))
        latent = self.encode(padded)
        condition = self.conditioning(latent, phase)
        if isinstance(generator, list):
            z = torch.stack([torch.randn(latent.shape[1:], device=latent.device, generator=rng)
                             for rng in generator])
        else:
            z = torch.randn(latent.shape, device=latent.device, generator=generator)
        for step in range(steps):
            t = z.new_full((len(z),), step / steps)
            z = z + self(z, t, condition).float() / steps
        result = self.vae.model.decode(z.unsqueeze(2).to(self.dtype), self.vae.scale).float().mean(1)
        return ((result + 1) / 2)[:, 0, :, 16:-16].clamp(0, 1)

    def loss(self, source, target, phase):
        source = self.encode(F.pad(source, (16, 16)))
        clean = self.encode(F.pad(target, (16, 16)))
        noise = torch.randn_like(clean)
        t = torch.randn(len(clean), device=clean.device).sigmoid()
        noisy = torch.lerp(noise, clean, t[:, None, None, None])
        return F.mse_loss(self(noisy, t, self.conditioning(source, phase)).float(), clean - noise)
