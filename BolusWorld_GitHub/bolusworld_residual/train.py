import argparse
import hashlib
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from vae_stage import ASSETS, MANIFEST, ROOT, MRISlices, PHASES, decode, decoder_stage, emit, image_loss, save
from model import BolusWorld, LoRALinear
from run import evaluate, restore, trainable, write_image


class ResidualBolusWorld(BolusWorld):
    def __init__(self, decoder_path):

        super().__init__(ASSETS, rank=32, dtype=torch.float32)
        for scope in (self.backbone.showo.model, self.backbone.diffusion_head_a):
            for module in list(scope.modules()):
                if all(hasattr(module, p) for p in ("gate_proj", "up_proj", "down_proj")):
                    for p in ("gate_proj", "up_proj", "down_proj"):
                        setattr(module, p, LoRALinear(getattr(module, p), 32).cuda())
        if decoder_path.is_file():
            self.vae.model.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True)["decoder"])

    def image(self, latent):
        return decode(self.vae, latent)

    def loss(self, source, target, phase):
        src = self.encode(F.pad(source, (16, 16)))
        goal = self.encode(F.pad(target, (16, 16)))
        start = src + .1 * torch.randn_like(src)
        t = torch.randn(len(goal), device=goal.device).sigmoid()
        weight = t[:, None, None, None]
        z = torch.lerp(start, goal, weight)
        v = self(z, t, self.conditioning(src, phase))
        flow = F.mse_loss(v, goal - start)
        pred = checkpoint(self.image, z + (1 - weight) * v, use_reentrant=False)
        l1, structure = image_loss(pred, target)
        return flow + .1 * (l1 + structure), (flow.detach(), l1.detach(), structure.detach())

    @torch.no_grad()
    def generate(self, source, phase, steps, generator):
        src = self.encode(F.pad(source, (16, 16)))
        if isinstance(generator, list):
            noise = torch.stack([torch.randn(src.shape[1:], device=src.device, generator=rng) for rng in generator])
        else:
            noise = torch.randn(src.shape, device=src.device, generator=generator)
        z = src + .1 * noise
        cond = self.conditioning(src, phase)
        for i in range(steps):
            z = z + self(z, z.new_full((len(z),), i / steps), cond) / steps
        return self.image(z)[:, 0].clamp(0, 1)


@torch.no_grad()
def warm_start(params, path):
    old = torch.load(path, map_location="cpu", weights_only=True)["adapters"]
    for key, x in old.items():
        p = params[key]
        if p.shape == x.shape:
            p.copy_(x)
        elif key.endswith(".up.weight") and p.shape[0] == x.shape[0] and p.shape[1] > x.shape[1]:
            p[:, :x.shape[1]].copy_(x)
        elif key.endswith(".down.weight") and p.shape[1] == x.shape[1] and p.shape[0] > x.shape[0]:
            p[:x.shape[0]].copy_(x)
        else:
            raise ValueError(f"Incompatible warm start tensor: {key}")


@contextmanager
def use_ema(params, ema):
    live = {k: p.detach().clone() for k, p in params.items()}
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(ema[k])
    try:
        yield
    finally:
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(live[k])


def train(args, model, state):
    data = MRISlices(ROOT, MANIFEST, "train", "all")
    val = DataLoader(MRISlices(ROOT, MANIFEST, "val", "all", central_only=True), batch_size=1, num_workers=2)
    params = trainable(model)
    if args.warm_start:
        warm_start(params, args.warm_start)
    opt = torch.optim.AdamW(params.values(), lr=1e-4)
    opt_steps = math.ceil(math.ceil(len(data) / args.batch) / args.accum)
    manifest = manifest_hash()
    decoder = decoder_hash(args.run)
    ema = {k: p.detach().clone() for k, p in params.items()}
    first_epoch, update, best = 1, 0, -1.
    if state:
        schedule = (args.batch, args.accum, args.epochs)
        if state["schedule"] != schedule:
            raise ValueError("Resume requires the original batch, accumulation and epoch settings")
        restore(model, state["train_adapters"])
        opt.load_state_dict(state["optimizer"])
        for k, value in state["ema"].items():
            ema[k].copy_(value)
        first_epoch, update, best = state["epoch"] + 1, state["update"], state["best"]
    emit(event="start", batch=args.batch, accumulation=args.accum, rank=32, alpha=32, epochs=args.epochs,
         precision="tf32", device=torch.cuda.get_device_name(),
         init=str(args.warm_start or args.resume or "pretrained"), trainable=sum(p.numel() for p in params.values()),
         samples=len(data), decoder="tuned" if (args.run / "decoder.pt").exists() else "pretrained")
    for epoch in range(first_epoch, args.epochs + 1):
        loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=2,
                            generator=torch.Generator().manual_seed(42 + epoch))
        torch.manual_seed(42 + epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for i, (pre, target, phase, *_) in enumerate(loader, 1):
            loss, (flow, l1, structure) = model.loss(pre.cuda(), target.cuda(), phase.cuda())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            start = ((i - 1) // args.accum) * args.accum
            end = min(start + args.accum, len(loader))
            group_size = min(end * args.batch, len(data)) - start * args.batch
            (loss * (len(pre) / group_size)).backward()
            if i == end or args.probe:
                update += 1
                progress = max(0., (update - 500) / max(1, args.epochs * opt_steps - 500))
                lr = 1e-4 * update / 500 if update <= 500 else 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * progress)) / 2
                opt.param_groups[0]["lr"] = lr
                grad = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
                decay = min(0.999, (1 + update) / (10 + update))
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.values()), list(params.values()), 1 - decay)
                if update < 4 or update % 25 == 0 or i == len(loader) or args.probe:
                    emit(event="update", epoch=epoch, batch=i, batches=len(loader), update=update, lr=lr, ema_decay=decay,
                         loss=loss.item(), flow=flow.item(), l1=l1.item(), one_step_ssim=1-structure.item(),
                         grad=grad.item(), seconds_per_batch=(time.monotonic()-started)/i,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                if not args.probe and update == 500 and i != len(loader):
                    with use_ema(params, ema):
                        early = evaluate(model, val, 20)
                    emit(event="warmup_validation", epoch=epoch, update=update, **early["overall"])
                    if early["overall"]["ssim"] > best:
                        best = early["overall"]["ssim"]
                        save(args.run / "best.pt", dict(epoch=epoch, update=update, manifest=manifest,
                             decoder=decoder, best=best, validation=early,
                             adapters={k: p.detach().cpu() for k, p in ema.items()}))
                    model.train()
            if args.probe and i >= 3:
                break
        with use_ema(params, ema):
            metrics = evaluate(model, val, 20, limit=1 if args.probe else 0)
        emit(event="probe_done" if args.probe else "validation", epoch=epoch, **metrics["overall"])
        if args.probe:
            return
        value = metrics["overall"]["ssim"]
        meta = dict(epoch=epoch, manifest=manifest, validation=metrics,
                    decoder=decoder, best=max(best, value))
        exported = {k: p.detach().cpu() for k, p in ema.items()}
        if value > best:
            save(args.run / "best.pt", dict(**meta, adapters=exported))
            best = value
        save(args.run / "last.pt", dict(**meta, update=update, schedule=(args.batch, args.accum, args.epochs),
             train_adapters={k:p.detach().cpu() for k,p in params.items()}, ema=exported, optimizer=opt.state_dict()))


def manifest_hash():
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def decoder_hash(run):
    path = run / "decoder.pt"
    if path.is_file():
        with path.open("rb") as f:
            sha = hashlib.sha256()
            for block in iter(lambda: f.read(8 << 20), b""):
                sha.update(block)
        return sha.hexdigest()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("decoder", "train", "test", "generate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.accum, args.epochs) < 1 or (args.resume and args.warm_start):
        parser.error("Invalid batch/accum/epochs or simultaneous resume and warm start")
    if args.action == "generate" and not (args.input and args.output):
        parser.error("provide --input and --output")
    if torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name():
        raise RuntimeError("Expose only the RTX 3090")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.run.mkdir(parents=True, exist_ok=True)
    if args.action == "decoder":
        decoder_stage(args.run, args.batch, .8)
        return
    state = {}
    if args.resume or args.action != "train":
        state = torch.load(args.run / ("last.pt" if args.resume else "best.pt"), map_location="cpu", weights_only=True)
        if state["manifest"] != manifest_hash() or state["decoder"] != decoder_hash(args.run):
            raise ValueError("Manifest or decoder differs from checkpoint")
    elif not args.probe and (args.run / "last.pt").exists():
        raise ValueError("Use --resume for an existing run")
    model = ResidualBolusWorld(args.run / "decoder.pt")
    if args.action == "train":
        train(args, model, state)
        return
    restore(model, state["adapters"])
    model.eval()
    if args.action == "test":
        import json
        dl = DataLoader(MRISlices(ROOT, MANIFEST, "test", "all"), batch_size=args.batch, num_workers=2)
        report = evaluate(model, dl, 20, export=args.output)
        (args.run / "test.json").write_text(json.dumps(report, indent=2))
        emit(event="test", **report["overall"])
        return
    source = np.load(args.input, allow_pickle=False).astype(np.float32)
    if source.shape != (256, 224) or not np.isfinite(source).all() or source.min() < 0 or source.max() > 1:
        raise ValueError("Expected normalized pre-T1 slice of shape 256x224")
    source = torch.from_numpy(source)[None, None].cuda()
    for i, phase in enumerate(PHASES):
        label = torch.tensor([i], device="cuda")
        rng = torch.Generator(device="cuda").manual_seed(42)
        image = model.generate(source, label, 20, rng)[0].cpu().numpy()
        write_image(args.output / f"{phase}.png", image)
        np.save(args.output / f"{phase}.npy", image)


if __name__ == "__main__":
    main()
