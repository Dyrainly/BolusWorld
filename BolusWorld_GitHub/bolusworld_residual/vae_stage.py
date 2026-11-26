import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "bolusworld_assets"
MANIFEST = ROOT / "bolusworld/paired_slices.json"
sys.path.insert(0, str(ROOT / "bolusworld"))
from data import MRISlices, PHASES, load_volume
from run import emit, save


def encode(vae, image):
    with torch.no_grad():
        rgb = (F.pad(image, (16, 16)) * 2 - 1).expand(-1, 3, -1, -1).unsqueeze(2)
        return vae.model.sample(rgb.float(), vae.scale, deterministic=True).squeeze(2)


def decode(vae, latent):
    red_green_blue = vae.model.decode(latent.unsqueeze(2).float(), vae.scale)
    return (red_green_blue.mean(1)[..., 16:-16] + 1) / 2


def image_loss(pred, target):
    pred = pred.float()
    target = target.float()
    return F.l1_loss(pred, target), 1 - ssim(pred.clamp(0, 1), target, data_range=1., size_average=True)


def load_vae():
    sys.path.insert(0, str(ASSETS / "upstream"))
    from models import WanVAE
    return WanVAE(vae_pth=str(ASSETS / "wan/Wan2.1_VAE.pth"), dtype=torch.float32, device="cuda")


@torch.no_grad()
def ceiling(vae, batch):
    val = MRISlices(ROOT, MANIFEST, "val", "all", central_only=True)
    scores = defaultdict(list)
    pre_scores = defaultdict(list)
    for source, target, phases, datasets, patients, *_ in DataLoader(val, batch_size=batch, num_workers=2):
        pred = decode(vae, encode(vae, target.cuda())).clamp(0, 1).cpu().numpy()
        pre = decode(vae, encode(vae, source.cuda())).clamp(0, 1).cpu().numpy()
        for i, (d, p, patient) in enumerate(zip(datasets, phases.tolist(), patients)):
            for output, real, bins, key in ((pred, target, scores, (d, PHASES[p], patient)),
                                           (pre, source, pre_scores, (d, "pre", patient))):
                bins[key].append(structural_similarity(real[i, 0].numpy(), output[i, 0], data_range=1,
                                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    def aggregate(bins):
        groups = defaultdict(list)
        for (d, p, _), vals in bins.items():
            groups[f"{d}/{p}"].append(np.mean(vals))
        return {k: float(np.mean(v)) for k, v in groups.items()}
    by_phase = aggregate(scores)
    return dict(ssim=float(np.mean(list(by_phase.values()))), by_phase=by_phase, pre_ssim=aggregate(pre_scores))


class DecoderImages(Dataset):
    def __init__(self):
        pairs = MRISlices(ROOT, MANIFEST, "train", "all")
        self.images = sorted({(group[key], z) for group, z in pairs.items for key in ("source", "target")})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path, z = self.images[index]
        return torch.from_numpy(load_volume(ROOT / path)[:, :, z].astype(np.float32))[None]


def decoder_stage(run, batch, threshold):
    vae = load_vae()
    current = ceiling(vae, batch)
    emit(event="vae_ceiling", threshold=threshold, **current)
    if current["ssim"] >= threshold:
        return
    model = vae.model.decoder.requires_grad_(True).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    data = DecoderImages()
    best = -1.
    for epoch in range(1, 3):
        loader = DataLoader(data, batch_size=batch, shuffle=True, num_workers=2,
                            generator=torch.Generator().manual_seed(42 + epoch))
        torch.cuda.reset_peak_memory_stats()
        for step, image in enumerate(loader, 1):
            image = image.cuda()
            opt.zero_grad(set_to_none=True)
            pred = decode(vae, encode(vae, image))
            l1, structure = image_loss(pred, image)
            (l1 + structure).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if step == 1 or step % 100 == 0 or step == len(loader):
                emit(event="decoder_update", epoch=epoch, batch=step, batches=len(loader),
                     l1=l1.item(), ssim=1-structure.item(), peak_gib=torch.cuda.max_memory_reserved()/2**30)
        model.eval()
        score = ceiling(vae, batch)
        emit(event="decoder_validation", epoch=epoch, **score)
        if score["ssim"] > best:
            save(run / "decoder.pt", dict(epoch=epoch, validation=score, decoder=model.state_dict()))
            best = score["ssim"]
        model.train()
