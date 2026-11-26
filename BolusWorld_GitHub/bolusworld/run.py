import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from data import COHORTS, PHASES, MRISlices, build_manifest
from fetch_showo import SHA256
from model import BolusWorld


def emit(**row):
    print(json.dumps(row), flush=True)


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(image * 65535).astype(np.uint16)).save(path)


@torch.no_grad()
def evaluate(model, loader, steps, export=None, limit=0):
    model.eval()
    patients = defaultdict(list)
    names = ("psnr", "ssim", "mae", "rmse")
    for i, (pre, target, phase, cohorts, patient, cases, layers) in enumerate(loader):
        if limit and i >= limit:
            break

        generators = [torch.Generator(device="cuda").manual_seed(int.from_bytes(hashlib.sha256(
            f"{dataset}/{case}/{layer.item()}".encode()).digest()[:4], "little"))
                      for dataset, case, layer in zip(cohorts, cases, layers)]
        prediction = model.generate(pre.cuda(), phase.cuda(), steps, generators).cpu().numpy()
        for j, (pred, real) in enumerate(zip(prediction, target[:, 0].numpy())):
            d = pred - real
            mse = np.mean(d ** 2)
            p = PHASES[phase[j].item()]
            patients[cohorts[j], p, patient[j]].append([
                -10 * np.log10(max(mse, 1e-12)), structural_similarity(real, pred, data_range=1,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False),
                np.abs(d).mean(), np.sqrt(mse)])
            if export:
                write_image(export / cohorts[j] / cases[j] / p / f"{layers[j].item():03d}.png", pred)
    groups = defaultdict(list)
    for (cohort, phase, patient), scores in patients.items():
        groups[cohort, phase].append(np.mean(scores, axis=0))
    averages = {"/".join(k): np.mean(v, axis=0) for k, v in groups.items()}
    return dict(overall=dict(zip(names, np.mean(list(averages.values()), axis=0).tolist())),
                by_cohort_phase={k: dict(zip(names, v.tolist()), patients=len(groups[tuple(k.split("/"))]))
                                 for k, v in averages.items()},
                by_patient_phase=[dict(dataset=k[0], phase=k[1], patient=k[2], slices=len(v),
                                       **dict(zip(names, np.mean(v, axis=0).tolist())))
                                  for k, v in patients.items()])


def trainable(model):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def restore(model, state):
    current = trainable(model)
    if current.keys() != state.keys():
        raise ValueError("Checkpoint parameter layout differs")
    with torch.no_grad():
        for key, param in current.items():
            if param.shape != state[key].shape:
                raise ValueError(f"Checkpoint shape mismatch: {key}")
            param.copy_(state[key])


def save(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(args, model, state):
    dataset = MRISlices(args.root, args.manifest, "train", args.cohort)
    validation = DataLoader(MRISlices(args.root, args.manifest, "val", args.cohort,
                                      central_only=not args.full_validation), batch_size=1,
                            num_workers=args.workers)
    args.run.mkdir(parents=True, exist_ok=True)
    if not state and any((args.run / f"{x}.pt").exists() for x in ("best", "last")):
        raise ValueError("Run already exists. Resume with --checkpoint or use another --run.")
    params = trainable(model)
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if state:
        if state["cohort"] != args.cohort or state["manifest_hash"] != manifest_hash:
            raise ValueError("Resume requires the same cohort and patient manifest")
        optimizer.load_state_dict(state["optimizer"])
    best = state.get("best", -1.)
    emit(event="start", device=torch.cuda.get_device_name(), precision=args.precision, batch=args.batch,
         accumulation=args.accum, trainable=sum(p.numel() for p in params.values()), samples=len(dataset))
    for epoch in range(state.get("epoch", 0) + 1, args.epochs + 1):
        start_time = time.monotonic()
        torch.manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        for i, (pre, target, phase, *_) in enumerate(loader):
            loss = model.loss(pre.cuda(), target.cuda(), phase.cuda())
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite flow loss")
            end = min((i // args.accum + 1) * args.accum, len(loader))
            start = (i // args.accum) * args.accum
            group_size = min(end * args.batch, len(dataset)) - start * args.batch
            (loss * (len(pre) / group_size if not args.smoke else 1)).backward()
            if args.smoke or i + 1 == end:
                probes = {}
                if args.smoke:
                    for group in ("source_adapter", "phase_embedding", "lora"):
                        matching = [(k, v) for k, v in params.items()
                                    if k.startswith(group) or (group == "lora" and k.endswith(".up.weight"))]
                        probes[group] = next((v for _, v in matching if v.grad is not None and v.grad.abs().max() > 0), None)
                        if probes[group] is None:
                            raise RuntimeError(f"No nonzero gradients for {group}")
                    probes = {group: (param, param.detach().clone()) for group, param in probes.items()}
                grad = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
                optimizer.step()
                if probes and not all(not torch.equal(param, before) for param, before in probes.values()):
                    raise RuntimeError("A conditioning or LoRA parameter did not update")
                optimizer.zero_grad(set_to_none=True)
                if i < args.accum or (i + 1) % (20 * args.accum) == 0 or i + 1 == len(loader):
                    emit(event="update", epoch=epoch, batch=i + 1, batches=len(loader),
                         seconds_per_batch=(time.monotonic() - start_time) / (i + 1),
                         loss=loss.item(), grad=grad.item(),
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            if args.smoke:
                break
        metrics = evaluate(model, validation, args.steps, limit=1 if args.smoke else 0)
        emit(event="validation", epoch=epoch, **metrics["overall"])
        score = metrics["overall"]["ssim"]
        result = dict(epoch=epoch, rank=args.rank, cohort=args.cohort, manifest_hash=manifest_hash,
                      optimizer=optimizer.state_dict(), adapters={k: v.detach().cpu() for k, v in params.items()},
                      best=max(best, score), validation=metrics, smoke=args.smoke,
                      base_sha256=SHA256, precision=args.precision)
        save(args.run / "last.pt", result)
        if score > best:
            save(args.run / "best.pt", result)
            best = score
        if args.smoke:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("verify", "train", "test", "generate"))
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--assets", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--index", type=Path, help="Input case index for verify.")
    p.add_argument("--run", type=Path, default=Path(__file__).resolve().parent / "runs/default")
    p.add_argument("--cohort", choices=("all", *COHORTS), default="all")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--phase", choices=("all", *PHASES), default="all")
    p.add_argument("--output", type=Path)
    p.add_argument("--scan", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full-validation", action="store_true")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=("bf16", "tf32"))
    args = p.parse_args()
    args.root = args.root.resolve()
    args.assets = (args.assets or args.root / "bolusworld_assets").resolve()
    args.manifest = args.manifest or args.root / "bolusworld/paired_slices.json"
    if args.action == "verify":
        emit(event="verified", slices=build_manifest(args.root,
             (args.index or args.root / "data/cases.json"), args.manifest, args.scan))
        return
    if min(args.steps, args.epochs, args.batch, args.accum, args.rank) < 1:
        p.error("steps, epochs, batch, accum and rank must be positive")
    if args.action in ("test", "generate") and not args.checkpoint:
        p.error("test and generate require --checkpoint")
    if args.action == "generate" and (not args.input or not args.output):
        p.error("generate requires --input and --output")
    if torch.cuda.device_count() != 1 or "4090 D" not in torch.cuda.get_device_name():
        raise RuntimeError("Expose only the RTX 4090D using CUDA_VISIBLE_DEVICES")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else {}
    args.precision = args.precision or state.get("precision", "tf32")
    if state and state.get("base_sha256") != SHA256:
        p.error("Checkpoint requires a different pretrained model")
    if state.get("smoke") and args.action in ("train", "test") and not args.smoke:
        p.error("Use a full training checkpoint for training or test metrics")
    args.rank = state.get("rank", args.rank)
    model = BolusWorld(args.assets, args.rank, torch.bfloat16 if args.precision == "bf16" else torch.float32)
    if state:
        restore(model, state["adapters"])
    if args.action == "train":
        train(args, model, state)
    elif args.action == "test":
        if state["manifest_hash"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError("Test manifest differs from training manifest")
        if state["cohort"] != "all" and state["cohort"] != args.cohort:
            raise ValueError("Select the cohort associated with this checkpoint")
        loader = DataLoader(MRISlices(args.root, args.manifest, "test", args.cohort), batch_size=args.batch,
                            num_workers=args.workers)
        report = evaluate(model, loader, args.steps, args.output, limit=1 if args.smoke else 0)
        report["smoke"] = args.smoke
        args.run.mkdir(parents=True, exist_ok=True)
        (args.run / "test.json").write_text(json.dumps(report, indent=2))
        emit(event="test", **report["overall"])
    else:
        array = np.load(args.input, allow_pickle=False).astype(np.float32)
        if array.shape != (256, 224) or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Expected a preprocessed 256×224 pre-T1 .npy slice with intensities in [0,1]")
        model.eval()
        source = torch.from_numpy(array)[None, None].cuda()
        for phase in PHASES if args.phase == "all" else (args.phase,):
            rng = torch.Generator(device="cuda").manual_seed(args.seed)
            label = torch.tensor([PHASES.index(phase)], device="cuda")
            result = model.generate(source, label, args.steps, rng)[0].cpu().numpy()
            write_image(args.output / f"{phase}.png", result)
            np.save(args.output / f"{phase}.npy", result)


if __name__ == "__main__":
    main()
