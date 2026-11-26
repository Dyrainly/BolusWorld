"""Train and evaluate sequential pre-T1 → AP → PVP → DP latent transitions."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from vae_stage import MANIFEST, PHASES, ROOT, emit, image_loss, load_volume, save
from train import ResidualBolusWorld, decoder_hash, evaluate, manifest_hash, restore, trainable, use_ema, warm_start, write_image


class PairedSequences(Dataset):
    def __init__(self, split, center=False, manifest=MANIFEST, root=ROOT):
        self.root = root
        cases = defaultdict(dict)
        for row in json.loads(manifest.read_text()):
            if row["split"] == split:
                key = (row["dataset"], row["patient"], row["case"])
                if row["phase"] in cases[key]:
                    raise ValueError(f"Duplicate phase for {key}")
                cases[key][row["phase"]] = row
        self.samples = []
        for key, phases in sorted(cases.items()):
            if set(phases) != set(range(len(PHASES))):
                continue
            rows = [phases[i] for i in range(len(PHASES))]
            sources = {row["source"] for row in rows}
            if len(sources) != 1:
                raise ValueError(f"Inconsistent pre-T1 source for {key}")
            layers = sorted(set.intersection(*(set(row["layers"]) for row in rows)))
            if center and layers:
                layers = [layers[len(layers) // 2]]
            paths = (sources.pop(), *(row["target"] for row in rows))
            self.samples.extend((key, paths, layer) for layer in layers)
        if not self.samples:
            raise ValueError(f"No complete sequences for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        (cohort, patient, case), paths, layer = self.samples[i]
        images = [torch.from_numpy(load_volume(self.root / path)[:, :, layer].astype(np.float32))[None]
                  for path in paths]
        return images[0], torch.stack(images[1:]), cohort, patient, case, layer


class SequentialBolusWorld(ResidualBolusWorld):
    """Each transition starts from a carried latent and retains the original pre-T1 condition."""

    def transition(self, anchor, previous, phase, noise, steps, checkpoint_velocity=False):
        state = previous + 0.1 * noise
        cond = self.conditioning(anchor, phase)
        for i in range(steps):
            t = state.new_full((len(state),), i / steps)
            velocity = (checkpoint(self, state, t, cond, use_reentrant=False)
                        if checkpoint_velocity else self(state, t, cond))
            state = state + velocity / steps
        return state

    def rollout_latents(self, source, steps, generators, last_phase=2):
        anchor = self.encode(F.pad(source, (16, 16)))
        noise = torch.stack([torch.randn(anchor.shape[1:], device=anchor.device, generator=rng)
                             for rng in generators])
        state = anchor
        for i in range(last_phase + 1):
            phase = torch.full((len(anchor),), i, dtype=torch.long, device=anchor.device)
            state = self.transition(anchor, state, phase, noise, steps)
            yield state

    @torch.no_grad()
    def generate(self, source, phase, steps, generators):
        output = torch.empty_like(source[:, 0])
        for i, state in enumerate(self.rollout_latents(source, steps, generators, int(phase.max()))):
            selected = phase == i
            if selected.any():
                output[selected] = self.image(state[selected])[:, 0].clamp(0, 1)
        return output

    def transition_loss(self, source, targets, index, generated_history,
                        history_steps=2, start_fraction=0.0, image_weight=0.1,
                        rollout_loss_steps=0):
        anchor = self.encode(F.pad(source, (16, 16)))
        goal_image = targets[:, index]
        goal = self.encode(F.pad(goal_image, (16, 16)))
        phase = torch.full((len(anchor),), index, dtype=torch.long, device=anchor.device)
        noise = torch.randn_like(anchor)
        previous = anchor
        if index and generated_history:

            with torch.no_grad():
                for j in range(index):
                    previous = self.transition(anchor, previous, torch.full_like(phase, j), noise,
                                               steps=history_steps)
        elif index:
            previous = self.encode(F.pad(targets[:, index - 1], (16, 16)))
        start = previous + 0.1 * noise
        t = torch.randn(len(goal), device=goal.device).sigmoid()
        if start_fraction:

            t = t.masked_fill(torch.rand_like(t) < start_fraction, 0.)
        weight = t[:, None, None, None]
        z = torch.lerp(start, goal, weight)
        cond = self.conditioning(anchor, phase)
        velocity = (checkpoint(self, z, t, cond, use_reentrant=False)
                    if rollout_loss_steps else self(z, t, cond))
        flow = F.mse_loss(velocity, goal - start)
        endpoint = z + (1 - weight) * velocity
        if rollout_loss_steps:
            endpoint = self.transition(anchor, previous, phase, noise, rollout_loss_steps,
                                       checkpoint_velocity=True)
        image = checkpoint(self.image, endpoint, use_reentrant=False)
        l1, structure = image_loss(image, goal_image)
        return flow + image_weight * (l1 + structure)


@torch.no_grad()
def evaluate_sequences(model, sequences, steps, limit=0):
    model.eval()

    def batches():
        for i, (pre, targets, cohorts, patients, cases, layers) in enumerate(sequences):
            if limit and i >= limit:
                break
            count = len(PHASES)
            yield (pre.expand(count, -1, -1, -1), targets[0], torch.arange(count),
                   cohorts * count, patients * count, cases * count, layers.expand(count))

    class Predictor:
        def eval(self):
            model.eval()

        def generate(self, pre, phase, steps, generators):
            return torch.cat([model.image(z)[:, 0].clamp(0, 1)
                              for z in model.rollout_latents(pre[:1], steps, generators[:1])])

    return evaluate(Predictor(), batches(), steps)


def fit(args, model, state):
    samples = PairedSequences("train")
    cohort_counts = Counter(key[0] for key, _, _ in samples.samples)
    val = DataLoader(PairedSequences("val", center=True), batch_size=1)
    params = trainable(model)
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr)
    ema = {k: p.detach().clone() for k, p in params.items()}
    update, best = 0, -1.
    if state:
        restore(model, state["train_adapters"])
        optimizer.load_state_dict(state["optimizer"])
        for k, tensor in state["adapters"].items():
            ema[k].copy_(tensor)
        update, best = state["update"], state["best"]
    elif args.init:
        warm_start(params, args.init)
        for k, p in params.items():
            ema[k].copy_(p.detach())
    emit(event="sequence_start", device=torch.cuda.get_device_name(), sequences=len(samples),
         batch=args.batch, steps=args.steps, epochs=args.epochs, seed=args.seed, peak_lr=args.lr,
         history_every=args.history_every, history_warmup_updates=100,
         history_steps=args.history_steps, start_fraction=args.start_fraction,
         image_weight=args.image_weight, rollout_loss_steps=args.rollout_loss_steps,
         balance_cohorts=args.balance_cohorts, cohort_counts=dict(cohort_counts),
         initialization=str(args.init or args.resume))
    for epoch in range(state.get("epoch", 0) + 1, args.epochs + 1):
        rng = torch.Generator().manual_seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        sampler = None
        if args.balance_cohorts:
            weights = [1 / cohort_counts[key[0]] for key, _, _ in samples.samples]
            sampler = WeightedRandomSampler(weights, len(samples), replacement=True, generator=rng)
        loader = DataLoader(samples, batch_size=args.batch, sampler=sampler,
                            shuffle=sampler is None, generator=rng, num_workers=2)
        total_updates = args.epochs * len(loader)
        model.train()
        for i, (pre, targets, *_) in enumerate(loader, 1):
            index = update % len(PHASES)
            generated = bool(index and (args.smoke or
                             (update >= 100 and update // len(PHASES) % args.history_every == 0)))
            optimizer.zero_grad(set_to_none=True)
            loss = model.transition_loss(pre.cuda(), targets.cuda(), index, generated,
                                         args.history_steps, args.start_fraction, args.image_weight,
                                         args.rollout_loss_steps)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite sequential transition loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
            update += 1
            progress = max(0., (update - 500) / max(1, total_updates - 500))
            lr = (args.lr * update / 500 if update <= 500 else
                  1e-6 + (args.lr - 1e-6) * (1 + math.cos(math.pi * progress)) / 2)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.step()
            with torch.no_grad():
                decay = min(.999, (1 + update) / (10 + update))
                torch._foreach_lerp_(list(ema.values()), list(params.values()), 1 - decay)
            if i == 1 or i % 25 == 0 or args.smoke:
                emit(event="sequence_update", epoch=epoch, batch=i, update=update, lr=lr, phase=PHASES[index],
                     generated_history=generated, loss=loss.item(), grad=grad.item(),
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            if generated and update <= 110 and not args.smoke:
                emit(event="generated_history_update", update=update, phase=PHASES[index],
                     loss=loss.item())
            if args.early_val and update == args.early_val and i != len(loader) and not args.smoke:
                with use_ema(params, ema):
                    early = evaluate_sequences(model, val, args.steps)
                emit(event="early_sequence_validation", epoch=epoch, update=update,
                     by_cohort_phase=early["by_cohort_phase"], **early["overall"])
                if early["overall"]["ssim"] > best:
                    best = early["overall"]["ssim"]
                    save(args.run / "best.pt", dict(epoch=epoch, update=update, best=best,
                         validation=early, manifest=manifest_hash(), decoder=decoder_hash(args.run),
                         steps=args.steps, seed=args.seed, peak_lr=args.lr, history_every=args.history_every,
                         history_steps=args.history_steps, start_fraction=args.start_fraction,
                         image_weight=args.image_weight, rollout_loss_steps=args.rollout_loss_steps,
                         balance_cohorts=args.balance_cohorts,
                         adapters={k: p.detach().cpu() for k, p in ema.items()}))
                model.train()
            if args.smoke and i == len(PHASES):
                break
        with use_ema(params, ema):
            report = evaluate_sequences(model, val, args.steps, limit=1 if args.smoke else 0)
        emit(event="sequence_validation", epoch=epoch,
             by_cohort_phase=report["by_cohort_phase"], **report["overall"])
        score = report["overall"]["ssim"]
        payload = dict(epoch=epoch, update=update, best=max(best, score), validation=report,
                       manifest=manifest_hash(), decoder=decoder_hash(args.run), schedule=(args.batch, args.epochs),
                       steps=args.steps, seed=args.seed, smoke=args.smoke, history_every=args.history_every,
                       peak_lr=args.lr, history_steps=args.history_steps, start_fraction=args.start_fraction,
                       image_weight=args.image_weight, rollout_loss_steps=args.rollout_loss_steps,
                       balance_cohorts=args.balance_cohorts,
                       adapters={k: p.detach().cpu() for k, p in ema.items()})
        if score > best:
            save(args.run / "best.pt", payload)
            best = score
        save(args.run / "last.pt", dict(**payload, optimizer=optimizer.state_dict(),
                                       train_adapters={k: p.detach().cpu() for k, p in params.items()}))
        if args.smoke:
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("audit", "train", "test", "generate"))
    p.add_argument("--run", type=Path, default=ROOT / "bolusworld/sequence")
    p.add_argument("--init", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate after the 500-update warmup.")
    p.add_argument("--history-every", type=int, default=4,
                   help="After warmup, use generated predecessors every N AP/PVP/DP training cycles.")
    p.add_argument("--history-steps", type=int, default=2)
    p.add_argument("--start-fraction", type=float, default=0.)
    p.add_argument("--image-weight", type=float, default=0.1,
                   help="Weight on decoded endpoint L1 plus (1 - SSIM).")
    p.add_argument("--early-val", type=int, default=0)
    p.add_argument("--rollout-loss-steps", type=int, default=0,
                   help="Euler steps for decoded endpoint supervision. Zero uses a one-step estimate.")
    p.add_argument("--balance-cohorts", action="store_true",
                   help="Sample each cohort with equal probability within each training epoch.")
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "audit":
        for split in ("train", "val", "test"):
            data = PairedSequences(split)
            cases = Counter(c[0] for c in {key for key, _, _ in data.samples})
            layers = Counter(key[0] for key, _, _ in data.samples)
            emit(split=split, cases=dict(cases), aligned_sequences=dict(layers))
        return
    if min(args.batch, args.steps, args.epochs, args.history_every, args.history_steps) < 1 or (args.init and args.resume):
        p.error("Invalid batch, steps, epochs, history interval or resume/init combination")
    if args.rollout_loss_steps < 0 or not 0 <= args.start_fraction < 1 or args.early_val < 0:
        p.error("Invalid start-state fraction or early validation interval")
    if not math.isfinite(args.image_weight) or args.image_weight <= 0:
        p.error("Image loss weight must be positive and finite")
    if not math.isfinite(args.lr) or args.lr < 1e-6:
        p.error("Peak learning rate must be finite and at least 1e-6")
    if args.action == "generate" and not (args.input and args.output):
        p.error("generate requires --input and --output")
    if torch.cuda.device_count() != 1:
        p.error("Expose exactly one CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    args.run.mkdir(parents=True, exist_ok=True)
    state = {}
    if args.resume or args.action != "train":
        state = torch.load(args.run / ("last.pt" if args.resume else "best.pt"), map_location="cpu", weights_only=True)
        if (state["manifest"], state["decoder"]) != (manifest_hash(), decoder_hash(args.run)):
            raise ValueError("Manifest or decoder changed")
        if state.get("smoke") and not args.smoke and (args.resume or args.action == "test"):
            p.error("Smoke checkpoints cannot be used for formal training or evaluation")
        if args.resume and (state["schedule"], state["steps"]) != ((args.batch, args.epochs), args.steps):
            raise ValueError("Resume requires the original batch, epochs and sampler steps")
        if args.resume and state.get("seed", 42) != args.seed:
            raise ValueError("Resume requires the original training seed")
        if args.resume and state.get("history_every", 4) != args.history_every:
            raise ValueError("Resume requires the original generated-history interval")
        if args.resume and state.get("peak_lr", 1e-4) != args.lr:
            raise ValueError("Resume requires the original peak learning rate")
        if args.resume and (state.get("history_steps", 2), state.get("start_fraction", 0.)) != (
                args.history_steps, args.start_fraction):
            raise ValueError("Resume requires the original training rollout settings")
        if args.resume and state.get("image_weight", .1) != args.image_weight:
            raise ValueError("Resume requires the original image loss weight")
        if args.resume and (state.get("rollout_loss_steps", 0), state.get("balance_cohorts", False)) != (
                args.rollout_loss_steps, args.balance_cohorts):
            raise ValueError("Resume requires the original endpoint supervision and cohort sampling settings")
    elif any((args.run / name).exists() for name in ("last.pt", "best.pt")):
        p.error("Use --resume or a fresh run directory")
    torch.manual_seed(args.seed)
    model = SequentialBolusWorld(args.run / "decoder.pt")
    if args.action == "train":
        fit(args, model, state)
    else:
        restore(model, state["adapters"])
        model.eval()
        if args.action == "test":
            report = evaluate_sequences(model, DataLoader(PairedSequences("test"), batch_size=1), args.steps)
            (args.run / "test.json").write_text(json.dumps(report, indent=2))
            emit(event="sequence_test", **report["overall"])
            return
        pre = np.load(args.input, allow_pickle=False).astype(np.float32)
        if pre.shape != (256, 224) or not np.isfinite(pre).all() or pre.min() < 0 or pre.max() > 1:
            raise ValueError("Expected normalized 256×224 pre-T1")
        source = torch.from_numpy(pre)[None, None].cuda()
        with torch.no_grad():
            rng = torch.Generator(device="cuda").manual_seed(42)
            for phase, latent in zip(PHASES, model.rollout_latents(source, args.steps, [rng])):
                image = model.image(latent)[0, 0].clamp(0, 1).cpu().numpy()
                write_image(args.output / f"{phase}.png", image)
                np.save(args.output / f"{phase}.npy", image)


if __name__ == "__main__":
    main()
