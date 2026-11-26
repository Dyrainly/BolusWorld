# BolusWorld

**A sequential visual world model enables contrast-free multiphasic liver MRI synthesis**

BolusWorld synthesizes the arterial phase (AP), portal venous phase (PVP), and delayed phase (DP) from a single non-contrast T1-weighted MRI slice. It uses the pre-trained Wan2.1 VAE for slice-wise encoding and a Show-o2 1.5B backbone with spatial and phase conditioning for sequential latent flow matching.

## Environment

Use Linux and Python 3.10 for training, evaluation, and inference. Install a CUDA-compatible PyTorch and torchvision build for your system, then run:

```bash
python -m pip install -r requirements.txt
```

The sequential entry point expects exactly one visible CUDA device. Run the commands below from the repository root.

## Model files

Download the Show-o2 base checkpoint with the included downloader. It checks the expected file size and SHA-256:

```bash
python bolusworld/fetch_showo.py bolusworld_assets/showo
```

Place the **matching Wan2.1 VAE** checkpoint at `bolusworld_assets/wan/Wan2.1_VAE.pth`. The package provides the model configurations and Qwen tokenizer, but trained BolusWorld checkpoints and pretrained weights are obtained separately.

## Data and manifest

Training and evaluation use co-registered, source-normalized float arrays of shape `256 × 224 × 72`. Intensities must be in `[0, 1]`. The data loader expects one `pre.npy` per examination and phase targets named `arterial.npy`, `portal.npy`, and `delayed.npy` as available. Sequential training uses only slices present in all three targets.

Supply your own authorized, de-identified case index as a JSON array. Each record has `dataset`, `case`, `split`, `directory`, `phases`, and `coverage` keys. Use examination aliases compatible with the reader, such as `PATIENT0_MR0` for `internal`, `sub-0` for `osh`, or `MR-0` for `lld`. These are synthetic format examples. Use `train` or `test` for `split`, make `directory` a repository-relative path ending with the case alias, and list zero-based slice indices in `coverage` in arterial, portal, delayed order. An absent phase must have empty coverage.

Build and audit the sequence manifest locally:

```bash
python bolusworld/run.py verify --index data/cases.json --scan
python bolusworld_residual/train_sequence.py audit
```

The archived manifest builder expects all three cohort keys and train, validation, and test coverage for each phase. It assigns validation patients from the training partition and checks that an alias does not span the original partitions. The manifest is written to `bolusworld/paired_slices.json` by default. Use the same manifest for checkpoint evaluation and inference.

## Training, evaluation, and inference

```bash
CUDA_VISIBLE_DEVICES=0 python -u bolusworld_residual/train_sequence.py train \
  --run runs/sequence --epochs 20 --batch 4 --steps 20 --seed 43 --lr 5e-5 \
  --history-every 1 --history-steps 8 --start-fraction .5 \
  --early-val 500 --rollout-loss-steps 4 --balance-cohorts --image-weight .10

CUDA_VISIBLE_DEVICES=0 python bolusworld_residual/train_sequence.py test \
  --run runs/sequence --steps 20

CUDA_VISIBLE_DEVICES=0 python bolusworld_residual/train_sequence.py generate \
  --run runs/sequence --steps 20 \
  --input data/example_slice.npy --output outputs/example
```

To initialize from a compatible warm-start checkpoint, add `--init runs/initialization.pt` to the training command. To resume an existing run, use `--resume` with its original training options. Evaluation and generation load `best.pt` from the run directory and validate the manifest and decoder against the checkpoint. Place the matching `decoder.pt` in the same run directory if that checkpoint used an adapted decoder.

Generation expects one normalized `256 × 224` float NPY slice. It writes `AP`, `PVP`, and `DP` as float NPY arrays and 16-bit PNG images. Test metrics are written to `test.json` in the run directory.

## Repository layout

| Path | Purpose |
| --- | --- |
| `bolusworld_residual/train_sequence.py` | Sequential training, testing, and inference |
| `bolusworld_residual/train.py`, `vae_stage.py` | Model and decoder components used by the sequence workflow |
| `bolusworld/run.py`, `data.py` | Manifest creation, data loading, and phase-wise workflow |
| `bolusworld/fetch_showo.py` | Verified Show-o2 checkpoint download |
| `bolusworld_assets/` | Model configurations, tokenizer, and vendored upstream code |
| `train_4090d.sh` | Bash launcher for the archived RTX 4090D training schedule |

## Data and third-party resources

The repository contains code and public model configuration artifacts. Keep case indexes, source scans, derived images, local checkpoints, and experiment logs outside version control. This implementation is intended for research. See [Third-party notices](THIRD_PARTY_NOTICES.md) for upstream licenses and attribution.
