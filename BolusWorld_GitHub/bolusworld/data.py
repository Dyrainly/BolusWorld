import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


COHORTS = ("internal", "osh", "lld")
PHASES = ("AP", "PVP", "DP")
STEMS = ("arterial", "portal", "delayed")
SHAPE = (256, 224, 72)


def patient_id(record):
    case = record["case"]
    dataset = record["dataset"]
    pattern = {"internal": r"(PATIENT\d+)_MR\d+", "osh": r"(sub-\d+)", "lld": r"MR-?(\d+)"}[dataset]
    match = re.fullmatch(pattern, case)
    if not match:
        raise ValueError(f"Unknown patient identifier: {dataset}/{case}")
    return dataset, f"MR{match[1]}" if dataset == "lld" else match[1]


def build_manifest(root, index, output, scan=False):
    rows = json.loads(index.read_text(encoding="utf-8"))
    patients, scans, pre_hashes, groups = {}, set(), {}, []
    counts = Counter()
    for row in rows:
        key = patient_id(row)
        cohort, case, split = row["dataset"], row["case"], row["split"]
        if split not in {"train", "test"} or (cohort, case) in scans:
            raise ValueError(f"Invalid or duplicate scan: {cohort}/{case}")
        scans.add((cohort, case))
        if patients.setdefault(key, split) != split:
            raise ValueError(f"Patient across original partitions: {key}")
        folder = (root / row["directory"]).resolve()
        if not folder.is_relative_to(root) or folder.name != case:
            raise ValueError(f"Invalid case directory: {folder}")
        source = folder / "pre.npy"
        pre = load_volume(source, scan)

        if scan:
            digest = hashlib.sha256(pre.tobytes()).hexdigest()
            if pre_hashes.setdefault(digest, key) != key:
                raise ValueError(f"Identical pre-T1 in different patient records: {key}")
        if set(row["phases"]) - set(STEMS) or len(row["coverage"]) != len(STEMS):
            raise ValueError(f"Unknown phase or coverage layout: {case}")
        for i, stem in enumerate(STEMS):
            layers = row["coverage"][i]
            if stem not in row["phases"]:
                if layers:
                    raise ValueError(f"Coverage for absent {stem}: {case}")
                continue
            if not layers or len(set(layers)) != len(layers) or any(type(z) is not int or not 0 <= z < SHAPE[2] for z in layers):
                raise ValueError(f"Invalid coverage for {stem}: {case}")
            target = folder / f"{stem}.npy"
            image = load_volume(target, scan)
            if scan and (not np.all(np.any(pre[:, :, layers] > 0, axis=(0, 1))) or
                         not np.all(np.any(image[:, :, layers] > 0, axis=(0, 1)))):
                raise ValueError(f"Empty indexed slice in {target}")
            groups.append(dict(dataset=cohort, patient=key[1], case=case, split=split, phase=i,
                               source=str(source.relative_to(root)), target=str(target.relative_to(root)), layers=layers))
    for cohort in COHORTS:
        pool = [k for k, split in patients.items() if k[0] == cohort and split == "train"]
        ordered = sorted(pool, key=lambda k: hashlib.sha256(f"BolusWorld-42/{k}".encode()).digest())
        val = set(ordered[:max(1, round(len(ordered) * .1))])
        for g in groups:
            if (g["dataset"], g["patient"]) in val:
                g["split"] = "val"
    for g in groups:
        counts[(g["dataset"], g["split"], PHASES[g["phase"]])] += len(g["layers"])
    if len(counts) != len(COHORTS) * 3 * len(PHASES):
        raise ValueError("Missing cohort/split/phase")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(groups), encoding="utf-8")
    temp.replace(output)
    return {"/".join(k): v for k, v in sorted(counts.items())}


def load_volume(path, scan=False):
    image = np.load(path, mmap_mode="r", allow_pickle=False)
    if image.shape != SHAPE or image.dtype.kind != "f":
        raise ValueError(f"Unexpected array at {path}: {image.shape} {image.dtype}")
    if scan and (not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError(f"Expected source-normalized intensities in [0, 1]: {path}")
    return image


class MRISlices(Dataset):
    def __init__(self, root, manifest, split, cohort, central_only=False):
        self.root = root
        self.items = [(g, z) for g in json.loads(manifest.read_text()) if g["split"] == split
                      and (cohort == "all" or g["dataset"] == cohort)
                      for z in ([g["layers"][len(g["layers"]) // 2]] if central_only else g["layers"])]
        if not self.items:
            raise ValueError(f"No samples for {cohort}/{split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        group, z = self.items[index]
        pre, target = [torch.from_numpy(load_volume(self.root / group[k])[:, :, z].astype(np.float32))[None]
                       for k in ("source", "target")]
        return pre, target, group["phase"], group["dataset"], group["patient"], group["case"], z
