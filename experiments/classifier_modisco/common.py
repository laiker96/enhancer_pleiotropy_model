import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def event(name, **values):
    print(json.dumps(dict(event=name, **values), allow_nan=False), flush=True)


def require_allocation(kind):
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("A CECAR Slurm compute allocation is required")
    allowed = (host.startswith("xg") and host[2:].isdigit()) if kind == "gpu" else (host.startswith("n") and host[1:].isdigit())
    if not allowed:
        raise RuntimeError("Refusing analysis on a login node or unexpected partition: "+host)


def load_inputs(root):
    config = json.loads((root/"config.json").read_text())
    provenance = json.loads((root/"input_provenance.json").read_text())
    for name, expected in provenance["inputs"].items():
        if digest(root/name) != expected: raise ValueError("Input changed: "+name)
    with np.load(root/"cohort.npz", allow_pickle=False) as saved: data = dict(saved)
    return config, provenance, data


def balanced_indices(labels, split, quality, bins, maximum, seed):
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels,[0,1]).all() or not labels.any(1).all():
        raise ValueError("Eight valid binary labels required")
    breadth = labels.sum(1)
    covered = np.zeros(len(labels), int)
    groups = []
    for low, high in bins:
        match = (breadth >= low) & (breadth <= high)
        covered += match
        groups.append(np.flatnonzero(match & (split == "train") & quality))
    if not (covered == 1).all(): raise ValueError("Discovery bins must partition exact breadths")
    count = min(maximum, *(len(group) for group in groups))
    if count < 1: raise ValueError("An empty breadth bin prevents balanced discovery")
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(group, count, replace=False) for group in groups])
    return rng.permutation(selected)


def projected_contributions(codes, hypothetical):
    if hypothetical.shape != (len(codes),4,codes.shape[1]) or not np.isfinite(hypothetical).all() or not np.isin(codes,range(4)).all():
        raise ValueError("Unaligned/nonfinite ACGT hypothetical contributions")
    return np.take_along_axis(hypothetical,codes[:,None,:],axis=1)[:,0]
