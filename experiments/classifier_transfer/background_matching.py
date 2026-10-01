"""Deterministic chromosome-wise GC matching shared by training and diagnostics."""
import hashlib

import numpy as np
from scipy.spatial import cKDTree


def gc_features(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != 2048 or not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError("Require unambiguous 2048-bp integer-encoded DNA")
    gc = (codes == 1) | (codes == 2)
    return np.stack((gc.mean(1), gc[:, 768:1280].mean(1)), axis=1)


def sequence_key(codes):
    return hashlib.sha256(min(codes.tobytes(), (3-codes[::-1]).tobytes())).digest()


def match_background(positive, background, seed=20260915, caliper=.02):
    positive, background = np.asarray(positive), np.asarray(background)
    if (positive.ndim != 2 or background.ndim != 2 or positive.shape[1] != 2
            or background.shape[1] != 2 or not len(background) or caliper <= 0
            or not np.isfinite(positive).all() or not np.isfinite(background).all()):
        raise ValueError("Require finite two-dimensional GC features and positive caliper")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(background))
    tree = cKDTree(background[permutation])
    closest = tree.query(positive, k=1, p=np.inf)[0]
    shuffled = rng.permutation(len(positive))
    order = shuffled[np.argsort(-closest[shuffled], kind="stable")]
    used = np.zeros(len(background), dtype=bool)
    match = np.full(len(positive), -1, dtype=np.int64)
    for i in order:
        candidates = np.asarray(tree.query_ball_point(positive[i], caliper, p=np.inf), dtype=int)
        candidates = candidates[~used[candidates]]
        if not len(candidates): continue
        distances = np.square(background[permutation[candidates]]-positive[i]).sum(1)
        selected = candidates[np.lexsort((candidates, distances))[0]]
        used[selected] = True
        match[i] = permutation[selected]
    return match
