"""Forward/RC logit attribution with exact dinucleotide-preserving references."""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from classifier_transfer.models import Classifier, DilatedBlock, make_regressor
from classifier_transfer.fast_convolution import interleaved_block_forward
from classifier_transfer.data import digest

CHECKPOINT_SHA = "c6dddf28f8025788ae8dd5348c5607baec7d302a719a71e9d606b86551482d48"


def seed_for(*values):
    return int.from_bytes(hashlib.sha256("|".join(map(str, values)).encode()).digest()[:8], "little")


def dinucleotide_shuffle(sequence, seed):
    """Randomized Euler trail; exact di/mononucleotide counts, not uniform sampling."""
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or not len(sequence) or not np.isin(sequence, range(4)).all():
        raise ValueError("Expected nonempty ACGT codes")
    sequence = sequence.astype(np.uint8)
    rng = np.random.default_rng(seed)
    adjacency = [sequence[1:][sequence[:-1] == i].copy() for i in range(4)]
    for values in adjacency: rng.shuffle(values)
    remaining = [len(v) for v in adjacency]
    stack, trail = [int(sequence[0])], []
    while stack:
        v = stack[-1]
        if remaining[v]:
            remaining[v] -= 1
            stack.append(int(adjacency[v][remaining[v]]))
        else:
            trail.append(stack.pop())
    result = np.asarray(trail[::-1], np.uint8)
    if len(result) != len(sequence) or result[-1] != sequence[-1]:
        raise ValueError("Dinucleotide trail failed")
    return result


def load_model(path, device):
    if digest(path) != CHECKPOINT_SHA: raise ValueError("Selected classifier checksum changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["settings"]["architecture"] != "cnn" or saved["epoch"] != 38
            or saved["settings"]["training"].get("readout", "legacy") != "legacy"):
        raise ValueError("Unexpected classifier architecture")
    model = Classifier(make_regressor("cnn"))
    model.load_state_dict(saved["state_dict"], strict=True)
    DilatedBlock.forward = interleaved_block_forward
    return model.eval().requires_grad_(False).to(device)


def one_hot(codes, device):
    return F.one_hot(torch.as_tensor(np.asarray(codes).copy(), dtype=torch.long, device=device), 4).permute(0, 2, 1).float()


def active_weights(labels):
    if labels.ndim != 2 or labels.shape[1] != 8 or not torch.all((labels == 0) | (labels == 1)) or (labels.sum(1) == 0).any():
        raise ValueError("Observed binary active contexts required")
    return labels.float()/labels.sum(1, keepdim=True)


def ensemble(model, x):
    forward, reverse = model(x), model(x.flip((1, 2)))
    return (forward+reverse)*.5, (forward.sigmoid()+reverse.sigmoid())*.5


def integrated_gradients(model, x, baseline, weights, steps=32, internal_batch=64):
    """Gauss-Legendre integrated gradients, including hypothetical base scores.

    Primary target is mean observed-active-context forward/RC LOGIT. Integration
    covers the full input; cropping attribution for motifs happens afterwards.
    """
    if x.shape != baseline.shape or len(x) != len(weights) or internal_batch < len(x):
        raise ValueError("Unaligned attribution inputs / too-small internal batch")
    nodes, quadrature = np.polynomial.legendre.leggauss(steps)
    alpha = torch.as_tensor((nodes+1)/2, dtype=x.dtype, device=x.device)
    quadrature = torch.as_tensor(quadrature/2, dtype=x.dtype, device=x.device)
    average_gradient = torch.zeros_like(x)
    group = max(1, internal_batch//len(x))
    for begin in range(0, steps, group):
        a = alpha[begin:begin+group]
        points = (baseline[None]+a[:, None, None, None]*(x-baseline)[None]).flatten(0, 1).requires_grad_(True)
        logits, _ = ensemble(model, points)
        target = (logits*weights.repeat(len(a), 1)).sum()
        grad = torch.autograd.grad(target, points)[0].reshape(len(a), *x.shape)
        average_gradient += (grad*quadrature[begin:begin+len(a), None, None, None]).sum(0)
    # A hypothetical base replaces the baseline at just this position.
    hypothetical = average_gradient-(average_gradient*baseline).sum(1, keepdim=True)
    actual = (hypothetical*x).sum(1)
    with torch.no_grad():
        original = (ensemble(model, x)[0]*weights).sum(1)
        reference = (ensemble(model, baseline)[0]*weights).sum(1)
    delta = actual.sum(1)-(original-reference)
    return dict(hypothetical=hypothetical.detach(), actual=actual.detach(),
                delta=delta.detach(), target_difference=(original-reference).detach())


def score_batch(model, data, indices, config, device):
    x = one_hot(data["sequence"][indices], device)
    labels = torch.as_tensor(data["labels"][indices], device=device)
    weights = active_weights(labels)
    with torch.no_grad(): logits, probabilities = ensemble(model, x)
    accumulated, hyp = torch.zeros_like(x[:, 0]), torch.zeros_like(x[:, :, 768:1280])
    deltas, differences, used_steps, passed = [], [], [], []
    for reference in range(config["references"]):
        shuffled = np.stack([dinucleotide_shuffle(data["sequence"][i], seed_for(config["seed"], data["ids"][i], reference)) for i in indices])
        baseline = one_hot(shuffled, device)
        result = integrated_gradients(model, x, baseline, weights, config["steps"], config["internal_batch"])
        steps_used = torch.full((len(indices),), config["steps"], device=device, dtype=torch.int32)
        for steps in (64, 128):
            if steps <= config["steps"]: continue
            tolerance = config["absolute_tolerance"]+config["relative_tolerance"]*result["target_difference"].abs()
            bad = torch.where(result["delta"].abs() > tolerance)[0]
            if not len(bad): break
            refined = integrated_gradients(model, x[bad], baseline[bad], weights[bad], steps, config["internal_batch"])
            for k, value in refined.items(): result[k][bad] = value
            steps_used[bad] = steps
        tolerance = config["absolute_tolerance"]+config["relative_tolerance"]*result["target_difference"].abs()
        passed.append(result["delta"].abs() <= tolerance)
        accumulated += result["actual"]/config["references"]
        hyp += result["hypothetical"][:, :, 768:1280]/config["references"]
        deltas.append(result["delta"]); differences.append(result["target_difference"]); used_steps.append(steps_used)
    values = dict(actual=accumulated, hypothetical_central512=hyp, logits=logits, probabilities=probabilities,
                  delta=torch.stack(deltas, 1), target_difference=torch.stack(differences, 1),
                  steps=torch.stack(used_steps, 1), quality_pass=torch.stack(passed, 1).all(1))
    return {k:v.detach().cpu().numpy() for k,v in values.items()}
