"""Explicit cloud opt-in; never turn a cluster/login host into a cloud worker."""

import os
from pathlib import Path
import re
import socket


def cluster_host(host):
    return host in {"neocranex", "sauron"} or host.startswith("nodo")


def execution_backend():
    backend = os.environ.get("ENHANCER_EXECUTION_BACKEND", "slurm")
    if backend not in {"slurm", "runpod"}:
        raise RuntimeError("Unknown execution backend")
    return backend


def require_runpod():
    host = socket.gethostname().split(".")[0]
    if cluster_host(host) or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("RunPod opt-in is forbidden on cluster hosts/allocations")
    pod = os.environ.get("RUNPOD_POD_ID", "")
    if (execution_backend() != "runpod" or not re.fullmatch(r"[A-Za-z0-9_-]+", pod)
            or pod != os.environ.get("ENHANCER_APPROVED_POD_ID")):
        raise RuntimeError("RunPod requires an explicit matching approved Pod ID")
    workspace = Path("/workspace")
    if not workspace.is_mount() or not Path.cwd().resolve().is_relative_to(workspace):
        raise RuntimeError("RunPod work must stay on the persistent /workspace mount")
    return pod


def cloud_provenance():
    return {"execution_backend": "runpod", "pod_id": require_runpod()} if execution_backend() == "runpod" else {}


if __name__ == "__main__":
    require_runpod()
