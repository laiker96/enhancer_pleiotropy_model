"""Compute-node entry point for the frozen two-target occurrence analysis."""
import argparse
import os
from pathlib import Path
import socket

from . import motif_occurrence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('setup','scan','pilot','full','report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int, choices=(0,1), default=0)
    args = parser.parse_args()
    host = socket.gethostname().split('.')[0]
    gpu = args.stage in ('pilot','full')
    allowed = host in ('a100','xg05','xg07','xg08','xg09') if gpu else host.startswith('n') and host[1:].isdigit()
    if not os.environ.get('SLURM_JOB_ID') or not allowed:
        raise RuntimeError('Refusing execution outside the requested CECAR compute allocation')
    if gpu:
        import torch
        from . import finemo_native
        if not torch.cuda.is_available(): raise RuntimeError('Allocated CUDA device unavailable')
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.stage == 'pilot': finemo_native.pilot(args.root, 'cuda')
        else: finemo_native.full(args.root)
    elif args.stage == 'setup': motif_occurrence.setup(args.root)
    elif args.stage == 'scan': motif_occurrence.run_scan(args.root,args.task)
    else:
        from .occurrence_finemo_report import run
        run(args.root)


if __name__ == '__main__': main()
