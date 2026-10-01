#!/usr/bin/env bash
# Site-specific module commands go here, if needed. No environment is copied
# from another machine. Python paths are configured in site.yaml.
# module purge
# module load cuda/11.8
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
