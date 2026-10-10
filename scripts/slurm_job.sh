#!/bin/bash
# Run one epm command as a single-GPU Slurm job. Adjust the #SBATCH lines to your cluster.
#
#   sbatch scripts/slurm_job.sh epm train-regressor --architecture cnn --resume
#
# --resume makes a resubmitted job continue from last_checkpoint.pt (a new run
# starts normally). Run from the repository root with the `epm` environment set
# in ENV_NAME (default: epm).
#SBATCH --job-name=epm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-epm}"
echo "host $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}: $*"
exec "$@"
