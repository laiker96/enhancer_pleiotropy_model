#!/usr/bin/env bash
#SBATCH --job-name=enh_crested4x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --exclude=nodo9
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
umask 002

if [[ $(hostname -s) == neocranex ]]; then
  echo "Refusing to run model work on the login node" >&2
  exit 1
fi

repository=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
config_path=results/final_4x_chr2l_halfval_chr3r_test/run_config.yaml

required=(
  "$repository/.venv/bin/python"
  "$repository/$config_path"
  "$repository/results/final_4x_chr2l_halfval_chr3r_test/data/windows.tsv.gz"
  "$repository/results/final_4x_chr2l_halfval_chr3r_test/data/profiles/atac/train_profiles.npy"
  "$repository/results/final_4x_chr2l_halfval_chr3r_test/data/profiles/atac/validation_profiles.npy"
  "$repository/results/final_4x_chr2l_halfval_chr3r_test/data/profiles/h3k27ac/train_profiles.npy"
  "$repository/results/final_4x_chr2l_halfval_chr3r_test/data/profiles/h3k27ac/validation_profiles.npy"
)
for path in "${required[@]}"; do
  if [[ ! -s $path ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

cd "$repository"
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --ntasks=1 .venv/bin/python -m enhancer_pleiotropy_model.training \
  --config "$config_path" \
  --resume
