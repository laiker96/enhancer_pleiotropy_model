#!/usr/bin/env bash
#SBATCH --job-name=enh_pleio_4x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
umask 002

if [[ $(hostname -s) == neocranex ]]; then
  echo "Refusing to run model work on the login node" >&2
  exit 1
fi

repository=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
config_path=${MODEL_CONFIG:-config/default.yaml}

required=(
  "$repository/.venv/bin/python"
  "$repository/.venv/bin/snakemake"
  "$repository/$config_path"
  "$repository/Snakefile"
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
srun --ntasks=1 .venv/bin/snakemake \
  --configfile "$config_path" \
  --cores "${SLURM_CPUS_PER_TASK:-4}" \
  --rerun-incomplete \
  --resources gpu=1
