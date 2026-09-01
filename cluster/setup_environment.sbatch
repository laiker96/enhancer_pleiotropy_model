#!/usr/bin/env bash
#SBATCH --job-name=enh_pleio_env
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=nodo9
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
umask 002

if [[ $(hostname -s) == neocranex ]]; then
  echo "Refusing to install packages on the login node" >&2
  exit 1
fi

repository=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
source_environment=/grupos/Lucai/ANTARKA/cCREs/.venv
micromamba=/home/ilaiker/.local/bin/micromamba

cd "$repository"
mkdir -p logs
if [[ ! -x .venv/bin/python ]]; then
  "$micromamba" create --yes --prefix "$repository/.venv" --clone "$source_environment"
fi
.venv/bin/python -m pip install --no-deps --editable .
.venv/bin/python -c 'import numpy, torch, yaml; print(torch.__version__, torch.cuda.is_available())'
