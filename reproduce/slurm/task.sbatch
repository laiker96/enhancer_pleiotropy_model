#!/usr/bin/env bash
# Resources, log paths and dependencies are supplied by submit.py.
set -euo pipefail
umask 077
[[ $# -ge 5 && -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || exit 2
project_root=$1
python_executable=$2
environment_script=$3
stage=$4
config=$5
shift 5
cd "$project_root"
source "$environment_script"
export PYTHONPATH="$project_root/src:$project_root/experiments:$project_root/scripts"
# CPU jobs must never opportunistically use a GPU on a mixed compute node.
case "$stage" in
  train-regressor|train-classifier|evaluate|attribute-pilot|attribute) ;;
  *) export CUDA_VISIBLE_DEVICES='' ;;
esac
exec "$python_executable" -u scripts/reproduce.py "$stage" --config "$config" "$@"
