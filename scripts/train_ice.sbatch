#!/usr/bin/env bash
# Submit from the repository root. Supply partition/GPU/time according to your allocation.
#SBATCH --job-name=dora-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
srun .venv/bin/python -m dora_spectral.experiment train "$@"
