#!/usr/bin/env bash
# Pass partition, GPU type, and walltime to sbatch.
#SBATCH --job-name=dora-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
test -x .venv/bin/python || { echo 'Run setup_pace.sh first.' >&2; exit 1; }
mkdir -p "runs/ice-smoke-${SLURM_JOB_ID}"
git rev-parse HEAD > "runs/ice-smoke-${SLURM_JOB_ID}/git-commit.txt"
git diff > "runs/ice-smoke-${SLURM_JOB_ID}/working-tree.patch"
nvidia-smi > "runs/ice-smoke-${SLURM_JOB_ID}/nvidia-smi.txt"
srun .venv/bin/python -m dora_spectral.smoke \
  --config configs/smoke.yaml --device cuda \
  --output "runs/ice-smoke-${SLURM_JOB_ID}/smoke.json"
