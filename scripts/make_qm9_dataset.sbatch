#!/bin/bash
# The full QM9 dataset as one Slurm job (about 1.5 h on 64 cores). Submit from the repository root.
# Extra arguments go to the script, e.g.
#     sbatch scripts/make_qm9_dataset.sbatch --name qm9s --rotations /path/to/qm9s_rotations.npz
# Finer images (cost ~ pixels^2; measured on 32 cores: 64 px 1 h, 128 px 4 h, 256 px 15 h):
#     sbatch --cpus-per-task=32 --time=20:00:00 --mem=48G scripts/make_qm9_dataset.sbatch --name qm9s_256 --pixels 256 --rotations ...
#SBATCH --job-name=stm_qm9
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1   # one numpy thread per worker
cd "$SLURM_SUBMIT_DIR" && "$PYTHON" scripts/make_qm9_dataset.py --qh9 "$QH9" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK" "$@"
