#!/bin/bash
#SBATCH --job-name=stentor_cv
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0
#SBATCH --output=logs/cv_fold%a_%j.out
#SBATCH --error=logs/cv_fold%a_%j.err

set -e
cd /n/home06/ktruong/projects/stentor-cnn
mkdir -p logs results/fold_results cross_val/checkpoints

echo "======================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID}  Fold: ${SLURM_ARRAY_TASK_ID}"
echo "======================================"

module load python
module load cuda/12.2.0-fasrc01
source activate stentor

python -c "import torch; print('cuda:', torch.cuda.is_available())"

python -u cross_val/train_fold.py \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --k 5 \
    --epochs 200 \
    --seed 42 \
    --margin 0.2

echo "Fold ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
