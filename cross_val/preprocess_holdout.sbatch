#!/bin/bash
#SBATCH --job-name=stentor_preprocess_holdout
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_holdout_%j.out
#SBATCH --error=logs/preprocess_holdout_%j.err

set -e
cd /n/home06/ktruong/projects/stentor-cnn
module load python
module load cuda/12.2.0-fasrc01
conda activate stentor

python -u cross_val/build_holdout_cache.py
