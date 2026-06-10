#!/bin/bash
#SBATCH --job-name=stentor_preprocess
#SBATCH --account=gershman_lab
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

set -e
cd /n/home06/ktruong/projects/stentor-cnn
module load python
module load cuda/12.2.0-fasrc01
conda activate stentor

python -u cross_val/build_cache.py
