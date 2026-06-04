#!/bin/bash
#SBATCH --job-name=stentor_mix20
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stentor_%j.out
#SBATCH --error=logs/stentor_%j.err

set -e

cd /n/home06/ktruong/projects/stentor-cnn
mkdir -p logs outputs

echo "======================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Repo folder: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "======================================"

echo "Loading environment..."
module load python
module load cuda/12.2.0-fasrc01
source activate stentor

echo "Checking Python:"
which python
python --version

echo "Checking CUDA:"
python -c "import torch; print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

echo "Starting ONE mixed training run with 20 selected datasets..."

python -u stentor_cnn/train.py \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_02_39_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_02_39_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_02_39_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_03_58_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_03_58_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_03_58_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_05_18_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_05_18_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_20_03_05_18_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_10_23_01_04_20_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_23_01_04_20_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_23_01_04_20_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_10_28_02_06_46_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_28_02_06_46_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_28_02_06_46_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_10_29_23_53_24_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_29_23_53_24_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_10_29_23_53_24_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_11_01_17_57_30_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_01_17_57_30_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_01_17_57_30_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_11_02_23_30_22_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_02_23_30_22_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_02_23_30_22_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_11_03_20_47_00_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_03_20_47_00_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_03_20_47_00_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_11_03_20_48_20_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_03_20_48_20_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_03_20_48_20_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_11_04_21_38_54_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_04_21_38_54_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_11_04_21_38_54_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_09_12_01_32_16_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_09_12_01_32_16_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_09_12_01_32_16_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2024_12_27_05_56_43_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_12_27_05_56_43_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_12_27_05_56_43_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_09_12_01_33_36_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_09_12_01_33_36_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_09_12_01_33_36_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_09_13_19_33_32_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_09_13_19_33_32_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_09_13_19_33_32_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2024_11_07_02_09_48_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_11_07_02_09_48_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_11_07_02_09_48_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2024_12_11_03_59_55_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_12_11_03_59_55_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_12_11_03_59_55_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2024_12_27_05_58_03_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_12_27_05_58_03_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2024_12_27_05_58_03_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_06_10_02_53_54_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_06_10_02_53_54_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_06_10_02_53_54_contractions.h5 \
  /n/home06/ktruong/projects/stentor-cnn/data/2025_04_13_00_28_31_tiled.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_04_13_00_28_31_tiled_data.h5 /n/home06/ktruong/projects/stentor-cnn/data/2025_04_13_00_28_31_contractions.h5

echo "======================================"
echo "Mixed 20-dataset training finished at: $(date)"
echo "======================================"
