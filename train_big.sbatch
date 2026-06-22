#!/bin/bash
#SBATCH --job-name=stentor_big
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/stentor_%j.out
#SBATCH --error=logs/stentor_%j.err

set -e
cd /n/home06/ktruong/projects/stentor-cnn
mkdir -p logs outputs

echo "======================================"
echo "Job started at: $(date)"
echo "======================================"

module load python
module load cuda/12.2.0-fasrc01
source activate stentor

python -u stentor_cnn/train.py $(python3 -c "
import os
data_dir = 'data'
holdout = {
    '2025_10_20_03_02_39', '2025_11_02_23_30_22', '2025_11_03_20_45_41',
    '2025_10_30_20_47_23', '2025_11_01_17_58_50', '2025_09_22_02_43_12',
    '2025_04_26_23_21_10', '2025_06_12_22_58_40', '2025_04_14_20_26_34',
    '2025_05_25_00_54_10', '2025_10_29_23_54_44', '2025_10_30_20_46_05',
    '2025_11_02_23_29_03', '2025_11_02_23_31_42', '2025_09_22_02_41_52',
    '2025_09_23_19_20_47', '2025_06_13_23_05_12', '2025_05_30_20_11_44',
    '2025_04_26_23_24_08', '2024_12_29_03_27_19',
    '2024_11_10_11_53_54', '2026_05_05_17_39_45'
}
datasets = []
for f in sorted(os.listdir(data_dir)):
    if f.endswith('_tiled.h5'):
        name = f.replace('_tiled.h5', '')
        if name not in holdout:
            t = f'data/{name}_tiled.h5'
            m = f'data/{name}_tiled_data.h5'
            c = f'data/{name}_contractions.h5'
            if os.path.exists(t) and os.path.exists(m) and os.path.exists(c):
                datasets.append(f'{t} {m} {c}')
print(' '.join(datasets))
")

echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"
