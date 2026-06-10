#!/bin/bash
#SBATCH --job-name=stentor_eval_folds
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/eval_fold%a_%j.out
#SBATCH --error=logs/eval_fold%a_%j.err

set -e
cd /n/home06/ktruong/projects/stentor-cnn
module load python
module load cuda/12.2.0-fasrc01
conda activate stentor

FOLD=${SLURM_ARRAY_TASK_ID}
CKPT="cross_val/checkpoints/fold_${FOLD}/best_model.pt"

echo "======================================"
echo "Evaluating fold ${FOLD} checkpoint: ${CKPT}"
echo "======================================"

HOLDOUT=(
    "2025_10_20_03_02_39" "2025_11_02_23_30_22" "2025_11_03_20_45_41"
    "2025_10_30_20_47_23" "2025_11_01_17_58_50" "2025_09_22_02_43_12"
    "2025_04_26_23_21_10" "2025_06_12_22_58_40" "2025_04_14_20_26_34"
    "2025_05_25_00_54_10" "2025_10_29_23_54_44" "2025_10_30_20_46_05"
    "2025_11_02_23_29_03" "2025_11_02_23_31_42" "2025_09_22_02_41_52"
    "2025_09_23_19_20_47" "2025_06_13_23_05_12" "2025_05_30_20_11_44"
    "2025_04_26_23_24_08" "2024_12_29_03_27_19"
)

mkdir -p results/fold_eval/fold_${FOLD}

for NAME in "${HOLDOUT[@]}"; do
    T="data/${NAME}_tiled.h5"
    M="data/${NAME}_tiled_data.h5"
    G="data/${NAME}_contractions.h5"
    if [[ -f "$T" && -f "$M" && -f "$G" ]]; then
        echo "--- ${NAME} ---"
        python -u stentor_cnn/evaluate.py "$T" "$M" "$G" "$CKPT" \
            2>&1 | tee -a results/fold_eval/fold_${FOLD}/${NAME}.txt
    else
        echo "SKIP ${NAME} (missing files)"
    fi
done

echo "======================================"
echo "Fold ${FOLD} eval done at: $(date)"
echo "======================================"
