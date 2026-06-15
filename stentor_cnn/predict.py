"""
predict.py
----------
redict.py
----------
Run a pretrained model on a new unlabeled dataset.
Outputs 0/1/null per stimulus + uncertain stimuli JSON.
 
Usage:
    python predict.py <tiled.h5> <meta.h5> [checkpoint.pt]
 
Example:
    python predict.py ../tiles/2025_01_10_tiled.h5 ../meta/2025_01_10_tiled_data.h5
    
"""
import sys, os, json 
import torch 
from torch.utils.data import DataLoader
import loader
from model import StentorSequenceModel , count_params
from evaluate import predict_all
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

#-----Config----
DROPOUT = 0.3
CHECKPOINT = os.path.join(THIS_DIR, "checkpoints", "best_model.pt")
THRESH_FILE = os.path.join(THIS_DIR, "checkpoints", "best_thresh.json")
OUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

#----parse args----
if len(sys.argv) < 3:
    print("Usage: python predict.py <tiled.h5> <meta.h5> [checkpoint.pt]")
    sys.exit(1)
    
TILED_H5 = sys.argv[1]
META_H5 = sys.argv[2]
if len(sys.argv) > 3:
    CHECKPOINT = sys.argv[3]
    

#----main----
def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device     : {device}")
    print(f"checkpoint : {CHECKPOINT}")
    print(f"dataset    : {TILED_H5}\n")
    #----load threshold-----
    with open(THRESH_FILE) as f:
        best_thresh = json.load(f)["threshold"]
    
    #load model 
    model = StentorSequenceModel(dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    print(f"model params: {count_params(model):,}\n")
    
    # load tiles - no labels, dummy all-NaN manual array
    tiles, meta = loader.load_tiles(TILED_H5, META_H5)
    total_stims = meta["total_stims"]
    manual      = np.full((meta["num_cells"], total_stims), np.nan)
    all_cells   = list(range(meta["num_cells"]))
    
    ds = loader.StentorPairs(tiles, manual, all_cells, tiled_h5_path=TILED_H5)
    print(f"cells loaded: {len(ds)}")
    # run predictions
    all_predictions, uncertain_stimuli = predict_all(model, [ds], device, best_thresh)
 
    n_certain   = sum(1 for p in all_predictions if p["prediction"] is not None)
    n_uncertain = sum(1 for p in all_predictions if p["prediction"] is None)
    print(f"certain     : {n_certain}")
    print(f"uncertain   : {n_uncertain}")
    # save JSONs
    dataset_name = os.path.basename(TILED_H5).replace("_tiled.h5", "")
 
    pred_path = os.path.join(OUT_DIR, f"predictions_{dataset_name}.json")
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\npredictions saved → {pred_path}")
 
    unc_path = os.path.join(OUT_DIR, f"uncertain_{dataset_name}.json")
    with open(unc_path, "w") as f:
        json.dump(uncertain_stimuli, f, indent=2)
    print(f"uncertain saved   → {unc_path}")
 
if __name__ == "__main__":
    main()