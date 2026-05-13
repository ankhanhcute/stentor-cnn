"""
"""
from __future__ import annotations 
import torch 
import torch.nn as nn
from model import StentorCNN
import sys 
import os 

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import loader


#------LOADER-----
OUT_DIR = os.path.join(THIS_DIR, 'outputs')
OUT_PNG = os.path.join(THIS_DIR, 'visualize_failure.png')
if len(sys.argv) != 5:
    print("Usage: python visualize_failures.py <tile.h5> <meta.h5> <contractions.h5> <checkpoint.pt>")
    sys.exit(1)
    
TILED_H5 = sys.argv[1]
META_H5 = sys.argv[2]
GT_H5 = sys.argv[3]
CKPT = sys.argv[4]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    
    #load data
    tiles, meta = loader.load_tiles(TILED_H5, META_H5)
    manual = loader.load_manual_labels(GT_H5)
    
    #load model
    model = StentorCNN(in_channels=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()