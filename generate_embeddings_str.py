#!/usr/bin/env python3
"""
Generate trajectory embeddings using a trained STR model.

Usage:
    python scripts/generate_embeddings_str.py --dataset parquet_trajectories_1000
"""

import argparse
import numpy as np
import torch
import sys
import os
import yaml
import pickle
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STR_DIR = PROJECT_ROOT / "src" / "str"

# Add STR to path for imports
if str(STR_DIR) not in sys.path:
    sys.path.insert(0, str(STR_DIR))
if str(STR_DIR / "model") not in sys.path:
    sys.path.insert(0, str(STR_DIR / "model"))

from model.STRmodel import ExpSTRmodel

def compute_data_ranges(traj_path):
    with open(traj_path, "rb") as f:
        trajs = pickle.load(f)
        
    x_coords, y_coords, t_coords = [], [], []
    for traj in trajs:
        for pt in traj:
            x_coords.append(pt[0])
            y_coords.append(pt[1])
            t_coords.append(pt[2])
            
    x_range = [min(x_coords), max(x_coords)]
    y_range = [min(y_coords), max(y_coords)]
    z_range = [min(t_coords), max(t_coords)]
    # data_features needed by STR normalize logic
    data_features = [x_range, y_range, z_range, [0, 7], [0, 480], [0, 7], [0, 480]]
    return x_range, y_range, z_range, data_features

def main():
    parser = argparse.ArgumentParser(description='Generate STR embeddings')
    parser.add_argument('--dataset', required=True, help='Dataset prefix (e.g. parquet_trajectories_1000)')
    parser.add_argument('--gpu', type=str, default="0", help='GPU ID')
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed"
    n_str = args.dataset.split('_')[-1]
    
    output_npy_path = processed_dir / f"{args.dataset}_str_embeddings.npy"
    
    # Locate configuration
    config_path = STR_DIR / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config["data"] = f"str_porto_{n_str}"
    config["traj_num"] = int(n_str)
    
    # Replace format strings with absolute paths
    config["traj_path"] = str(processed_dir / f"{config['data']}_st.pkl")
    config["stdis_matrix_path"] = str(processed_dir / f"{config['data']}_dtw_st_distance_all_{config['traj_num']}.pkl")
    
    # Auto-find the best STR model
    snapshot_dir = STR_DIR / "exp" / "snapshots"
    model_candidates = list(snapshot_dir.glob(f"{config['data']}*dtw*pt"))
    
    if not model_candidates:
        print(f"WARNING: No trained STR model found for {config['data']}.")
        print("Using untrained randomly initialized model for testing.")
        load_model = None
    else:
        load_model = str(model_candidates[-1])  # Take latest or best
        print(f"Using trained model: {load_model}")

    # Compute ranges dynamically
    x_range, y_range, z_range, data_features = compute_data_ranges(config["traj_path"])
    config["x_range"] = x_range
    config["y_range"] = y_range
    config["z_range"] = z_range
    config["data_features"] = data_features
    
    # Output file to match TSMini's logic
    # Set it via config so ExpSTRmodel can write it out inside embedding()
    config["embeddings_path"] = str(processed_dir / f"{config['data']}_embeddings_vec.pkl")

    # Run embedding logic
    print("Initializing STR embedding generation (this includes building the Octree)...")
    exp = ExpSTRmodel(config=config, gpu_id=args.gpu, load_model=load_model, just_embeddings=True)
    exp.embedding()
    
    # The ExpSTRmodel explicitly saves the vectors using LoadSave to embeddings_path
    # We load them and save them to .npy so pipeline can use them
    with open(config["embeddings_path"], "rb") as f:
        embeddings_tensor = pickle.load(f)
        
    embeddings_np = embeddings_tensor.detach().cpu().numpy()
    np.save(output_npy_path, embeddings_np)
    print(f"✅ SUCCESS: Saved {embeddings_np.shape} embeddings to {output_npy_path}")

if __name__ == '__main__':
    main()
