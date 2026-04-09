import pandas as pd
import argparse
import yaml
import warnings
import torch
import sys
import os
from pathlib import Path

# Add src/str to path
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent.parent

from model.STRmodel import ExpSTRmodel
import pickle

warnings.filterwarnings("ignore")

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
    data_features = [x_range, y_range, z_range, [0, 7], [0, 480], [0, 7], [0, 480]]
    return x_range, y_range, z_range, data_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STR")
    parser.add_argument("-C", "--config", type=str, default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("-D", "--dis_type", type=str, default="dtw")
    parser.add_argument("-T", "--traj_num", type=str, default=15000)
    parser.add_argument("-X", "--data", type=str, default="str_porto")
    parser.add_argument("-G", "--gpu", type=str, default="0")
    parser.add_argument("-L", "--load-model", type=str)
    parser.add_argument("-J", "--just_embedding", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config["dis_type"] = args.dis_type
    config["traj_num"] = args.traj_num
    # Ensure values are strings for formatting
    config["data"] = config["data"].format(args.data)
    
    # Resolve all relative paths to project root
    config["traj_path"] = str(PROJECT_ROOT / config["traj_path"].format(config["data"]))
    config["stdis_matrix_path"] = str(PROJECT_ROOT / config["stdis_matrix_path"].format(args.data, config["dis_type"], config["traj_num"]))
    
    config["model_best_wts_path"] = str(PROJECT_ROOT / config["model_best_wts_path"].format(config["data"], config["length"], config["model"], config["dis_type"])) + " {:.4f}.pt"
    config["model_best_topAcc_path"] =  str(PROJECT_ROOT / config["model_best_topAcc_path"].format(config["data"], config["length"], config["model"], config["dis_type"]))
    
    config["embeddings_path"] =  str(PROJECT_ROOT / config["embeddings_path"].format(config["data"], config["length"], config["model"], config["dis_type"]))

    # Auto compute ranges
    x_range, y_range, z_range, data_features = compute_data_ranges(config["traj_path"])
    config["x_range"] = x_range
    config["y_range"] = y_range
    config["z_range"] = z_range
    config["data_features"] = data_features

    # Auto compute train/val splits
    total = int(config["traj_num"])
    train_end = int(total * 0.8)
    config["train_data_range"] = [0, train_end]
    config["val_data_range"] = [train_end, total]

    print("Args in experiment:")
    print(config)
    print("GPU:", args.gpu)
    print("Load model:", args.load_model)
    print("Store embeddings:", args.just_embedding, "\n")

    if args.just_embedding:
        ExpSTRmodel(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).embedding()
    else:
        ExpSTRmodel(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).train()

    torch.cuda.empty_cache()
