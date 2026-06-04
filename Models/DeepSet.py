import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import pandas as pd

#wandb config
wandb.init(
    project="xenon-deepset",
    config={
        "top13_ns": 192_600,
        "batch_size": 512,
        "epochs": 10,
        "learning_rate": 1e-3,
        "latent_dim": 64,#defining L
        "phi_hidden": 64,
        "rho_hidden": 64,
        "loss": "BCELoss",
        "model_type": "deepset",
    }
)

#data cleaning
data = np.load("/Users/adeepsen/xenon_deepset_nn/data/s2_tag_training_clusters.npy")
df = pd.DataFrame(data)

top13_ns = 192_600

event_min_drift = df.groupby("event_number")["drift_time_mean"].min()
bad_event_ids = event_min_drift[event_min_drift < top13_ns].index.to_numpy()

df = df[~df["event_number"].isin(bad_event_ids)].copy()

df["p_alt"] = df["p_alt"].clip(0, 1)

print("Rows after fiducial cut:", len(df))
print("Remaining p_alt > 1:", (df["p_alt"] > 1).sum())
