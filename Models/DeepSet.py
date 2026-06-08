import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


FEATURES = [
    "x",
    "y",
    "n_electrons_interface",
    "drift_time_mean",
    "drift_time_spread",
]
TARGETS = ["p_main", "p_alt"]
EVENT_COL = "event_number" #splitting by event


event_ids = df[EVENT_COL].unique()

#70% train 15% val 15% test split
train_events, temp_events = train_test_split(
    event_ids,
    test_size=0.30,
    random_state=42,
    shuffle=True,
)

val_events, test_events = train_test_split(
    temp_events,
    test_size=0.50,
    random_state=42,
    shuffle=True,
)

train_df = df[df[EVENT_COL].isin(train_events)].copy()
val_df   = df[df[EVENT_COL].isin(val_events)].copy()
test_df  = df[df[EVENT_COL].isin(test_events)].copy()

print(len(train_df), len(val_df), len(test_df))

#normalizing
scaler = StandardScaler()
scaler.fit(train_df[FEATURES])

train_df.loc[:, FEATURES] = scaler.transform(train_df[FEATURES])
val_df.loc[:, FEATURES]   = scaler.transform(val_df[FEATURES])
test_df.loc[:, FEATURES]  = scaler.transform(test_df[FEATURES])

