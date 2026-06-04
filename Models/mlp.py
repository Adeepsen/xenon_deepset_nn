import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb
import matplotlib.pyplot as plt

# W&B setup
wandb.init(
    project="xenon-mlp",
    config={
        "top13_ns": 192_600,
        "batch_size": 4096,
        "epochs": 10,
        "learning_rate": 1e-3,
        "hidden_size": 64,
        "loss": "MSELoss",
        "model_type": "per_cluster_mlp",
        "eval_event_sample_size": 50000,  # smaller sample for faster evaluation
    }
)

# data cleaning
data = np.load("/Users/adeepsen/xenon_deepset_nn/data/s2_tag_training_clusters.npy")
df = pd.DataFrame(data)

top13_ns = 192_600  # ~13 cm using 0.675 mm/us drift velocity

event_min_drift = df.groupby("event_number")["drift_time_mean"].min()
bad_event_ids = event_min_drift[event_min_drift < top13_ns].index.to_numpy()

df = df[~df["event_number"].isin(bad_event_ids)].copy()

df["p_alt"] = df["p_alt"].clip(0, 1)  # in case somehow theres p_alt > 1 values; there shouldn't be any after the 13 cm filtering

print("Rows after fiducial cut:", len(df))
print("Remaining p_alt > 1:", (df["p_alt"] > 1).sum())

# train/val/test split by event
event_ids = df["event_number"].unique()  # getting all unique event ids

# refer to sklearn libraries for better explanations
train_ids, temp_ids = train_test_split(
    event_ids,
    test_size=0.3,
    random_state=42
)  # 30% of event ids go into temp_ids, the other 70% train, random state is setting the seed

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.5,
    random_state=42
)  # splits the 30% that are in temp_ids further, 50% test and 50% validate

# constructing data frames using the saved ids for each group
train_df = df[df["event_number"].isin(train_ids)].copy()
val_df   = df[df["event_number"].isin(val_ids)].copy()
test_df  = df[df["event_number"].isin(test_ids)].copy()

print("Train rows:", len(train_df))
print("Val rows:", len(val_df))
print("Test rows:", len(test_df))

# setting features and targets
feature_cols = [
    "x",
    "y",
    "n_electrons_interface",
    "drift_time_mean",
    "drift_time_spread",
]
target_cols = ["p_main", "p_alt"]

# standardizing input scale so that algorithm isn't confused by units
# fit only on train data, then transform val/test with train statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].to_numpy())
X_val   = scaler.transform(val_df[feature_cols].to_numpy())
X_test  = scaler.transform(test_df[feature_cols].to_numpy())

y_train = train_df[target_cols].to_numpy(dtype=np.float32)
y_val   = val_df[target_cols].to_numpy(dtype=np.float32)
y_test  = test_df[target_cols].to_numpy(dtype=np.float32)

# Dataset
# this tells pytorch how many examples there are and how to fetch one at a time
# return items have form (feature vector, target vector)
class ClusterDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# data loaders
train_loader = DataLoader(
    ClusterDataset(X_train, y_train),
    batch_size=4096,
    shuffle=True
)
val_loader = DataLoader(
    ClusterDataset(X_val, y_val),
    batch_size=4096,
    shuffle=False
)
test_loader = DataLoader(
    ClusterDataset(X_test, y_test),
    batch_size=4096,
    shuffle=False
)

# The actual model:
class MLP(nn.Module):
    def __init__(self, in_dim=5, hidden=64, out_dim=2):  # 5 num in, 2 hidden layers with 64 neurons, 2 num out
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick gpus if possible
print("Using device:", device)

model = MLP().to(device)

criterion = nn.MSELoss()  # use mse loss function, could use BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# Training
def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            preds = model(Xb)
            loss = criterion(preds, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(Xb)

    return total_loss / len(loader.dataset)

num_epochs = wandb.config.epochs
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_lr,
    })

    print(f"Epoch {epoch+1:02d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.2e}")

# validation plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="train")
plt.plot(range(1, num_epochs + 1), val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.tight_layout()
plt.show()

# smaller evaluation sample
eval_event_sample_size = wandb.config.eval_event_sample_size

test_event_ids = test_df["event_number"].unique()

if len(test_event_ids) > eval_event_sample_size:
    sampled_event_ids = np.random.choice(
        test_event_ids,
        size=eval_event_sample_size,
        replace=False
    )
else:
    sampled_event_ids = test_event_ids

test_eval_df = test_df[test_df["event_number"].isin(sampled_event_ids)].copy()

print("Eval events:", len(np.unique(test_eval_df["event_number"])))
print("Eval rows:", len(test_eval_df))

# test metrics on sample
# make sure the sampled test dataframe and prediction order match
test_eval_df = test_eval_df.reset_index(drop=True)

X_eval = scaler.transform(test_eval_df[feature_cols].to_numpy())
y_eval = test_eval_df[target_cols].to_numpy(dtype=np.float32)

eval_loader = DataLoader(
    ClusterDataset(X_eval, y_eval),
    batch_size=4096,
    shuffle=False
)

model.eval()
all_probs = []
all_true = []

with torch.no_grad():
    for Xb, yb in eval_loader:
        Xb = Xb.to(device)
        probs = model(Xb).cpu().numpy()
        all_probs.append(probs)
        all_true.append(yb.numpy())

all_probs = np.vstack(all_probs)
all_true = np.vstack(all_true)

# test MSE
test_mse = np.mean((all_probs - all_true) ** 2)
print("Test MSE:", test_mse)

# per-cluster ROC-AUC
auc_main = roc_auc_score((all_true[:, 0] > 0).astype(int), all_probs[:, 0])
auc_alt = roc_auc_score((all_true[:, 1] > 0).astype(int), all_probs[:, 1])

print("Test p_main ROC-AUC:", auc_main)
print("Test p_alt ROC-AUC:", auc_alt)

# event-level main-cluster accuracy using numpy blocks instead of pandas groupby
event_numbers = test_eval_df["event_number"].to_numpy()
true_p_main = all_true[:, 0]
pred_p_main = all_probs[:, 0]

order = np.argsort(event_numbers, kind="mergesort")
event_numbers = event_numbers[order]
true_p_main = true_p_main[order]
pred_p_main = pred_p_main[order]

boundaries = np.flatnonzero(event_numbers[1:] != event_numbers[:-1]) + 1
blocks = np.split(np.arange(len(event_numbers)), boundaries)

event_correct = 0
num_events = 0

for idx in blocks:
    true_main_idx = np.argmax(true_p_main[idx])
    pred_main_idx = np.argmax(pred_p_main[idx])

    if true_main_idx == pred_main_idx:
        event_correct += 1

    num_events += 1

event_acc = event_correct / num_events
print("Event-level main accuracy:", event_acc)

# log final metrics to W&B
wandb.log({
    "test_mse": test_mse,
    "test_p_main_auc": auc_main,
    "test_p_alt_auc": auc_alt,
    "event_level_main_accuracy": event_acc,
    "eval_events": num_events,
    "eval_rows": len(test_eval_df),
})

wandb.finish()