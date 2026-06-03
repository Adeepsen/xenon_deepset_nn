import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#data cleaning
data = np.load("/Users/adeepsen/xenon_deepset_nn/data/s2_tag_training_clusters.npy")
df = pd.DataFrame(data)

top13_ns = 192_600  # ~13 cm using 0.675 mm/us drift velocity

event_min_drift = df.groupby("event_number")["drift_time_mean"].min()
bad_event_ids = event_min_drift[event_min_drift < top13_ns].index.to_numpy()

df = df[~df["event_number"].isin(bad_event_ids)].copy()

df["p_alt"] = df["p_alt"].clip(0, 1) # in case somehow theres p_alt > 1 values; there shouldn't be any after the 13 cm filtering

print("Rows after fiducial cut:", len(df))
print("Remaining p_alt > 1:", (df["p_alt"] > 1).sum())

event_ids = df["event_number"].unique() # getting all unique event ids

#refer to sklearn libraries for better explanations
train_ids, temp_ids = train_test_split(event_ids, test_size=0.3, random_state=42)  # 30% of event ids go into temp_ids, the other 70% train, random state is setting the seed
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)  # splits the 30% that are in temp_ids further, 50% test and 50% validate

#constructing data frames using the saved ids for each group
train_df = df[df["event_number"].isin(train_ids)]
val_df   = df[df["event_number"].isin(val_ids)]
test_df  = df[df["event_number"].isin(test_ids)]

#setting features and targets
feature_cols = [
    "x",
    "y",
    "n_electrons_interface",
    "drift_time_mean",
    "drift_time_spread",
]
target_cols = ["p_main", "p_alt"]

#standardizing input scale so that algorithm isn't confused by units
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].to_numpy())
X_val   = scaler.transform(val_df[feature_cols].to_numpy())
X_test  = scaler.transform(test_df[feature_cols].to_numpy())

y_train = train_df[target_cols].to_numpy(dtype=np.float32)
y_val   = val_df[target_cols].to_numpy(dtype=np.float32)
y_test  = test_df[target_cols].to_numpy(dtype=np.float32)

#this tells pytorch how many examples there are and how to fetch one at a time
#return items have form (feature vector, target vector)
class ClusterDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#data loaders
train_loader = DataLoader(ClusterDataset(X_train, y_train), batch_size=4096, shuffle=True)
val_loader   = DataLoader(ClusterDataset(X_val, y_val), batch_size=4096, shuffle=False)
test_loader  = DataLoader(ClusterDataset(X_test, y_test), batch_size=4096, shuffle=False)

#The actual model:
class MLP(nn.Module):
    def __init__(self, in_dim=5, hidden=64, out_dim=2): # 5 num in, 2 hidden layers with 64 neurons, 2 num out
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pick gpus if possible
model = MLP().to(device)

criterion = nn.MSELoss()  # use mse loss function, could use BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Training
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

for epoch in range(10):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    print(f"Epoch {epoch+1:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

#test metrics
model.eval()
all_probs = []
all_true = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        probs = model(Xb).cpu().numpy()
        all_probs.append(probs)
        all_true.append(yb.numpy())

all_probs = np.vstack(all_probs)
all_true = np.vstack(all_true)

#test MSE
test_mse = np.mean((all_probs - all_true) ** 2)
print("Test MSE:", test_mse)

#per-cluster ROC-AUC
auc_main = roc_auc_score((all_true[:, 0] > 0).astype(int), all_probs[:, 0])
auc_alt = roc_auc_score((all_true[:, 1] > 0).astype(int), all_probs[:, 1])

print("Test p_main ROC-AUC:", auc_main)
print("Test p_alt ROC-AUC:", auc_alt)

#event-level main-cluster accuracy
test_eval = test_df.copy()
test_eval["pred_p_main"] = all_probs[:, 0]

event_correct = 0
num_events = 0

for event_id, event in test_eval.groupby("event_number"):
    true_main_idx = event["p_main"].to_numpy().argmax()
    pred_main_idx = event["pred_p_main"].to_numpy().argmax()

    if true_main_idx == pred_main_idx:
        event_correct += 1

    num_events += 1

print("Event-level main accuracy:", event_correct / num_events)