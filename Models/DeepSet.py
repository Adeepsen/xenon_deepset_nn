import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
        "loss": "BCEWithLogitsLoss",
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

for d in [train_df, val_df, test_df]:
    d[FEATURES] = d[FEATURES].astype(np.float32)
    
#normalizing
scaler = StandardScaler()
scaler.fit(train_df[FEATURES])

train_df[FEATURES] = scaler.transform(train_df[FEATURES]).astype(np.float32)
val_df[FEATURES]   = scaler.transform(val_df[FEATURES]).astype(np.float32)
test_df[FEATURES]  = scaler.transform(test_df[FEATURES]).astype(np.float32)

#event dataset
def build_event_groups(df):
    grouped = df.groupby(EVENT_COL).indices
    return list(grouped.values())

class S2EventDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.event_groups = build_event_groups(self.df)

    def __len__(self):
        return len(self.event_groups)

    def __getitem__(self, idx):
        rows = self.event_groups[idx]

        x = self.df.iloc[rows][FEATURES].to_numpy(dtype=np.float32)
        y = self.df.iloc[rows][TARGETS].to_numpy(dtype=np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "n_clusters": len(rows),
        }

#Collate function
def collate_events(batch):
    batch_size = len(batch)
    max_clusters = max(item["n_clusters"] for item in batch)

    x_dim = batch[0]["x"].shape[1]  # should be 5
    y_dim = batch[0]["y"].shape[1]  # should be 2

    x_padded = torch.zeros(batch_size, max_clusters, x_dim)
    y_padded = torch.zeros(batch_size, max_clusters, y_dim)

    mask = torch.zeros(batch_size, max_clusters, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["n_clusters"]

        x_padded[i, :n] = item["x"]
        y_padded[i, :n] = item["y"]

        mask[i, :n] = True

    return {
        "x": x_padded,
        "y": y_padded,
        "mask": mask,
    }


#build datasets
train_dataset = S2EventDataset(train_df)
val_dataset = S2EventDataset(val_df)
test_dataset = S2EventDataset(test_df)

#build dataloaders using wandb batch size 
batch_size = wandb.config.batch_size

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_events,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_events,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_events,
)



class DeepSet(nn.Module):
    def __init__(self, input_dim=5, latent_dim=64, phi_hidden=64, rho_hidden=64, output_dim=2):
        super().__init__()

        # per-cluster encoder phi
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, latent_dim),
            nn.ReLU(),
        )

        # decoder rho: takes cluster embedding + event embedding
        self.rho = nn.Sequential(
            nn.Linear(latent_dim * 2, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, output_dim),
        )

    def forward(self, x, mask):
        """
        x:    (B, K, 5)
        mask: (B, K)  True for real clusters, False for padding
        returns:
               logits: (B, K, 2)
        """
        B, K, _ = x.shape

        # encode each cluster independently
        phi_x = self.phi(x)  # (B, K, latent_dim)

        # zero out padded clusters before summing
        mask_f = mask.unsqueeze(-1).float()  # (B, K, 1)
        phi_x_masked = phi_x * mask_f

        # permutation-invariant event embedding
        event_emb = phi_x_masked.sum(dim=1)  # (B, latent_dim)

        # broadcast event embedding back to every cluster
        event_emb_expanded = event_emb.unsqueeze(1).expand(-1, K, -1)  # (B, K, latent_dim)

        # combine local cluster info with global event context
        h = torch.cat([phi_x_masked, event_emb_expanded], dim=-1)
        # predict main/alt logits per cluster
        logits = self.rho(h)  # (B, K, 2)

        return logits    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepSet(
    input_dim=5,
    latent_dim=wandb.config.latent_dim,
    phi_hidden=wandb.config.phi_hidden,
    rho_hidden=wandb.config.rho_hidden,
    output_dim=2,
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

#make sure loss isnt using padded data
def masked_bce_loss(logits, targets, mask):
    """
    logits:  (B, K, 2)
    targets: (B, K, 2)
    mask:    (B, K)
    """
    loss_per_entry = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )  # (B, K, 2)

    mask_f = mask.unsqueeze(-1).float()  # (B, K, 1)
    loss_per_entry = loss_per_entry * mask_f

    denom = mask_f.sum() * logits.shape[-1]
    return loss_per_entry.sum() / denom.clamp_min(1.0)


def run_epoch(model, loader, optimizer=None):
    """
    If optimizer is provided, runs training.
    If optimizer is None, runs evaluation.
    Returns:
        avg_loss, all_targets, all_probs
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    all_targets = []
    all_probs = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(x, mask)
            loss = masked_bce_loss(logits, y, mask)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)

            # keep only real clusters
            flat_mask = mask.view(-1)
            flat_y = y.view(-1, 2)[flat_mask]
            flat_p = probs.view(-1, 2)[flat_mask]

            all_targets.append(flat_y.detach().cpu())
            all_probs.append(flat_p.detach().cpu())

    avg_loss = total_loss / max(n_batches, 1)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    return avg_loss, all_targets, all_probs


def compute_auc(y_true, y_prob):
    aucs = {}
    for i, name in enumerate(TARGETS):
        try:
            aucs[f"{name}_auc"] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            aucs[f"{name}_auc"] = float("nan")
    return aucs


# training loop
best_val_loss = float("inf")
best_state_dict = None

for epoch in range(wandb.config.epochs):
    train_loss, train_y, train_p = run_epoch(model, train_loader, optimizer=optimizer)
    val_loss, val_y, val_p = run_epoch(model, val_loader, optimizer=None)

    train_auc = compute_auc(train_y, train_p)
    val_auc = compute_auc(val_y, val_p)

    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"train_main_auc={train_auc['p_main_auc']:.4f} | val_main_auc={val_auc['p_main_auc']:.4f} | "
        f"train_alt_auc={train_auc['p_alt_auc']:.4f} | val_alt_auc={val_auc['p_alt_auc']:.4f}"
    )

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_p_main_auc": train_auc["p_main_auc"],
        "train_p_alt_auc": train_auc["p_alt_auc"],
        "val_p_main_auc": val_auc["p_main_auc"],
        "val_p_alt_auc": val_auc["p_alt_auc"],
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# restore best model
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# final test evaluation
test_loss, test_y, test_p = run_epoch(model, test_loader, optimizer=None)
test_auc = compute_auc(test_y, test_p)

print("\nFinal test results")
print(f"test_loss={test_loss:.4f}")
print(f"test_p_main_auc={test_auc['p_main_auc']:.4f}")
print(f"test_p_alt_auc={test_auc['p_alt_auc']:.4f}")

wandb.log({
    "test_loss": test_loss,
    "test_p_main_auc": test_auc["p_main_auc"],
    "test_p_alt_auc": test_auc["p_alt_auc"],
})