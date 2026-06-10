import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# -----------------------
# speed / GPU settings
# -----------------------
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# W&B config
# -----------------------
wandb.init(
    project="xenon-deepset",
    config={
        "top13_ns": 192_600,
        "batch_size": 512,
        "epochs": 100,
        "learning_rate": 1e-3,
        "latent_dim": 128,
        "phi_hidden": 64,
        "rho_hidden": 64,
        "loss": "BCEWithLogitsLoss",
        "model_type": "deepset",
        "num_workers": 4,
    }
)

CACHE_FILE = "deepset_processed_data.npz"
FEATURES = [
    "x",
    "y",
    "n_electrons_interface",
    "drift_time_mean",
    "drift_time_spread",
]
TARGETS = ["p_main", "p_alt"]
EVENT_COL = "event_number"

# -----------------------
# cache load / preprocess
# -----------------------
if os.path.exists(CACHE_FILE):
    print(f"Loading cached data from {CACHE_FILE}")
    cached = np.load(CACHE_FILE, allow_pickle=True)

    X_train = cached["X_train"]
    Y_train = cached["Y_train"]
    E_train = cached["E_train"]
    train_groups = [np.asarray(g, dtype=np.int64) for g in cached["train_groups"]]

    X_val = cached["X_val"]
    Y_val = cached["Y_val"]
    E_val = cached["E_val"]
    val_groups = [np.asarray(g, dtype=np.int64) for g in cached["val_groups"]]

    X_test = cached["X_test"]
    Y_test = cached["Y_test"]
    E_test = cached["E_test"]
    test_groups = [np.asarray(g, dtype=np.int64) for g in cached["test_groups"]]

    print("Cache loaded successfully")

else:
    print("No cache found. Running preprocessing...")

    data = np.load("/home/adeeps/projects/xenon_deepset_nn/data/s2_tag_training_clusters.npy")
    df = pd.DataFrame(data)

    top13_ns = 192_600
    event_min_drift = df.groupby(EVENT_COL)["drift_time_mean"].min()
    bad_event_ids = event_min_drift[event_min_drift < top13_ns].index.to_numpy()

    df = df[~df[EVENT_COL].isin(bad_event_ids)].copy()
    df["p_alt"] = df["p_alt"].clip(0, 1)

    print("Rows after fiducial cut:", len(df))
    print("Remaining p_alt > 1:", (df["p_alt"] > 1).sum())

    event_ids = df[EVENT_COL].unique()

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

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])

    train_df[FEATURES] = scaler.transform(train_df[FEATURES]).astype(np.float32)
    val_df[FEATURES]   = scaler.transform(val_df[FEATURES]).astype(np.float32)
    test_df[FEATURES]  = scaler.transform(test_df[FEATURES]).astype(np.float32)

    X_train = train_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    Y_train = train_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
    E_train = train_df[EVENT_COL].to_numpy(copy=True)

    X_val = val_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    Y_val = val_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
    E_val = val_df[EVENT_COL].to_numpy(copy=True)

    X_test = test_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    Y_test = test_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
    E_test = test_df[EVENT_COL].to_numpy(copy=True)

    def build_event_groups(event_ids_array):
        order = np.argsort(event_ids_array, kind="mergesort")
        sorted_events = event_ids_array[order]
        boundaries = np.flatnonzero(sorted_events[1:] != sorted_events[:-1]) + 1
        return np.split(order, boundaries)

    train_groups = build_event_groups(E_train)
    val_groups   = build_event_groups(E_val)
    test_groups  = build_event_groups(E_test)

    np.savez_compressed(
        CACHE_FILE,
        X_train=X_train,
        Y_train=Y_train,
        E_train=E_train,
        train_groups=np.array(train_groups, dtype=object),
        X_val=X_val,
        Y_val=Y_val,
        E_val=E_val,
        val_groups=np.array(val_groups, dtype=object),
        X_test=X_test,
        Y_test=Y_test,
        E_test=E_test,
        test_groups=np.array(test_groups, dtype=object),
    )
    print(f"Saved cache to {CACHE_FILE}")

# -----------------------
# dataset
# -----------------------
class S2EventDataset(Dataset):
    def __init__(self, X, Y, event_groups):
        self.X = X
        self.Y = Y
        self.event_groups = event_groups

    def __len__(self):
        return len(self.event_groups)

    def __getitem__(self, idx):
        rows = self.event_groups[idx]
        x = self.X[rows]
        y = self.Y[rows]
        return {
            "x": torch.from_numpy(x.copy()),
            "y": torch.from_numpy(y.copy()),
            "n_clusters": len(rows),
        }

def collate_events(batch):
    batch_size = len(batch)
    max_clusters = max(item["n_clusters"] for item in batch)

    x_dim = batch[0]["x"].shape[1]
    y_dim = batch[0]["y"].shape[1]

    x_padded = torch.zeros(batch_size, max_clusters, x_dim, dtype=torch.float32)
    y_padded = torch.zeros(batch_size, max_clusters, y_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_clusters, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["n_clusters"]
        x_padded[i, :n] = item["x"]
        y_padded[i, :n] = item["y"]
        mask[i, :n] = True

    return {"x": x_padded, "y": y_padded, "mask": mask}

train_dataset = S2EventDataset(X_train, Y_train, train_groups)
val_dataset   = S2EventDataset(X_val, Y_val, val_groups)
test_dataset  = S2EventDataset(X_test, Y_test, test_groups)

batch_size = wandb.config.batch_size
num_workers = wandb.config.num_workers

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_events,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_events,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_events,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
)

# -----------------------
# model
# -----------------------
class DeepSet(nn.Module):
    def __init__(self, input_dim=5, latent_dim=64, phi_hidden=64, rho_hidden=64, output_dim=2):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, latent_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(latent_dim * 2, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, output_dim),
        )

    def forward(self, x, mask):
        B, K, _ = x.shape

        phi_x = self.phi(x)  # (B, K, latent_dim)

        mask_f = mask.unsqueeze(-1).float()
        phi_x_masked = phi_x * mask_f

        event_emb = phi_x_masked.sum(dim=1)  # (B, latent_dim)
        event_emb_expanded = event_emb.unsqueeze(1).expand(-1, K, -1)

        h = torch.cat([phi_x_masked, event_emb_expanded], dim=-1)
        logits = self.rho(h)  # (B, K, 2)

        return logits

model = DeepSet(
    input_dim=5,
    latent_dim=wandb.config.latent_dim,
    phi_hidden=wandb.config.phi_hidden,
    rho_hidden=wandb.config.rho_hidden,
    output_dim=2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# -----------------------
# loss
# -----------------------
def masked_bce_loss(logits, targets, mask):
    loss_per_entry = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )  # (B, K, 2)

    mask_f = mask.unsqueeze(-1).float()
    loss_per_entry = loss_per_entry * mask_f

    denom = mask_f.sum() * logits.shape[-1]
    return loss_per_entry.sum() / denom.clamp_min(1.0)

# -----------------------
# metrics
# -----------------------
def compute_soft_metrics(y_true, y_prob):
    metrics = {}
    metrics["mse"] = ((y_prob - y_true) ** 2).mean()

    for i, name in enumerate(TARGETS):
        err = np.abs(y_true[:, i] - y_prob[:, i])
        sq_err = (y_true[:, i] - y_prob[:, i]) ** 2
        metrics[f"{name}_mae"] = err.mean()
        metrics[f"{name}_brier"] = sq_err.mean()

        y_bin = (y_true[:, i] > 0.5).astype(np.int32)
        try:
            metrics[f"{name}_auc"] = roc_auc_score(y_bin, y_prob[:, i])
        except ValueError:
            metrics[f"{name}_auc"] = float("nan")

    metrics["mean_mae"] = 0.5 * (metrics["p_main_mae"] + metrics["p_alt_mae"])
    metrics["mean_brier"] = 0.5 * (metrics["p_main_brier"] + metrics["p_alt_brier"])
    metrics["mean_auc"] = 0.5 * (metrics["p_main_auc"] + metrics["p_alt_auc"])
    return metrics

def run_epoch(model, loader, optimizer=None, collect_metrics=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    all_targets = []
    all_probs = []

    event_correct = 0
    event_total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x, mask)
            loss = masked_bce_loss(logits, y, mask)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if collect_metrics:
                probs = torch.sigmoid(logits)

                flat_mask = mask.view(-1)
                flat_y = y.view(-1, 2)[flat_mask]
                flat_p = probs.view(-1, 2)[flat_mask]

                all_targets.append(flat_y.detach().cpu())
                all_probs.append(flat_p.detach().cpu())

                B = x.shape[0]
                for b in range(B):
                    valid = mask[b]
                    if valid.sum().item() == 0:
                        continue

                    pred_main = probs[b, valid, 0]
                    true_main = y[b, valid, 0]

                    pred_idx = torch.argmax(pred_main).item()
                    true_idx = torch.argmax(true_main).item()

                    event_correct += int(pred_idx == true_idx)
                    event_total += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {}

    if collect_metrics and len(all_targets) > 0:
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()

        metrics.update(compute_soft_metrics(all_targets, all_probs))
        metrics["event_main_accuracy"] = event_correct / max(event_total, 1)

    return avg_loss, metrics

# -----------------------
# training loop
# -----------------------
best_val_loss = float("inf")
best_state = None

for epoch in range(wandb.config.epochs):
    train_loss, _ = run_epoch(model, train_loader, optimizer=optimizer, collect_metrics=False)
    val_loss, val_metrics = run_epoch(model, val_loader, optimizer=None, collect_metrics=True)

    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"val_mse={val_metrics['mse']:.4f} | "
        f"val_p_main_auc={val_metrics['p_main_auc']:.4f} | "
        f"val_p_alt_auc={val_metrics['p_alt_auc']:.4f} | "
        f"val_event_acc={val_metrics['event_main_accuracy']:.4f}"
    )

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mse": val_metrics["mse"],
        "val_p_main_mae": val_metrics["p_main_mae"],
        "val_p_alt_mae": val_metrics["p_alt_mae"],
        "val_mean_mae": val_metrics["mean_mae"],
        "val_p_main_brier": val_metrics["p_main_brier"],
        "val_p_alt_brier": val_metrics["p_alt_brier"],
        "val_mean_brier": val_metrics["mean_brier"],
        "val_p_main_auc": val_metrics["p_main_auc"],
        "val_p_alt_auc": val_metrics["p_alt_auc"],
        "val_mean_auc": val_metrics["mean_auc"],
        "val_event_main_accuracy": val_metrics["event_main_accuracy"],
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(best_state, "best_deepset.pt")

# restore best model
if best_state is not None:
    model.load_state_dict(best_state["model_state_dict"])

# -----------------------
# final test evaluation
# -----------------------
test_loss, test_metrics = run_epoch(model, test_loader, optimizer=None, collect_metrics=True)

print("\nFinal test results")
print(f"test_loss={test_loss:.4f}")
print(f"test_mse={test_metrics['mse']:.4f}")
print(f"test_p_main_mae={test_metrics['p_main_mae']:.4f}")
print(f"test_p_alt_mae={test_metrics['p_alt_mae']:.4f}")
print(f"test_mean_mae={test_metrics['mean_mae']:.4f}")
print(f"test_p_main_auc={test_metrics['p_main_auc']:.4f}")
print(f"test_p_alt_auc={test_metrics['p_alt_auc']:.4f}")
print(f"test_event_main_accuracy={test_metrics['event_main_accuracy']:.4f}")

wandb.log({
    "test_loss": test_loss,
    "test_mse": test_metrics["mse"],
    "test_p_main_mae": test_metrics["p_main_mae"],
    "test_p_alt_mae": test_metrics["p_alt_mae"],
    "test_mean_mae": test_metrics["mean_mae"],
    "test_p_main_brier": test_metrics["p_main_brier"],
    "test_p_alt_brier": test_metrics["p_alt_brier"],
    "test_mean_brier": test_metrics["mean_brier"],
    "test_p_main_auc": test_metrics["p_main_auc"],
    "test_p_alt_auc": test_metrics["p_alt_auc"],
    "test_mean_auc": test_metrics["mean_auc"],
    "test_event_main_accuracy": test_metrics["event_main_accuracy"],
})