import wandb
from deepset_train import train

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_event_main_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "batch_size": {"values": [256, 512]},
        "epochs": {"value": 100},
        "learning_rate": {"values": [1e-4, 3e-4, 1e-3, 3e-3]},
        "latent_dim": {"values": [16, 32, 64, 128, 256]},
        "phi_hidden": {"values": [32, 64, 128]},
        "rho_hidden": {"values": [32, 64, 128]},
        "num_workers": {"value": 4},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="xenon-deepset")
    wandb.agent(sweep_id, function=train, count=20)