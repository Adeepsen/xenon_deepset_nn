import wandb
import pandas as pd

api = wandb.Api()

# Replace with your entity/project
runs = api.runs("senadeep5-clemson-university/xenon-deepset")

rows = []

for run in runs:
    row = {}

    # Hyperparameters
    row.update(run.config)

    # Final metrics
    row.update(run.summary)

    # Metadata
    row["run_name"] = run.name
    row["run_id"] = run.id
    row["state"] = run.state

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("wandb_runs.csv", index=False)

print(df.head())