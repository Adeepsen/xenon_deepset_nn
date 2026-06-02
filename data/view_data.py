import numpy as np
import matplotlib.pyplot as plt

# Load data

data = np.load(
    "/Users/adeepsen/xenon_deepset_nn/data/s2_tag_training_clusters.npy"
)

print("Fields:")
print(data.dtype.names)

# Dataset statistics

num_events = np.unique(data["event_number"]).size

print("\nDataset Summary")
print("----------------")
print("Total clusters:", len(data))
print("Total events:", num_events)

unique_events, counts = np.unique(
    data["event_number"],
    return_counts=True
)

print("Mean clusters/event:", counts.mean())
print("Median clusters/event:", np.median(counts))
print("Max clusters/event:", counts.max())
print("95th percentile:", np.percentile(counts, 95))
print("99th percentile:", np.percentile(counts, 99))

# Feature statistics

features = [
    "x",
    "y",
    "n_electrons_interface",
    "drift_time_mean",
    "drift_time_spread"
]

print("\nFeature Statistics")
print("------------------")

for feature in features:
    values = data[feature]

    print(f"\n{feature}")
    print("min :", values.min())
    print("max :", values.max())
    print("mean:", values.mean())
    print("std :", values.std())

# Feature histograms

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feature in enumerate(features):
    axes[i].hist(data[feature], bins=100)
    axes[i].set_title(feature)

# sixth panel = log electrons

log_ne = np.log10(
    data["n_electrons_interface"] + 1
)

axes[5].hist(log_ne, bins=100)
axes[5].set_title("log10(n_electrons_interface + 1)")

plt.tight_layout()
plt.show()

# Label histograms

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

labels = ["p_main", "p_alt"]

for i, label in enumerate(labels):
    axes[i].hist(data[label], bins=100)
    axes[i].set_title(label)

plt.tight_layout()
plt.show()

# Event multiplicity

plt.figure(figsize=(8, 4))
plt.hist(counts, bins=100)
plt.title("Clusters per Event")
plt.xlabel("Number of Clusters")
plt.ylabel("Number of Events")
plt.tight_layout()
plt.show()

# Inspect a few events

print("\nExample Events")
print("--------------")

for event_id in unique_events[:5]:

    event = data[
        data["event_number"] == event_id
    ]

    print(f"\nEvent {event_id}")
    print("Clusters:", len(event))

    print(
        event[
            [
                "n_electrons_interface",
                "drift_time_mean",
                "p_main",
                "p_alt"
            ]
        ]
    )

print()
print(
"p_alt > 1:", np.sum(data["p_alt"] > 1)
)
print("p_main > 1:", np.sum(data["p_main"] > 1))