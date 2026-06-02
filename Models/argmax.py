import numpy as np
import matplotlib.pyplot as plt

data = np.load("/Users/adeepsen/xenon_deepset_nn/data/s2_tag_training_clusters.npy")

# Get event boundaries once
event_ids, start_idx, counts = np.unique(
    data["event_number"],
    return_index=True,
    return_counts=True
)

def get_event_by_slice(start, count):
    return data[start:start + count]

def rank_clusters_by_electrons(event):
    return np.argsort(event["n_electrons_interface"])[::-1]

main_correct = 0
alt_correct = 0
num_events_with_alt = 0

sample_idx = np.random.choice(len(event_ids), size=10000, replace=False)

for i in sample_idx:
    start = start_idx[i]
    count = counts[i]
    event = get_event_by_slice(start, count)

    order = rank_clusters_by_electrons(event)

    pred_main_idx = order[0]
    pred_alt_idx = order[1] if len(order) > 1 else None

    true_main_idx = np.argmax(event["p_main"])
    true_alt_idx = np.argmax(event["p_alt"])

    if pred_main_idx == true_main_idx:
        main_correct += 1

    if pred_alt_idx is not None:
        num_events_with_alt += 1
        if pred_alt_idx == true_alt_idx:
            alt_correct += 1

print("Event-level argmax baseline")
print("---------------------------")
print("Main accuracy:", main_correct / len(sample_idx))
print("Alt accuracy:", alt_correct / num_events_with_alt)