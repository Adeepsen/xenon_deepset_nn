import numpy as np
import pandas as pd

data = np.load("/Users/adeepsen/xenon_deepset_nn/data/s2_tag_training_clusters.npy")

#Removing top 13 cm of TPC data: Using 0.675 mm/us drift velocity ~192.6 us
top13_ns = 192_600

# Keep only events with 0 clusters in the top 13 cm
event_has_top13 = (
    pd.DataFrame(data)[["event_number", "drift_time_mean"]]
    .groupby("event_number")["drift_time_mean"]
    .min()
    .lt(top13_ns)
)

bad_event_ids = event_has_top13[event_has_top13].index.to_numpy()
good_mask = ~np.isin(data["event_number"], bad_event_ids)
data_fid = data[good_mask]

print("Events removed:", len(bad_event_ids))
print("Remaining p_alt > 1:", np.sum(data_fid["p_alt"] > 1))

event_ids = np.unique(data_fid["event_number"])

#Random sample of events to save my laptop from exploding
sample_size = 10000

if len(event_ids) > sample_size:
    event_ids = np.random.choice(
        event_ids,
        size=sample_size,
        replace=False
    )
main_correct = 0
alt_correct = 0
num_events_with_alt = 0

for event_id in event_ids:
    event = data_fid[data_fid["event_number"] == event_id]

    order = np.argsort(event["n_electrons_interface"])[::-1]

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

print("Event-level argmax baseline on fiducial data")
print("Main accuracy:", main_correct / len(event_ids))
print("Alt accuracy:", alt_correct / num_events_with_alt)