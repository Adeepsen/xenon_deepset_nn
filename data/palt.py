import numpy as np

data = np.load(
    "/Users/adeepsen/xenon_deepset_nn/data/s2_tag_training_clusters.npy"
)

# Rows with anomalous p_alt
bad_rows = data[data["p_alt"] > 1]

# Unique event IDs containing at least one anomaly
events = np.unique(bad_rows["event_number"])

print(f"Found {len(events)} events with p_alt > 1")



for event_id in events[:20]: 
    event = data[data["event_number"] == event_id]

    print(f"\nEvent {event_id}")
    print(event[["p_main","p_alt", "drift_time_mean", "n_electrons_interface"]])


bad = data[data["p_alt"] > 1]

print("Number of anomalous clusters:", len(bad))
print("Maximum p_alt:", bad["p_alt"].max())

print("Median drift time:", np.median(bad["drift_time_mean"]))
print("Median drift spread:", np.median(bad["drift_time_spread"]))


bad = data[data["p_alt"] > 1]

print(np.percentile(bad["drift_time_mean"], [0, 25, 50, 75, 100]))


top13_us = 130.0 / 0.675   # 192.6 us
top13_ns = top13_us * 1000  # 192600 ns

bad_events = np.unique(data[data["drift_time_mean"] < top13_ns]["event_number"])
kept = data[~np.isin(data["event_number"], bad_events)]


print("remaining p_alt > 1 events:",(kept["p_alt"] > 1).sum())

print("Events removed:", len(bad_events))
