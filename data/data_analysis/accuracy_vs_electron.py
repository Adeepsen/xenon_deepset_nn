import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/adeepsen/Downloads/test_main_event_diagnostics.csv")

df["electron_rank_bin"] = pd.cut(
    df["true_main_electron_rank"],
    bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 200],
    labels=["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51+"],
    include_lowest=True,
)

rank_summary = (
    df.groupby("electron_rank_bin", observed=False)
    .agg(
        n_events=("main_correct", "size"),
        main_accuracy=("main_correct", "mean"),
        mean_n_clusters=("n_clusters", "mean"),
        mean_pred_main_margin=("pred_main_margin", "mean"),
        mean_true_main_margin=("true_main_margin", "mean"),
    )
    .reset_index()
    
)

print(rank_summary)

plt.figure(figsize=(9, 5))
plt.plot(
    rank_summary["electron_rank_bin"].astype(str),
    rank_summary["main_accuracy"],
    marker="o",
)
plt.xlabel("True main cluster rank by electron count")
plt.ylabel("Main-cluster accuracy")
plt.title("Does the model succeed only when true main has the most electrons?")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


df["n_cluster_bin"] = pd.cut(
    df["n_clusters"],
    bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 200],
    labels=["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51+"],
    include_lowest=True,
)

mult_summary = (
    df.groupby("n_cluster_bin", observed=False)
    .agg(
        n_events=("main_correct", "size"),
        main_accuracy=("main_correct", "mean"),
        alt_accuracy=("alt_correct", "mean"),
        both_accuracy=("main_and_alt_correct", "mean"),
        mean_true_main_margin=("true_main_margin", "mean"),
        median_true_main_margin=("true_main_margin", "median"),
    )
    .reset_index()
)

#accuracy vs event multiplicity
print(mult_summary)

plt.figure(figsize=(9, 5))
plt.plot(mult_summary["n_cluster_bin"].astype(str), mult_summary["main_accuracy"], marker="o", label="main")
plt.plot(mult_summary["n_cluster_bin"].astype(str), mult_summary["alt_accuracy"], marker="o", label="alt")
plt.plot(mult_summary["n_cluster_bin"].astype(str), mult_summary["both_accuracy"], marker="o", label="both")
plt.xlabel("Number of clusters in event")
plt.ylabel("Accuracy")
plt.title("Accuracy vs event multiplicity")
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



#See if model is picking higher electron clusters over the correct clusters
plt.figure(figsize=(7, 5))
df.boxplot(
    column="raw_pred_minus_true_main_n_electrons_interface",
    by="main_correct",
)
plt.suptitle("")
plt.title("Electron-count difference: predicted main - true main")
plt.xlabel("Was main cluster correct?")
plt.ylabel("Electron-count difference")
plt.tight_layout()
plt.show()

#checks if misses are due to ambiguous data(small margins)

plt.figure(figsize=(7, 5))
df.boxplot(
    column="true_main_margin",
    by="main_correct",
)
plt.suptitle("")
plt.title("True main margin by hit/miss")
plt.xlabel("Was main cluster correct?")
plt.ylabel("true top p_main - second-highest true p_main")
plt.tight_layout()
plt.show()


#how confident is the model when it misses?
plt.figure(figsize=(7, 5))
df.boxplot(
    column="pred_main_margin",
    by="main_correct",
)
plt.suptitle("")
plt.title("Predicted main margin by hit/miss")
plt.xlabel("Was main cluster correct?")
plt.ylabel("predicted top p_main - second-highest predicted p_main")
plt.tight_layout()
plt.show()




#tie awareness
tol = 1e-6
df["main_correct_tie_aware"] = (
    df["true_main_value_at_pred"] >= df["max_true_main"] - tol
)

tie_summary = (
    df.groupby("n_cluster_bin", observed=False)
    .agg(
        n_events=("main_correct", "size"),
        argmax_accuracy=("main_correct", "mean"),
        tie_aware_accuracy=("main_correct_tie_aware", "mean"),
        median_true_main_margin=("true_main_margin", "median"),
    )
    .reset_index()
)

print(tie_summary)

plt.figure(figsize=(9, 5))
plt.plot(tie_summary["n_cluster_bin"].astype(str), tie_summary["argmax_accuracy"], marker="o", label="argmax accuracy")
plt.plot(tie_summary["n_cluster_bin"].astype(str), tie_summary["tie_aware_accuracy"], marker="o", label="tie-aware accuracy")
plt.xlabel("Number of clusters in event")
plt.ylabel("Main accuracy")
plt.title("Argmax vs tie-aware main accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#scatter plot for unambiguous, highly confident misses
real_misses = df[
    (df["main_correct"] == False)
    & (df["true_main_margin"] > 0.1)
    & (df["pred_main_margin"] > 0.1)
].copy()

plt.figure(figsize=(7, 5))
plt.scatter(
    real_misses["raw_pred_minus_true_main_n_electrons_interface"],
    real_misses["raw_pred_minus_true_main_drift_time_mean"],
    alpha=0.3,
)
plt.axvline(0, linestyle="--")
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted main - true main electron count")
plt.ylabel("Predicted main - true main drift time mean")
plt.title("Confident non-ambiguous misses")
plt.tight_layout()
plt.show()

print(real_misses[
    [
        "event_id",
        "n_clusters",
        "pred_main_margin",
        "true_main_margin",
        "true_main_electron_rank",
        "pred_main_electron_rank",
        "raw_true_main_n_electrons_interface",
        "raw_pred_main_n_electrons_interface",
        "raw_pred_minus_true_main_n_electrons_interface",
        "raw_true_main_drift_time_mean",
        "raw_pred_main_drift_time_mean",
        "raw_pred_minus_true_main_drift_time_mean",
    ]
].sort_values("pred_main_margin", ascending=False).head(25))