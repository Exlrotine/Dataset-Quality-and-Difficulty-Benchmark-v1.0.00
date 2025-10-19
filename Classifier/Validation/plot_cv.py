import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("result/k_fold/paircv_k10_base.csv")

k_minus_1 = 9                       # 你留了 9 次记录
corr_cols = [f"correct_{i}" for i in range(k_minus_1)]
prob_cols = [f"prob_{i}"    for i in range(k_minus_1)]
loss_cols = [f"loss_{i}"    for i in range(k_minus_1)]

df["acc_ratio"] = df[corr_cols].mean(1)                 # 0–1
df["avg_prob"]  = df[prob_cols].mean(1)
df["avg_loss"]  = df[loss_cols].mean(1)
df["score"]     = df["acc_ratio"] * df["avg_prob"]      # 综合记忆-置信度

plt.figure(figsize=(6,3))
plt.hist(df["score"], bins=40)
plt.title("Score distribution"); plt.xlabel("score"); plt.ylabel("#samples")
plt.show()

# TARGET_CLASS = 0          # ↩︎ 换成你关心的类
# for TARGET_CLASS in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
#     df_sub = df[df["label"] == TARGET_CLASS]
#     plt.figure(figsize=(2,2))
#     plt.hist(df_sub["score"], bins=30, color="tab:blue")
#     plt.title(f"Class {TARGET_CLASS}(n={len(df_sub)})")
#     plt.xlabel("score"); plt.ylabel("#samples")
#     plt.tight_layout(); plt.show()


def plot_combined_histograms(df, target_classes=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]):
    # Create a 2x6 grid of subplots, with independent y-axes
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(12, 4), sharex=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Iterate over each target class and corresponding subplot
    for idx, TARGET_CLASS in enumerate(target_classes):
        df_sub = df[df["label"] == TARGET_CLASS]
        if len(df_sub) > 0:  # Only plot if there is data
            axes[idx].hist(df_sub["score"], bins=30, color="tab:blue")
            # Set y-axis limits based on the histogram's data
            counts, _ = np.histogram(df_sub["score"], bins=30)
            axes[idx].set_ylim(0, counts.max() * 1.1)  # Add 10% padding for visibility
        axes[idx].set_title(f"C_{TARGET_CLASS} (n={len(df_sub)})", fontsize=10)
        axes[idx].set_xlabel("score", fontsize=10)
        axes[idx].set_ylabel("samples", fontsize=10)
        axes[idx].tick_params(axis='both', labelsize=10)  # Smaller tick labels

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
plot_combined_histograms(df)
# Example usage (assuming df is defined with 'label' and 'score' columns)
# plot_combined_histograms(df)