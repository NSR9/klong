#!/usr/bin/env python3
"""Generate experiment report plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path(__file__).parent
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"base": "#4A90D9", "sft": "#E8555A", "sft_v1": "#F5A623", "sft_v3": "#2ECC40"}


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Training Loss Curve — v2 vs v3 comparison
# ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

epochs = [2.53, 5.0, 7.53, 10.0]
loss_v2 = [0.6819, 0.601, 0.5511, 0.517]
loss_v3 = [0.6719, 0.5888, 0.5399, 0.508]

ax.plot(epochs, loss_v2, "o--", color=COLORS["sft"], linewidth=2, markersize=7, label="v2 (no action mask)")
ax.plot(epochs, loss_v3, "s-", color=COLORS["sft_v3"], linewidth=2.5, markersize=7, label="v3 (action mask ON)")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training Loss", fontsize=12)
ax.set_ylim(0.45, 0.72)
ax.legend(fontsize=11)
plt.title("Training Loss: v2 vs v3 (Action Masking Fix)\nQwen 2.5-1.5B-Instruct, LoRA r=32, 10 epochs", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(output_dir / "plot_training_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_training_curve.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Perplexity Comparison (v1 vs v2 vs v3)
# ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

labels = ["v1: Base 1.5B\n(r=16, 3ep, no mask)", "v2: Instruct 1.5B\n(r=32, 10ep, no mask)", "v3: Instruct 1.5B\n(r=32, 10ep, masked)"]
base_ppl = [1.89, 1.88, 1.88]
sft_ppl = [1.85, 1.68, 1.67]

x = np.arange(len(labels))
width = 0.3

bars1 = ax.bar(x - width/2, base_ppl, width, label="Base Model", color=COLORS["base"], edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x + width/2, sft_ppl, width, label="SFT Model", color=COLORS["sft"], edgecolor="white", linewidth=1.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.annotate("-2.2%", xy=(0.15, 1.85), xytext=(0.4, 1.82),
            fontsize=10, color="#2ECC40", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ECC40", lw=1.5))
ax.annotate("-10.6%", xy=(1.15, 1.68), xytext=(1.4, 1.65),
            fontsize=10, color="#2ECC40", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ECC40", lw=1.5))
ax.annotate("-11.2%", xy=(2.15, 1.67), xytext=(2.4, 1.64),
            fontsize=10, color="#2ECC40", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ECC40", lw=1.5))

ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(1.5, 2.05)
ax.legend(fontsize=11)
ax.set_title("Perplexity on Held-Out Trajectories", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig(output_dir / "plot_perplexity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_perplexity.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Format Correctness Comparison (v1 vs v2 vs v3)
# ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

labels = ["v1: Base 1.5B\n(r=16, 3ep, no mask)", "v2: Instruct 1.5B\n(r=32, 10ep, no mask)", "v3: Instruct 1.5B\n(r=32, 10ep, masked)"]
base_fmt = [0, 0, 0]
sft_fmt = [0, 20, 20]

x = np.arange(len(labels))
width = 0.3

bars1 = ax.bar(x - width/2, base_fmt, width, label="Base Model", color=COLORS["base"], edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x + width/2, sft_fmt, width, label="SFT Model", color=COLORS["sft"], edgecolor="white", linewidth=1.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{int(bar.get_height())}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{int(bar.get_height())}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.annotate("+20pp", xy=(1.15, 20), xytext=(1.4, 30),
            fontsize=12, color="#2ECC40", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ECC40", lw=2))
ax.annotate("+20pp", xy=(2.15, 20), xytext=(2.4, 30),
            fontsize=12, color="#2ECC40", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ECC40", lw=2))

ax.set_ylabel("Format Correctness (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 40)
ax.legend(fontsize=11)
ax.set_title("Valid Tool Call Generation (Format Correctness)\n10 eval tasks per model", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig(output_dir / "plot_format_correctness.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_format_correctness.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 4: Combined Summary — v1 vs v2 vs v3
# ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Loss progression
ax = axes[0]
configs = ["v1\n(base, r=16\n3ep)", "v2\n(instruct, r=32\n10ep)", "v3\n(instruct, r=32\n10ep, masked)"]
final_losses = [0.71, 0.517, 0.508]
bar_colors = [COLORS["sft_v1"], COLORS["sft"], COLORS["sft_v3"]]
bars = ax.bar(configs, final_losses, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.6)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel("Final Training Loss", fontsize=11)
ax.set_title("Training Loss", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.85)

# Panel 2: Perplexity delta
ax = axes[1]
configs = ["v1", "v2", "v3"]
deltas = [-2.2, -10.6, -11.2]
bar_colors = [COLORS["sft_v1"], COLORS["sft"], COLORS["sft_v3"]]
bars = ax.bar(configs, deltas, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.5)
for bar, d in zip(bars, deltas):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 0.8,
            f'{d:.1f}%', ha='center', va='top', fontweight='bold', fontsize=12, color="white")
ax.set_ylabel("Perplexity Change (%)", fontsize=11)
ax.set_title("Perplexity Improvement", fontsize=12, fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
ax.set_ylim(-15, 2)

# Panel 3: Format correctness
ax = axes[2]
configs = ["v1", "v2", "v3"]
fmt_rates = [0, 20, 20]
bar_colors = [COLORS["sft_v1"], COLORS["sft"], COLORS["sft_v3"]]
bars = ax.bar(configs, fmt_rates, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.5)
for bar, f in zip(bars, fmt_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel("Format Correctness (%)", fontsize=11)
ax.set_title("Tool Call Validity", fontsize=12, fontweight="bold")
ax.set_ylim(0, 35)

fig.suptitle("KLong SFT Experiment: v1 vs v2 vs v3 Summary", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(output_dir / "plot_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_summary.png")

print("\nAll plots generated successfully!")
