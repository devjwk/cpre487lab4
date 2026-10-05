import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/model/"
layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "dense1", "dense2"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
print(f"| Layer | #W | W min | W max | B min | B max |\n|---|---|---|---|---|---|")
for ax, name in zip(axes.flat, layers):
    w = np.fromfile(f"{path}/{name}_weights.bin", dtype=np.float32)
    b = np.fromfile(f"{path}/{name}_biases.bin", dtype=np.float32)
    ax.hist(w, bins=100, color="steelblue")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"{name} ({w.size:,} weights)")
    ax.set_xlabel("weight value (fp32)")
    ax.set_ylabel("count")
    print(f"| {name} | {w.size:,} | {w.min():.4f} | {w.max():.4f} | {b.min():.4f} | {b.max():.4f} |")
fig.suptitle("fp32 weight distributions per layer (biases excluded)")
fig.tight_layout()
fig.savefig("weight_histograms.png", dpi=130)
