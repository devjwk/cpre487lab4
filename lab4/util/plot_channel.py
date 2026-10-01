# Lab 4 Section 5.3: same image / layer / channel across fp32, 8, 4, 2 bit.
# Usage (after running ./build/ml with QUANT_BITS=8, 4 and 2): python3 plot_channel.py [data_dir]
#   fp32: Keras reference outputs (our fp32 C++ build matches them, cosine 1.0)
#   N bit: dequantized outputs dumped by runQuantCompare to data/model/q<N>/image_0_layer_<i>_deq.bin
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = sys.argv[1] if len(sys.argv) > 1 else "../framework/data"
layers = [(0, "conv1", (60, 60, 32)), (1, "conv2", (56, 56, 32))]  # (C++ layer index, name, output shape)

image = np.fromfile(f"{data}/image_0.bin", dtype=np.float32).reshape(64, 64, 3)
fig, axes = plt.subplots(len(layers), 5, figsize=(16, 3.4 * len(layers)))

for row, (idx, name, shape) in zip(axes, layers):
    ref = np.fromfile(f"{data}/image_0_data/layer_{idx}_output.bin", dtype=np.float32).reshape(shape)
    ch = int(ref.reshape(-1, shape[2]).std(axis=0).argmax())  # most active channel in fp32
    maps = [("fp32", ref)] + [(f"{b} bit", np.fromfile(f"{data}/model/q{b}/image_0_layer_{idx}_deq.bin",
                                                       dtype=np.float32).reshape(shape)) for b in (8, 4, 2)]
    row[0].imshow(image)
    row[0].set_title("input image_0")
    vmax = ref[:, :, ch].max()  # same color scale for every bit width
    for ax, (label, m) in zip(row[1:], maps):
        err = np.abs(m[:, :, ch] - ref[:, :, ch]).mean()
        im = ax.imshow(m[:, :, ch], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{name} ch{ch} - {label}\nmean |err| = {err:.3f}")
    fig.colorbar(im, ax=row[1:].tolist(), shrink=0.8)
    for ax in row:
        ax.axis("off")
    print(f"{name}: channel {ch}")

fig.suptitle("Effect of quantization on one output channel (image_0)")
fig.savefig("quant_channel_compare.png", dpi=130, bbox_inches="tight")
print("saved quant_channel_compare.png")
