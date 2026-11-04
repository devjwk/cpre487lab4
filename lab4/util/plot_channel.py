# One channel of image_0 at fp32, 8, 4, 2 bit. Usage: python3 plot_channel.py [data_dir]
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = sys.argv[1] if len(sys.argv) > 1 else "../framework/data"
layers = [(0, "conv1", (60, 60, 32)), (1, "conv2", (56, 56, 32))]

image = np.fromfile(f"{data}/image_0.bin", dtype=np.float32).reshape(64, 64, 3)
fig, axes = plt.subplots(len(layers), 5, figsize=(16, 3.4 * len(layers)))

for row, (idx, name, shape) in zip(axes, layers):
    ref = np.fromfile(f"{data}/image_0_data/layer_{idx}_output.bin", dtype=np.float32).reshape(shape)
    ch = int(ref.reshape(-1, shape[2]).std(axis=0).argmax())  # most active channel in fp32
    maps = [("fp32", ref)] + [(f"{b} bit", np.fromfile(f"{data}/model/q{b}/image_0_layer_{idx}_deq.bin",
                                                       dtype=np.float32).reshape(shape)) for b in (8, 4, 2)]
    row[0].imshow(image)
    row[0].set_title("input image_0")
    vmax = ref[:, :, ch].max()
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
