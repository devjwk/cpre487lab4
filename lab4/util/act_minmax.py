# Lab 4 §3.3: per-layer activation min/max over 1,000 random training images.
# Run inside the notebook after `model` and `tiny_imagenet_builder` exist:
#   %run -i act_minmax.py
import json
import numpy as np
import tensorflow as tf

N_TRAIN = tiny_imagenet_builder.info.splits["train"].num_examples  # 100,000
N_SAMPLE, BATCH, SEED = 1000, 100, 487

mask = np.zeros(N_TRAIN, dtype=bool)
mask[np.random.default_rng(SEED).choice(N_TRAIN, N_SAMPLE, replace=False)] = True
mask = tf.constant(mask)

ds_calib = (tiny_imagenet_builder.as_dataset()["train"]
            .enumerate()
            .filter(lambda i, x: tf.gather(mask, i))
            .map(lambda i, x: tf.cast(x["image"], tf.float32) / 255.0)
            .batch(BATCH))

act_model = tf.keras.Model(inputs=model.inputs, outputs=[l.output for l in model.layers])
names = ["input"] + [l.name for l in model.layers]
stats = {n: {"min": np.inf, "max": -np.inf, "sum": 0.0, "n": 0} for n in names}

seen = 0
for x in ds_calib:
    for name, a in zip(names, [x] + list(act_model(x, training=False))):
        a, s = a.numpy(), stats[name]
        s["min"], s["max"] = min(s["min"], float(a.min())), max(s["max"], float(a.max()))
        s["sum"] += float(a.sum(dtype=np.float64))
        s["n"] += a.size
    seen += x.shape[0]
assert seen == N_SAMPLE, f"expected {N_SAMPLE} images, got {seen}"

result = {n: {"min": s["min"], "max": s["max"], "avg": s["sum"] / s["n"]} for n, s in stats.items()}
print(f"{'Layer':16s} {'min':>10s} {'max':>10s} {'avg':>10s}")
for n, r in result.items():
    print(f"{n:16s} {r['min']:10.4f} {r['max']:10.4f} {r['avg']:10.4f}")

with open("activation_ranges.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved activation_ranges.json")
