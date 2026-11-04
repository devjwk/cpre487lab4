# 1,000 validation images (uint8) and labels -> ../framework/data/val. In the notebook: %run -i export_val.py
import os
import numpy as np

N_VAL = 1000
out_dir = "../framework/data/val"
os.makedirs(out_dir, exist_ok=True)

images, labels = [], []
for ex in tiny_imagenet_builder.as_dataset()["validation"].take(N_VAL):
    images.append(ex["image"].numpy())
    labels.append(int(ex["label"].numpy()))
images = np.stack(images).astype(np.uint8)
labels = np.array(labels, dtype=np.int32)
assert images.shape == (N_VAL, 64, 64, 3), images.shape

images.tofile(os.path.join(out_dir, "val_images_u8.bin"))
labels.tofile(os.path.join(out_dir, "val_labels_i32.bin"))
print(f"saved {N_VAL} images, {len(np.unique(labels))} distinct classes -> {out_dir}")

# Keras accuracy on the same images
pred = model.predict(images.astype(np.float32) / 255.0, batch_size=100, verbose=0)
top10 = np.argsort(-pred, axis=1)[:, :10]
print(f"Keras fp32: top-1 = {(top10[:, 0] == labels).mean() * 100:.1f}%, "
      f"top-10 = {(top10 == labels[:, None]).any(axis=1).mean() * 100:.1f}%")
