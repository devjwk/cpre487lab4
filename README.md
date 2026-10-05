# CprE 487/587 Lab 4 - Quantization and Reduced Precision (Team 06)

Authors: Zach Dixon, Jongwoo Kim

Post-training quantization (8, 4 and 2 bit) of the TinyImageNet CNN from Labs 1-2, implemented in our
C++ inference framework and compared against the original fp32 model.

## Repository layout

```
lab4/
  framework/     C++ framework (Lab 2 code + quantization), one code base for fp32 / 8 / 4 / 2 bit
  util/          notebook and helper scripts
lab1_06/         Lab 1 submission (notebook, exported weights)
lab2_src_06/     Lab 2 submission (fp32 C++ framework this lab builds on)
lab3_src_6/      Lab 3 submission (MAC unit)
```

## Build and run

```bash
cd lab4/framework
make clean && make build QUANT_BITS=8   # 32 (fp32), 8, 4 or 2; default is 8 (src/Config.h)
./build/ml          # framework tests; quantized builds also print per-layer error vs fp32 and top-1 checks
./build/ml val      # top-1 / top-10 accuracy and latency over the 1,000 validation images
```

Run `make clean` whenever `QUANT_BITS` changes. A quantized build loads `data/model/q<bits>/`, the fp32
build loads `data/model/*.bin`.

## Results (lab machine, 1,000 validation images)

| Model | Storage | Latency (avg) | Top-1 | Top-10 |
|---|---|---|---|---|
| fp32 | 3,081 KB | 142.9 ms | 23.5% | 60.5% |
| 8 bit | 773 KB | 87.6 ms | 23.8% | 60.4% |
| 4 bit | 773 KB (388 KB if packed) | 87.6 ms | 6.6% | 24.1% |
| 2 bit | 773 KB (196 KB if packed) | 87.8 ms | 0.3% | 4.6% |

4-bit and 2-bit values are stored in int8 containers, so their storage and latency match 8 bit.

## How the quantized inference works

1. `util/act_minmax.py` profiles each layer's activation range on 1,000 training images.
2. `util/quantize_export.py <bits>` computes per-layer input scale / zero-point and weight scale, and
   writes int8 weights, int32 biases and `quant_params.txt` to `framework/data/model/q<bits>/`.
3. In C++, each conv/dense layer accumulates `(input - zero_point) * weight` in int32, dequantizes,
   applies ReLU, and requantizes to int8 with the next layer's parameters. MaxPool and Flatten work
   directly on int8; the last dense layer outputs fp32 for the unchanged softmax.

## util scripts

| File | Purpose |
|---|---|
| `lab4_06.ipynb` | Lab 1 notebook extended for Lab 4 (activation profiling, validation export) |
| `weight_hist.py` | weight histograms and weight/bias min-max |
| `act_minmax.py` | activation min/max/avg -> `activation_ranges.json` (run in the notebook with `%run -i`) |
| `quantize_export.py` | quantized parameters for a given bit width |
| `export_val.py` | export 1,000 validation images and labels (run in the notebook with `%run -i`) |
| `plot_channel.py` | same output channel at fp32 / 8 / 4 / 2 bit -> `quant_channel_compare.png` |

The notebook needs the Lab 1 environment (`lab1_venv`) and `CNN_TinyImageNet.h5` next to it; the model
file is not tracked in this repository.
