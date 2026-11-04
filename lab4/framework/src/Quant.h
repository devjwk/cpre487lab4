#pragma once

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "Config.h"
#include "Types.h"
#include "Utils.h"

namespace ML {

// Quantized range for QUANT_BITS
constexpr i32 QMAX = (i32)((1LL << (QUANT_BITS - 1)) - 1);
constexpr i32 QMIN = -QMAX - 1;

// float = (int - zero) / scale; outScale == 0 means the output stays fp32
struct QuantParams {
    bool enabled = false;
    fp32 inScale = 1;
    i32 inZero = 0;
    fp32 wScale = 1;
    fp32 outScale = 0;
    i32 outZero = 0;
};

// Round half to even (like numpy) and clamp
inline i8 quantize(fp32 x, fp32 scale, i32 zero) {
    i32 q = (i32)std::nearbyint(x * scale) + zero;
    if (q > QMAX) q = QMAX;
    if (q < QMIN) q = QMIN;
    return (i8)q;
}

inline fp32 dequantize(i32 q, fp32 scale, i32 zero) { return (q - zero) / scale; }

// One line per conv/dense layer: <layer> <inScale> <inZero> <wScale>
inline std::vector<QuantParams> loadQuantParams(const Path& file) {
    std::ifstream in(file);
    if (!in.is_open()) throw std::runtime_error("Failed to open quantization params: " + file);

    std::vector<QuantParams> params;
    std::string name;
    QuantParams q;
    q.enabled = true;
    while (in >> name >> q.inScale >> q.inZero >> q.wScale) params.push_back(q);

    for (std::size_t i = 0; i + 1 < params.size(); i++) {
        params[i].outScale = params[i + 1].inScale;
        params[i].outZero = params[i + 1].inZero;
    }
    return params;
}

}  // namespace ML
