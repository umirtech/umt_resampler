# umt_resampler

Lightweight, high-performance, header-only audio resampler with fused channel mixing, designed for real-time and streaming applications.

---

## Overview

`umt_resampler` is a minimal, dependency-free C++ audio resampler built for low-latency systems.  
It implements a **polyphase FIR resampler with Kaiser windowing** and integrates **channel mixing** in the same processing stage for maximum efficiency.

The implementation is written in clean scalar C++ to allow compilers (Clang/GCC) to **auto-vectorize (SIMD)** without requiring platform-specific intrinsics.

---

## Features

- **Real-time and streaming friendly**
  - No dynamic allocations in the processing path
  - Deterministic performance

- **High-performance polyphase FIR resampling**
  - Efficient sample rate conversion (e.g. 48kHz → 44.1kHz)
  - Kaiser window filter design

- **Configurable quality**
  - Supports **16-tap** and **32-tap** modes
  - Default: **16 taps** (balanced performance/quality)

- **Fused channel mixing**
  - Multi-channel input/output support
  - Layout-aware processing (e.g. stereo, surround)
  - Efficient downmix (e.g. 5.1 / 7.1 → stereo)

- **Float processing**
  - Input: `float`
  - Output: `float`
  - Expected range: `[-1.0, 1.0]`

- **Auto-vectorization friendly**
  - Scalar implementation optimized for compiler SIMD generation (NEON/SSE)

- **Header-only**
  - No external dependencies
  - Requires only standard C/C++ library

---

## Design

- Algorithm: **Polyphase FIR**
- Window: **Kaiser (β ≈ 6)**
- Focus: **low latency + high throughput**

---

### Basic Example

```cpp
#include "umt_resampler.hpp"

umt::Resampler rs;
rs.init(2, 2, 48000, 44100);

float input[960];   // 480 frames stereo
float output[960];

uint32_t consumed = 0;

uint32_t produced = rs.resample(
    input,
    960,
    output,
    consumed
);
