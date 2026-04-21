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
```

### Streaming Example (Recommended)

```cpp
#include "umt_resampler.hpp"
#include <vector>
#include <cstdint>

int main()
{
    const int srcChannels = 2;
    const int dstChannels = 2;
    const int srcRate = 48000;
    const int dstRate = 44100;

    umt::Resampler rs;
    rs.init(srcChannels, dstChannels, srcRate, dstRate);

    const int chunkFrames = 256;
    const int chunkSamples = chunkFrames * srcChannels;

    std::vector<float> input(chunkSamples);

    // Fill with dummy data (replace with real audio)
    for (auto &v : input) v = 0.1f;

    // Allocate correct output size
    uint32_t maxOut = rs.calculateOutputBufferSize(chunkSamples);
    std::vector<float> output(maxOut);

    uint32_t consumedTotal = 0;

    while (consumedTotal < chunkSamples)
    {
        uint32_t consumed = 0;

        uint32_t produced = rs.resample(
            input.data() + consumedTotal,
            chunkSamples - consumedTotal,
            output.data(),
            consumed
        );

        consumedTotal += consumed;

        if (consumed == 0)
            break; // safety guard

        // process 'produced' samples here
        // e.g. send to audio device or encoder
    }

    return 0;
}
```

## Benchmark

<!-- BENCHMARK_START -->
| Config | Value |
|--------|------|
| Channels | 2 → 2 |
| Sample Rate | 48000 → 44100 |
| Samples/sec | 140774946.01 |
| Frames/sec | 70387473.01 |
| Realtime Factor | 1596.09x |
| Last Updated | 2026-04-20 11:15 UTC |

<!-- BENCHMARK_END -->
