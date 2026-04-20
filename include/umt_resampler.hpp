
/*
 * umt_resampler - Lightweight high-performance audio resampler
 *
 * Copyright (c) 2026 Mohd Umeer https://github.com/umirtech
 * Licensed under the MIT License.
 *
 * ---------------------------------------------------------------------------
 * DESCRIPTION
 * ---------------------------------------------------------------------------
 * umt_resampler is a lightweight, header-only audio resampler designed for
 * real-time and streaming applications. It is built with performance,
 * simplicity, and predictability in mind, making it suitable for low-latency
 * audio pipelines such as media processing, VoIP, and embedded DSP systems.
 *
 * The implementation is based on a Polyphase FIR (Finite Impulse Response)
 * architecture using a Kaiser window, providing a good balance between
 * computational efficiency and audio quality.
 *
 * ---------------------------------------------------------------------------
 * FEATURES
 * ---------------------------------------------------------------------------
 * - Real-time and streaming friendly design
 *   • No dynamic allocations in the processing path
 *   • Deterministic performance suitable for low-latency systems
 *
 * - High-performance polyphase FIR resampling
 *   • Efficient multi-rate conversion (e.g., 48kHz → 44.1kHz)
 *   • Kaiser windowed filter design for controlled frequency response
 *
 * - Configurable filter quality
 *   • Supports 16-tap and 32-tap configurations
 *   • Default: 16 taps (balanced performance and quality)
 *
 * - Floating-point processing
 *   • Input:  float
 *   • Output: float
 *   • Range typically expected in [-1.0, 1.0]
 *
 * - Auto-vectorization friendly
 *   • Written in a clean scalar form
 *   • Structured to allow compilers (Clang/GCC) to auto-vectorize
 *     using SIMD (e.g., ARM NEON) without explicit intrinsics
 *
 * - Header-only and dependency-free
 *   • Requires only the C/C++ standard library
 *   • Easy integration into existing projects
 *
 * - Channel mixing support
 *   • Supports multi-channel input/output configurations
 *   • Layout-aware processing (e.g., stereo, surround)
 *   • Suitable for downmixing (e.g., 5.1 / 7.1 → stereo)
 *
 * ---------------------------------------------------------------------------
 * DESIGN GOALS
 * ---------------------------------------------------------------------------
 * - Minimal overhead and maximum throughput
 * - Predictable execution for real-time audio systems
 * - Clean and portable C++ implementation
 * - Easy integration without external dependencies
 *
 * ---------------------------------------------------------------------------
 * LICENSE (MIT)
 * ---------------------------------------------------------------------------
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef MEDIAENGINE_RESAMPLER_H
#define MEDIAENGINE_RESAMPLER_H
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

#define optimize __attribute__((__always_inline__))

namespace umt {

    // #define USE_32_TAPS
    static constexpr uint32_t kPhases = 64;
    static constexpr uint32_t kTaps   = 16;
    static constexpr uint32_t kDelay  = 64;


    namespace Mask
    {
        constexpr uint64_t FL  = 0x1;
        constexpr uint64_t FR  = 0x2;
        constexpr uint64_t FC  = 0x4;
        constexpr uint64_t LFE = 0x8;

        constexpr uint64_t BL  = 0x10;
        constexpr uint64_t BR  = 0x20;

        constexpr uint64_t SL  = 0x200;
        constexpr uint64_t SR  = 0x400;

        constexpr uint64_t FWL = 0x40;
        constexpr uint64_t FWR = 0x80;
    }

    class Resampler
    {
    private:
        // Utilities
        static constexpr uint32_t kMaxChannels = 10;
        static constexpr uint32_t kMask  = kDelay - 1;

        static constexpr float kCenter   = 0.7071f;
        static constexpr float kSurround = 0.7071f;
        static constexpr float kBack     = 0.5f;
        static constexpr float kWide     = 0.6f;
        static constexpr float kLFE      = 0.5f;

        static_assert((kPhases & (kPhases - 1)) == 0,"kPhases must be power of two");
        static constexpr uint32_t PHASE_BITS  = __builtin_ctz(kPhases);
        static_assert(PHASE_BITS > 0, "kPhases must be >= 2");
        static constexpr uint32_t PHASE_SHIFT = 32 - PHASE_BITS;


        static_assert(kTaps == 16, "kTaps must be 16 Or You can use 32 just Define USE_32_TAPS");


        struct Layout {
            int channels = 0;
            int index[kMaxChannels]{};

            void reset() {
                for (int & i : index) i = -1;
            }
        };

        static double besselI0(double x)
        {
            double ax = fabs(x);
            double y;

            if (ax < 3.75)
            {
                y = x / 3.75;
                y *= y;
                return 1.0 + y*(3.5156229 + y*(3.0899424 + y*(1.2067492
                + y*(0.2659732 + y*(0.0360768 + y*0.0045813)))));
            }
            else
            {
                y = 3.75 / ax;
                return (exp(ax) / sqrt(ax)) *
                       (0.39894228 + y*(0.01328592 + y*(0.00225319
                       + y*(-0.00157565 + y*(0.00916281 + y*(-0.02057706
                       + y*(0.02635537 + y*(-0.01647633 + y*0.00392377))))))));
            }
        }

        static inline float coeffs[kPhases][kTaps];

        static void initKernel() {
            static bool initialized = []{
                constexpr double PI = 3.141592653589793;
                constexpr double beta = 5.0;
                double denom = besselI0(beta);

                for (int p = 0; p < kPhases; ++p)
                {
                    double frac = double(p) / kPhases;
                    double norm = 0.0;

                    for (int t = 0; t < kTaps; ++t)
                    {
                        double n = (t - kTaps/2.0) - frac;
                        double x = PI * n;
                        double s = (fabs(x) < 1e-12) ? 1.0 : sin(x)/x;
                        double xt = (2.0 * t) / (kTaps - 1) - 1.0;
                        double rad = sqrt(1.0 - xt * xt);
                        double w = besselI0(beta * rad) / denom;
                        double c = s * w;
                        coeffs[p][t] = (float)c;
                        norm += c;
                    }
                    float inv = norm != 0 ? (float)(1.0 / norm) : 1.0f;
                    for (int t = 0; t < kTaps; ++t)
                        coeffs[p][t] *= inv;
                }
                return true;
            }();
        }

        static inline Layout buildLayout(uint64_t mask, int channels)
        {
            Layout l{};
            l.channels = channels;
            l.reset();

            int pos = 0;

            auto assign = [&](uint64_t bit, int chEnum)
            {
                if (mask & bit)
                {
                    l.index[chEnum] = pos++;
                }
            };

            assign(Mask::FL, 0);
            assign(Mask::FR, 1);
            assign(Mask::FC, 2);
            assign(Mask::LFE, 3);
            assign(Mask::SL, 4);
            assign(Mask::SR, 5);
            assign(Mask::BL, 6);
            assign(Mask::BR, 7);
            assign(Mask::FWL, 8);
            assign(Mask::FWR, 9);

            if (mask == 0)
            {
                for (int i = 0; i < channels; ++i)
                    l.index[i] = i;
            }

            return l;
        }

        template<typename Proc>
        inline void runLoop(int32_t& produced,float* __restrict output, Proc&& proc)
        {
            while (true)
            {
                uint32_t idx0 = phaseInt;
                if (idx0 + kTaps >= size)
                    break;
                uint32_t p = phaseFrac >> PHASE_SHIFT;
                proc(idx0, p, produced,output);
                uint32_t oldFrac = phaseFrac;
                phaseFrac += stepFrac;
                phaseInt  += stepInt + (phaseFrac < oldFrac);
                produced++;
            }
        }

        template<typename T>
        inline T* alloc_aligned(size_t count, size_t align = 64)
        {
            void* p = ::operator new[](count * sizeof(T), std::align_val_t(align));
            return reinterpret_cast<T*>(p);
        }

        template<typename T>
        inline void free_aligned(T* p)
        {
            ::operator delete[](p, std::align_val_t(64));
        }

        void freeMemory() {
            free_aligned(soa);
            free_aligned(fused);
            free_aligned(fusedPacked2x2);
            free_aligned(fusedPacked6x2);
            free_aligned(fusedPacked8x2);
            free_aligned(fusedPacked10x2);
        }

        void buildMixMatrix(float* __restrict mixMatrix)
        {
            auto has = [](const Layout& l, int ch) {
                return (ch < kMaxChannels && l.index[ch] >= 0);
            };

            for (int s = 0; s < srcChannels; ++s)
            {
                float* __restrict row = mixMatrix + s * dstChannels;
                for (int d = 0; d < dstChannels; ++d)
                    row[d] = 0.0f;
            }

            if (srcChannels == dstChannels)
            {
                for (int c = 0; c < srcChannels; ++c)
                    mixMatrix[c * dstChannels + c] = 1.0f;
            }
            else if (dstChannels == 2)
            {
                int L = dstLayout.index[0];
                int R = dstLayout.index[1];

                // Front
                if (has(srcLayout, 0))
                    mixMatrix[srcLayout.index[0] * dstChannels + L] += 1.0f;

                if (has(srcLayout, 1))
                    mixMatrix[srcLayout.index[1] * dstChannels + R] += 1.0f;

                // Center
                if (has(srcLayout, 2))
                {
                    int s = srcLayout.index[2];
                    mixMatrix[s * dstChannels + L] += kCenter;
                    mixMatrix[s * dstChannels + R] += kCenter;
                }

                // Surround
                if (has(srcLayout, 4))
                    mixMatrix[srcLayout.index[4] * dstChannels + L] += kSurround;

                if (has(srcLayout, 5))
                    mixMatrix[srcLayout.index[5] * dstChannels + R] += kSurround;

                // Back
                if (has(srcLayout, 6))
                    mixMatrix[srcLayout.index[6] * dstChannels + L] += kBack;

                if (has(srcLayout, 7))
                    mixMatrix[srcLayout.index[7] * dstChannels + R] += kBack;

                // Wide
                if (has(srcLayout, 8))
                    mixMatrix[srcLayout.index[8] * dstChannels + L] += kWide;

                if (has(srcLayout, 9))
                    mixMatrix[srcLayout.index[9] * dstChannels + R] += kWide;

                // LFE
                if (has(srcLayout, 3))
                {
                    int s = srcLayout.index[3];
                    mixMatrix[s * dstChannels + L] += kLFE;
                    mixMatrix[s * dstChannels + R] += kLFE;
                }

                // normalize
                for (int d = 0; d < 2; ++d)
                {
                    float sumSq = 0.0f;

                    for (int s = 0; s < srcChannels; ++s)
                    {
                        float v = mixMatrix[s * dstChannels + d];
                        sumSq += v * v;
                    }

                    if (sumSq > 0.0f)
                    {
                        float norm = 1.0f / sqrtf(sumSq);
                        for (int s = 0; s < srcChannels; ++s)
                            mixMatrix[s * dstChannels + d] *= norm;
                    }
                }
            }
            else if (srcChannels == 2 && dstChannels >= 6)
            {
                int FL = srcLayout.index[0];
                int FR = srcLayout.index[1];

                int dFL = dstLayout.index[0];
                int dFR = dstLayout.index[1];

                mixMatrix[FL * dstChannels + dFL] = 1.0f;
                mixMatrix[FR * dstChannels + dFR] = 1.0f;

                // Center
                if (has(dstLayout, 2))
                {
                    int dC = dstLayout.index[2];
                    mixMatrix[FL * dstChannels + dC] = 0.5f;
                    mixMatrix[FR * dstChannels + dC] = 0.5f;
                }

                // Surround
                if (has(dstLayout, 4))
                    mixMatrix[FL * dstChannels + dstLayout.index[4]] = 0.5f;

                if (has(dstLayout, 5))
                    mixMatrix[FR * dstChannels + dstLayout.index[5]] = 0.5f;

                // Back
                if (has(dstLayout, 6))
                    mixMatrix[FL * dstChannels + dstLayout.index[6]] = 0.3f;

                if (has(dstLayout, 7))
                    mixMatrix[FR * dstChannels + dstLayout.index[7]] = 0.3f;

                // Wide
                if (has(dstLayout, 8))
                {
                    int dFWL = dstLayout.index[8];
                    mixMatrix[FL * dstChannels + dFWL] = 0.7f;
                    mixMatrix[FR * dstChannels + dFWL] = 0.3f;
                }

                if (has(dstLayout, 9))
                {
                    int dFWR = dstLayout.index[9];
                    mixMatrix[FR * dstChannels + dFWR] = 0.7f;
                    mixMatrix[FL * dstChannels + dFWR] = 0.3f;
                }
            }
            else
            {
                for (uint32_t s = 0; s < srcChannels; ++s)
                {
                    uint32_t d = std::min<uint32_t>(s, dstChannels - 1);
                    mixMatrix[s * dstChannels + d] = 1.0f;
                }
            }
        }

        inline void pushFrame(const float* __restrict in)
        {
            uint32_t pos;

            if (size < kDelay)
            {
                pos = (head + size) & kMask;
                size++;
            }
            else
            {
                pos = head;
                head = (head + 1) & kMask;
            }

            float* __restrict ch = soa;
            for (int c = 0; c < srcChannels; ++c)
            {
                float v = in[c];

                ch[pos] = v;
                ch[pos + kDelay] = v;

                ch += soaStride;
            }

            if (head >= kDelay)
                head -= kDelay;
        }

        [[nodiscard]]
        optimize
        inline const float* __restrict window(uint32_t idx, uint32_t c) const
        {
            return soa + c * soaStride + head + idx;
        }
        
        
        struct KernelTap2x2
        {
            float k00, k01;
            float k10, k11;
        };

        struct KernelTap6x2
        {
            float k00, k01;
            float k10, k11;
            float k20, k21;
            float k30, k31;
            float k40, k41;
            float k50, k51;
        };

        struct KernelTap8x2
        {
            float k00, k01;
            float k10, k11;
            float k20, k21;
            float k30, k31;
            float k40, k41;
            float k50, k51;
            float k60, k61;
            float k70, k71;
        };

        struct KernelTap10x2
        {
            float k00, k01;
            float k10, k11;
            float k20, k21;
            float k30, k31;
            float k40, k41;
            float k50, k51;
            float k60, k61;
            float k70, k71;
            float k80, k81;
            float k90, k91;
        };
        
        
        void buildFusedKernels(const float* __restrict mixMatrix) {

            if (srcChannels == 2 && dstChannels == 2) {
                fusedPacked2x2 = alloc_aligned<KernelTap2x2>(kPhases * kTaps);

                float m00 = mixMatrix[0 * dstChannels + 0];
                float m01 = mixMatrix[0 * dstChannels + 1];
                float m10 = mixMatrix[1 * dstChannels + 0];
                float m11 = mixMatrix[1 * dstChannels + 1];

                for (int p = 0; p < kPhases; ++p)
                {
                    KernelTap2x2* __restrict phase = fusedPacked2x2 + (size_t)p * kTaps;
                    for (int t = 0; t < kTaps; ++t)
                    {
                        float c = coeffs[p][t];
                        phase[t].k00 = c * m00;
                        phase[t].k01 = c * m01;
                        phase[t].k10 = c * m10;
                        phase[t].k11 = c * m11;
                    }
                }
            }

            else if (srcChannels == 6 && dstChannels == 2)
            {
                fusedPacked6x2 = alloc_aligned<KernelTap6x2>(kPhases * kTaps);

                float m[6][2];
                for (int c = 0; c < 6; ++c)
                {
                    m[c][0] = mixMatrix[c * dstChannels + 0];
                    m[c][1] = mixMatrix[c * dstChannels + 1];
                }

                for (int p = 0; p < kPhases; ++p)
                {
                    auto* phase = fusedPacked6x2 + (size_t)p * kTaps;

                    for (int t = 0; t < kTaps; ++t)
                    {
                        float cval = coeffs[p][t];

                        phase[t].k00 = cval * m[0][0]; phase[t].k01 = cval * m[0][1];
                        phase[t].k10 = cval * m[1][0]; phase[t].k11 = cval * m[1][1];
                        phase[t].k20 = cval * m[2][0]; phase[t].k21 = cval * m[2][1];
                        phase[t].k30 = cval * m[3][0]; phase[t].k31 = cval * m[3][1];
                        phase[t].k40 = cval * m[4][0]; phase[t].k41 = cval * m[4][1];
                        phase[t].k50 = cval * m[5][0]; phase[t].k51 = cval * m[5][1];
                    }
                }
            }

            else if (srcChannels == 8 && dstChannels == 2)
            {
                fusedPacked8x2 = alloc_aligned<KernelTap8x2>(kPhases * kTaps);
                float m[8][2];
                for (int c = 0; c < 8; ++c)
                {
                    m[c][0] = mixMatrix[c * dstChannels + 0];
                    m[c][1] = mixMatrix[c * dstChannels + 1];
                }

                for (int p = 0; p < kPhases; ++p)
                {
                    KernelTap8x2* __restrict phase = fusedPacked8x2 + (size_t)p * kTaps;
                    for (int t = 0; t < kTaps; ++t)
                    {
                        float c = coeffs[p][t];
                        phase[t].k00 = c * m[0][0]; phase[t].k01 = c * m[0][1];
                        phase[t].k10 = c * m[1][0]; phase[t].k11 = c * m[1][1];
                        phase[t].k20 = c * m[2][0]; phase[t].k21 = c * m[2][1];
                        phase[t].k30 = c * m[3][0]; phase[t].k31 = c * m[3][1];
                        phase[t].k40 = c * m[4][0]; phase[t].k41 = c * m[4][1];
                        phase[t].k50 = c * m[5][0]; phase[t].k51 = c * m[5][1];
                        phase[t].k60 = c * m[6][0]; phase[t].k61 = c * m[6][1];
                        phase[t].k70 = c * m[7][0]; phase[t].k71 = c * m[7][1];
                    }
                }
            }

            else if (srcChannels == 10 && dstChannels == 2)
            {
                fusedPacked10x2 = alloc_aligned<KernelTap10x2>(kPhases * kTaps);

                float m[10][2];
                for (int c = 0; c < 10; ++c)
                {
                    m[c][0] = mixMatrix[c * dstChannels + 0];
                    m[c][1] = mixMatrix[c * dstChannels + 1];
                }

                for (int p = 0; p < kPhases; ++p)
                {
                    auto* phase = fusedPacked10x2 + (size_t)p * kTaps;

                    for (int t = 0; t < kTaps; ++t)
                    {
                        float cval = coeffs[p][t];

                        phase[t].k00 = cval * m[0][0]; phase[t].k01 = cval * m[0][1];
                        phase[t].k10 = cval * m[1][0]; phase[t].k11 = cval * m[1][1];
                        phase[t].k20 = cval * m[2][0]; phase[t].k21 = cval * m[2][1];
                        phase[t].k30 = cval * m[3][0]; phase[t].k31 = cval * m[3][1];
                        phase[t].k40 = cval * m[4][0]; phase[t].k41 = cval * m[4][1];
                        phase[t].k50 = cval * m[5][0]; phase[t].k51 = cval * m[5][1];
                        phase[t].k60 = cval * m[6][0]; phase[t].k61 = cval * m[6][1];
                        phase[t].k70 = cval * m[7][0]; phase[t].k71 = cval * m[7][1];
                        phase[t].k80 = cval * m[8][0]; phase[t].k81 = cval * m[8][1];
                        phase[t].k90 = cval * m[9][0]; phase[t].k91 = cval * m[9][1];
                    }
                }
            }

            else {
                size_t fusedCount = kPhases * srcChannels * dstChannels * kTaps;
                fused = alloc_aligned<float>(fusedCount);
                for (int p = 0; p < kPhases; ++p)
                {
                    for (int t = 0; t < kTaps; ++t)
                    {
                        float cval = coeffs[p][t];
                        for (int c = 0; c < srcChannels; ++c)
                        {
                            for (int d = 0; d < dstChannels; ++d)
                            {
                                size_t idx = (((size_t)p * srcChannels + c) * dstChannels + d) * kTaps + t;

                                fused[idx] = cval * mixMatrix[c * dstChannels + d];
                            }
                        }
                    }
                }
            }
        }


        // Dispatch Functions
        optimize
        inline void process_generic(
                uint32_t idx0,
                uint32_t p,
                uint32_t produced,
                float* __restrict output
        ){
            float* __restrict out = output + produced * dstChannels;
            size_t phaseBase = (size_t)p * srcChannels * dstChannels * kTaps;
            for (int d = 0; d < dstChannels; ++d)
            {
                float acc = 0.0f;
                size_t dBase = d * kTaps;
                for (int c = 0; c < srcChannels; ++c)
                {
                    const float* __restrict a = window(idx0, c);
                    const float* __restrict b = fused + phaseBase + c * dstChannels * kTaps + dBase;

                    acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] +
                           a[4]*b[4] + a[5]*b[5] + a[6]*b[6] + a[7]*b[7] +
                           a[8]*b[8] + a[9]*b[9] + a[10]*b[10] + a[11]*b[11] +
                           a[12]*b[12] + a[13]*b[13] + a[14]*b[14] + a[15]*b[15]

                    #if USE_32_TAPS
                           +
                           a[16]*b[16] + a[17]*b[17] + a[18]*b[18] + a[19]*b[19] +
                           a[20]*b[20] + a[21]*b[21] + a[22]*b[22] + a[23]*b[23] +
                           a[24]*b[24] + a[25]*b[25] + a[26]*b[26] + a[27]*b[27] +
                           a[28]*b[28] + a[29]*b[29] + a[30]*b[30] + a[31]*b[31];
                    #else
                           ;
                    #endif
                }

                out[d] = acc;
            }
        }


        optimize
        inline void process_2x2_fused(
                uint32_t idx0,
                uint32_t p,
                uint32_t produced,
                float* __restrict output
        ){
            const float* __restrict b0 = window(idx0, 0);
            const float* __restrict b1 = window(idx0, 1);

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            const KernelTap2x2* __restrict k = fusedPacked2x2 + p * kTaps;

            #define TAP(i)                               \
            do {                                         \
                const KernelTap2x2 ki = k[i];            \
                float s0 = b0[i];                        \
                float s1 = b1[i];                        \
                acc0 += s0 * ki.k00 + s1 * ki.k10;       \
                acc1 += s0 * ki.k01 + s1 * ki.k11;       \
            } while (0)

            TAP(0);  TAP(1);  TAP(2);  TAP(3);
            TAP(4);  TAP(5);  TAP(6);  TAP(7);
            TAP(8);  TAP(9);  TAP(10); TAP(11);
            TAP(12); TAP(13); TAP(14); TAP(15);
            #if USE_32_TAPS
            TAP(16); TAP(17); TAP(18); TAP(19);
            TAP(20); TAP(21); TAP(22); TAP(23);
            TAP(24); TAP(25); TAP(26); TAP(27);
            TAP(28); TAP(29); TAP(30); TAP(31);
            #endif

            #undef TAP

            float* __restrict out = output + produced * dstChannels;
            out[0] = acc0;
            out[1] = acc1;
        }


        optimize
        inline void process_6x2_fused(
                uint32_t idx0,
                uint32_t p,
                uint32_t produced,
                float* __restrict output
        ){
            const float* __restrict b0 = window(idx0, 0);
            const float* __restrict b1 = window(idx0, 1);
            const float* __restrict b2 = window(idx0, 2);
            const float* __restrict b3 = window(idx0, 3);
            const float* __restrict b4 = window(idx0, 4);
            const float* __restrict b5 = window(idx0, 5);

            const KernelTap6x2* __restrict k = fusedPacked6x2 + p * kTaps;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            #define TAP(i)                                               \
            do {                                                         \
                const KernelTap6x2 ki = k[i];                            \
                float s0 = b0[i]; float s1 = b1[i];                      \
                float s2 = b2[i]; float s3 = b3[i];                      \
                float s4 = b4[i]; float s5 = b5[i];                      \
                                                                         \
                acc0 +=                                                  \
                    s0*ki.k00 + s1*ki.k10 + s2*ki.k20 +                  \
                    s3*ki.k30 + s4*ki.k40 + s5*ki.k50;                   \
                                                                         \
                acc1 +=                                                  \
                    s0*ki.k01 + s1*ki.k11 + s2*ki.k21 +                  \
                    s3*ki.k31 + s4*ki.k41 + s5*ki.k51;                   \
                                                                         \
            } while (0)

            TAP(0);  TAP(1);  TAP(2);  TAP(3);
            TAP(4);  TAP(5);  TAP(6);  TAP(7);
            TAP(8);  TAP(9);  TAP(10); TAP(11);
            TAP(12); TAP(13); TAP(14); TAP(15);
            #ifdef USE_32_TAPS
            TAP(16); TAP(17); TAP(18); TAP(19);
            TAP(20); TAP(21); TAP(22); TAP(23);
            TAP(24); TAP(25); TAP(26); TAP(27);
            TAP(28); TAP(29); TAP(30); TAP(31);
            #endif
            #undef TAP

            float* __restrict out = output + produced * dstChannels;
            out[0] = acc0;
            out[1] = acc1;
        }


        optimize
        inline void process_8x2_fused(
                uint32_t idx0,
                uint32_t p,
                uint32_t produced,
                float* __restrict output
        ){
            const float* __restrict b0 = window(idx0, 0);
            const float* __restrict b1 = window(idx0, 1);
            const float* __restrict b2 = window(idx0, 2);
            const float* __restrict b3 = window(idx0, 3);
            const float* __restrict b4 = window(idx0, 4);
            const float* __restrict b5 = window(idx0, 5);
            const float* __restrict b6 = window(idx0, 6);
            const float* __restrict b7 = window(idx0, 7);

            const KernelTap8x2* __restrict k = fusedPacked8x2 + p * kTaps;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            #define TAP(i)                                               \
            do {                                                         \
                const KernelTap8x2 ki = k[i];                            \
                float s0 = b0[i]; float s1 = b1[i];                      \
                float s2 = b2[i]; float s3 = b3[i];                      \
                float s4 = b4[i]; float s5 = b5[i];                      \
                float s6 = b6[i]; float s7 = b7[i];                      \
                                                                         \
                acc0 +=                                                  \
                    s0*ki.k00 + s1*ki.k10 + s2*ki.k20 + s3*ki.k30 +      \
                    s4*ki.k40 + s5*ki.k50 + s6*ki.k60 + s7*ki.k70;       \
                                                                         \
                acc1 +=                                                  \
                    s0*ki.k01 + s1*ki.k11 + s2*ki.k21 + s3*ki.k31 +      \
                    s4*ki.k41 + s5*ki.k51 + s6*ki.k61 + s7*ki.k71;       \
            } while (0)

            TAP(0);  TAP(1);  TAP(2);  TAP(3);
            TAP(4);  TAP(5);  TAP(6);  TAP(7);
            TAP(8);  TAP(9);  TAP(10); TAP(11);
            TAP(12); TAP(13); TAP(14); TAP(15);
            #ifdef USE_32_TAPS
            TAP(16); TAP(17); TAP(18); TAP(19);
            TAP(20); TAP(21); TAP(22); TAP(23);
            TAP(24); TAP(25); TAP(26); TAP(27);
            TAP(28); TAP(29); TAP(30); TAP(31);
            #endif

            #undef TAP

            float* __restrict out = output + produced * dstChannels;
            out[0] = acc0;
            out[1] = acc1;
        }


        optimize
        inline void process_10x2_fused(
                uint32_t idx0,
                uint32_t p,
                uint32_t produced,
                float* __restrict output
        ){
            const float* __restrict b0 = window(idx0, 0);
            const float* __restrict b1 = window(idx0, 1);
            const float* __restrict b2 = window(idx0, 2);
            const float* __restrict b3 = window(idx0, 3);
            const float* __restrict b4 = window(idx0, 4);
            const float* __restrict b5 = window(idx0, 5);
            const float* __restrict b6 = window(idx0, 6);
            const float* __restrict b7 = window(idx0, 7);
            const float* __restrict b8 = window(idx0, 8);
            const float* __restrict b9 = window(idx0, 9);

            const KernelTap10x2* __restrict k = fusedPacked10x2 + p * kTaps;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            #define TAP(i)                                               \
            do {                                                         \
                const KernelTap10x2 ki = k[i];                           \
                float s0 = b0[i]; float s1 = b1[i];                      \
                float s2 = b2[i]; float s3 = b3[i];                      \
                float s4 = b4[i]; float s5 = b5[i];                      \
                float s6 = b6[i]; float s7 = b7[i];                      \
                float s8 = b8[i]; float s9 = b9[i];                      \
                                                                         \
                acc0 +=                                                  \
                    s0*ki.k00 + s1*ki.k10 + s2*ki.k20 + s3*ki.k30 +      \
                    s4*ki.k40 + s5*ki.k50 + s6*ki.k60 + s7*ki.k70 +      \
                    s8*ki.k80 + s9*ki.k90;                               \
                                                                         \
                acc1 +=                                                  \
                    s0*ki.k01 + s1*ki.k11 + s2*ki.k21 + s3*ki.k31 +      \
                    s4*ki.k41 + s5*ki.k51 + s6*ki.k61 + s7*ki.k71 +      \
                    s8*ki.k81 + s9*ki.k91;                               \
            } while (0)

            TAP(0);  TAP(1);  TAP(2);  TAP(3);
            TAP(4);  TAP(5);  TAP(6);  TAP(7);
            TAP(8);  TAP(9);  TAP(10); TAP(11);
            TAP(12); TAP(13); TAP(14); TAP(15);
            #ifdef USE_32_TAPS
            TAP(16); TAP(17); TAP(18); TAP(19);
            TAP(20); TAP(21); TAP(22); TAP(23);
            TAP(24); TAP(25); TAP(26); TAP(27);
            TAP(28); TAP(29); TAP(30); TAP(31);
            #endif

            #undef TAP

            float* __restrict out = output + produced * dstChannels;
            out[0] = acc0;
            out[1] = acc1;
        }


    public:

        ~Resampler() {
            freeMemory();
        }


        static inline uint64_t getDefaultChannelMask(int channels)
        {
            using namespace umt::Mask;

            switch (channels)
            {
                case 1: return FC;

                case 2: return FL | FR;

                case 3: return FL | FR | FC;

                case 4: return FL | FR | BL | BR;

                case 5: return FL | FR | FC | SL | SR;

                case 6: return FL | FR | FC | LFE | SL | SR;

                case 7: return FL | FR | FC | LFE | SL | SR | BL;

                case 8: return FL | FR | FC | LFE | SL | SR | BL | BR;

                case 9: return FL | FR | FC | SL | SR | BL | BR | FWL | FWR;

                case 10: return FL | FR | FC | LFE | SL | SR | BL | BR | FWL | FWR;

                default:
                    return 0;
            }
        }


        void init(int srcCh, int dstCh,
                  int srcRate, int dstRate,
                  uint64_t srcMask = 0,
                  uint64_t dstMask = 0
        ){
            freeMemory();

            srcChannels = srcCh;
            dstChannels = dstCh;
            srcRate_ = srcRate;
            dstRate_ = dstRate;

            uint64_t step64 = (uint64_t(srcRate) << 32) / dstRate;

            stepInt  = (uint32_t)(step64 >> 32);
            stepFrac = (uint32_t)(step64);

            if (srcMask == 0)
                srcMask = getDefaultChannelMask(srcCh);

            if (dstMask == 0)
                dstMask = getDefaultChannelMask(dstCh);

            srcLayout = buildLayout(srcMask, srcCh);
            dstLayout = buildLayout(dstMask, dstCh);


            size_t soaCount = srcChannels * (kDelay * 2);
            soa = alloc_aligned<float>(soaCount);


            // Build Mix Matrix
            auto* __restrict mixMatrix = alloc_aligned<float>(kMaxChannels * kMaxChannels);
            buildMixMatrix(mixMatrix);

            // Build Kernel
            initKernel();

            // Build Fused Kernels
            buildFusedKernels(mixMatrix);

            free_aligned(mixMatrix);

        }

        void reconfigure(int srcCh, int dstCh,
                         int srcRate, int dstRate,
                         uint64_t srcMask = 0,
                         uint64_t dstMask = 0)
        {
            init(srcCh, dstCh, srcRate, dstRate, srcMask, dstMask);
        }

        [[nodiscard]]
        uint32_t calculateOutputSampleCounts(uint32_t srcSamples) const
        {
            uint32_t frames = srcSamples / srcChannels;
            uint32_t outFrames = (frames * dstRate_ + srcRate_ - 1) / srcRate_;
            return outFrames * dstChannels;
        }


        void reset() {
            head = 0;
            size = 0;
            phaseFrac = 0;
            phaseInt = 0;
        }

        /*
         * Returns How Many Samples is Produced in Output Buffer
         * consumedInputSamples Used to Determine how much Input Samples is Consumed by Resampler if
         * So that they can be resend remaining inputSamples after consuming Resampled data
         */
        optimize
        uint32_t resample(
                const float* __restrict input,
                uint32_t srcSamples,
                float* __restrict output,
                uint32_t& consumedInputSamples)
        {
            consumedInputSamples = 0;

            if (!input || srcSamples <= 0)
                return 0;

            uint32_t frames = srcSamples / srcChannels;
            if (frames <= 0)
                return 0;

            uint32_t writable = kDelay - size;
            uint32_t framesToPush = std::min(writable, frames);

            for (uint32_t f = 0; f < framesToPush; ++f)
                pushFrame(&input[f * srcChannels]);

            consumedInputSamples = framesToPush * srcChannels;

            if (size < kTaps)
                return 0;

            int produced = 0;

            if (srcChannels == 2 && dstChannels == 2)
            {
                runLoop(produced,output,[this](uint32_t a, uint32_t b, uint32_t c, float* __restrict d) {
                    process_2x2_fused(a, b, c, d);
                });
            }

            else if (srcChannels == 6 && dstChannels == 2)
            {
                runLoop(produced,output,[this](uint32_t a, uint32_t b, uint32_t c, float* __restrict d) {
                    process_6x2_fused(a, b, c, d);
                });
            }


            else if (srcChannels == 8 && dstChannels == 2)
            {
                runLoop(produced,output,[this](uint32_t a, uint32_t b, uint32_t c, float* __restrict d) {
                    process_8x2_fused(a, b, c, d);
                });
            }

            else if (srcChannels == 10 && dstChannels == 2)
            {
                runLoop(produced,output,[this](uint32_t a, uint32_t b, uint32_t c, float* __restrict d) {
                    process_10x2_fused(a, b, c, d);
                });
            }

            else
            {
                runLoop(produced,output,[this](uint32_t a, uint32_t b, uint32_t c, float* __restrict d) {
                    process_generic(a, b, c, d);
                });
            }


            uint32_t consumed = phaseInt;

            if (consumed > 0)
            {
                phaseInt -= consumed;

                uint32_t consumedInt = std::min(consumed, (uint32_t)kDelay);

                head = (head + consumedInt) & kMask;
                size -= consumedInt;
            }

            return produced * dstChannels;
        }
    private:
        uint32_t srcChannels = 0;
        uint32_t dstChannels = 0;
        uint32_t srcRate_ = 0;
        uint32_t dstRate_ = 0;

        uint32_t phaseInt = 0;
        uint32_t phaseFrac = 0;

        uint32_t stepInt;
        uint32_t stepFrac;


        Layout srcLayout, dstLayout;

        uint32_t soaStride = kDelay * 2;
        float* __restrict soa = nullptr;
        float* __restrict fused = nullptr;
        KernelTap2x2* __restrict fusedPacked2x2 = nullptr;
        KernelTap6x2* __restrict fusedPacked6x2 = nullptr;
        KernelTap8x2* __restrict fusedPacked8x2 = nullptr;
        KernelTap10x2* __restrict fusedPacked10x2 = nullptr;

        uint32_t head = 0;
        uint32_t size = 0;
    };

}


#endif //MEDIAENGINE_RESAMPLER_H
