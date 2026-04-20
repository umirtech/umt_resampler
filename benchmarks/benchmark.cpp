#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>

#include "umt_resampler.hpp"

#define LOG(...) std::printf(__VA_ARGS__), std::printf("\n")

void run_example()
{
    LOG("=== Example Run ===");

    umt::Resampler rs;
    rs.init(2, 2, 48000, 44100);

    constexpr int frames = 480;
    float input[frames * 2] = {};
    float output[frames * 2] = {};

    uint32_t consumed = 0;

    uint32_t produced = rs.resample(
        input,
        frames * 2,
        output,
        consumed
    );

    LOG("Consumed: %u, Produced: %u", consumed, produced);
}

void benchmarkResampler(
    int srcChannels = 2,
    int dstChannels = 2,
    int srcRate = 48000,
    int dstRate = 44100,
    int seconds = 5
){
    using clock = std::chrono::high_resolution_clock;

        const int totalFrames = srcRate * seconds;
        const int totalSamples = totalFrames * srcChannels;


        std::vector<float> input(totalSamples);

        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto& v : input)
            v = dist(rng);

        umt::Resampler rs;
        rs.init(srcChannels, dstChannels, srcRate, dstRate);

        uint32_t expectedOut = rs.calculateOutputSampleCounts(totalSamples);
        std::vector<float> output(expectedOut);


        for (int i = 0; i < 5; ++i)
        {
            rs.reset();

            uint32_t consumedTotal = 0;

            while (consumedTotal < totalSamples)
            {
                uint32_t consumed = 0;

                rs.resample(
                        input.data() + consumedTotal,
                        totalSamples - consumedTotal,
                        output.data(),
                        consumed
                );

                consumedTotal += consumed;
            }
        }


        const int iterations = 10;

        double totalTime = 0.0;
        int64_t totalProduced = 0;

        for (int it = 0; it < iterations; ++it)
        {
            rs.reset();

            uint32_t consumedTotal = 0;
            uint32_t producedTotal = 0;

            auto t0 = clock::now();

            while (consumedTotal < totalSamples)
            {
                uint32_t consumed = 0;

                uint32_t produced = rs.resample(
                        input.data() + consumedTotal,
                        totalSamples - consumedTotal,
                        output.data(),
                        consumed
                );

                consumedTotal += consumed;
                producedTotal += produced;
            }

            auto t1 = clock::now();

            totalTime += std::chrono::duration<double>(t1 - t0).count();
            totalProduced = producedTotal;
        }

        double avgTime = totalTime / iterations;

        double samplesPerSec = totalProduced / avgTime;
        double framesPerSec  = (totalProduced / dstChannels) / avgTime;
        double realtimeFactor = framesPerSec / dstRate;


        LOG("=== Resampler Benchmark ===");
        LOG("SrcCh=%d DstCh=%d SrcRate=%d DstRate=%d",
            srcChannels, dstChannels, srcRate, dstRate);

        LOG("TotalProducedSamples=%lld AvgTime=%.6f sec",
            (long long)totalProduced, avgTime);

        LOG("SamplesPerSec=%.2f FramesPerSec=%.2f",
            samplesPerSec, framesPerSec);

        LOG("RealtimeFactor=%.2fx", realtimeFactor);
}


int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "bench") == 0)
    {
        benchmarkResampler();
    }
    else
    {
        run_example();
    }

    return 0;
}
