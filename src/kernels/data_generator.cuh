#ifndef KERNEL_DATA_GENERATOR_CUH
#define KERNEL_DATA_GENERATOR_CUH

#include <cstdint>

#include "cuda_time.cuh"
#include "cuda_try.cuh"
#include "fast_prng.cuh"

// count: number of bytes to generate
// selectivity: [0,1] chance for every bit to be a 1, default is 0
__global__ void kernel_generate_mask_uniform(uint8_t* d_buffer, uint64_t count, double selectivity)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridstride = blockDim.x * gridDim.x;

    fast_prng rng(tid);
    uint32_t p_adjusted = selectivity * UINT32_MAX;

    for (uint64_t i = tid; i < count; i += gridstride) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (rng.rand() < p_adjusted) {
                acc |= (1 << j);
            }
        }
        d_buffer[i] = acc;
    }
}

__global__ void kernel_generate_mask_zipf(uint8_t* d_buffer, uint64_t count)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridstride = blockDim.x * gridDim.x;

    fast_prng rng(tid);
    // probably r = a * (c * x)^-k
    // empirical:
    uint64_t n = count * 8;
    double a = 1.2;
    double c = log10(static_cast<double>(n)) / static_cast<double>(n);
    double k = 1.43;

    for (uint64_t i = tid; i < count; i += gridstride) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            double ev = a * (1 / (pow((c * (i * 8 + (7 - j))), k)));
            double rv = static_cast<double>(rng.rand()) / static_cast<double>(UINT32_MAX);
            if (rv < ev) {
                acc |= (1 << j);
            }
        }
        d_buffer[i] = acc;
    }
}

__global__ void kernel_generate_mask_burst(uint8_t* d_buffer, uint64_t count, double segment_sizer)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridstride = blockDim.x * gridDim.x;

    fast_prng rng(tid);
    // segment_sizer sets pseudo segment distance, can be modified by up to
    // +/-50% in size and is randomly 1/0
    double segment = static_cast<double>(count) * segment_sizer;
    double rv = static_cast<double>(rng.rand()) / static_cast<double>(UINT32_MAX);
    uint64_t current_length = static_cast<uint64_t>(segment * (rv + 0.5));
    bool is_one = false;

    for (uint64_t i = tid; i < count; i += gridstride) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (is_one) {
                acc |= (1 << j);
            }
            if (--current_length <= 0) {
                rv = static_cast<double>(rng.rand()) / static_cast<double>(UINT32_MAX);
                current_length = static_cast<uint64_t>(segment * (rv + 0.5));
                is_one = !is_one;
            }
        }
        d_buffer[i] = acc;
    }
}

__global__ void kernel_generate_mask_offset(uint8_t* d_buffer, uint64_t count, int64_t spacing)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridstride = blockDim.x * gridDim.x;

    fast_prng rng(tid);
    bool invert = spacing < 0;
    spacing = (spacing == 0) ? 1 : spacing;

    for (uint64_t i = tid; i < count; i += gridstride) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if ((i * 8 + (7 - j)) % spacing == 0) {
                acc |= (1 << j);
            }
        }
        d_buffer[i] = (invert ? ~acc : acc);
    }
}

__global__ void kernel_generate_mask_pattern(uint8_t* d_buffer, uint64_t count, uint32_t pattern = 0, uint32_t pattern_length = 0)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridstride = blockDim.x * gridDim.x;

    for (uint64_t i = tid; i < count; i += gridstride) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if ((pattern >> (31 - ((i * 8 + (7 - j)) % pattern_length))) & 0b1) {
                acc |= (1 << j);
            }
        }
        d_buffer[i] = acc;
    }
}

template <typename T>
__global__ void kernel_check_validation(T* d_validation, T* d_data, uint64_t count, uint64_t* d_failure_count, bool report_failures)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridstride = blockDim.x * gridDim.x;

    uint64_t failures = 0;

    for (uint64_t i = tid; i < count; i += gridstride) {
        char d[sizeof(T)], v[sizeof(T)];
        memcpy(&d, (void*)&d_data[i], sizeof(T));
        memcpy(&v, (void*)&d_validation[i], sizeof(T));
        for (int j = 0; j < sizeof(T); j++) {
            if (d[j] != v[j]) {
                if (report_failures) {
                    if constexpr (std::is_integral<T>::value) {
                        printf("failure: index: %lu: expected: %lu, got: %lu\n", i, (size_t)d_validation[i], (size_t)d_data[i]);
                    }
                    else {
                        printf("failure: index: %lu: expected: %f, got: %f\n", i, d_validation[i], d_data[i]);
                    }
                }
                failures++;
                break;
            }
        }
    }
    if (d_failure_count) {
#if defined(__CUDACC__)
        atomicAdd(reinterpret_cast<unsigned long long int*>(d_failure_count), failures);
#else
        __ullAtomicAdd(reinterpret_cast<unsigned long long int*>(d_failure_count), failures);
#endif
    }
}

#endif
