#ifndef DATA_GENERATOR_CUH
#define DATA_GENERATOR_CUH

#include <vector>

#include <cstdint>
#include <stdio.h>

#include "fast_prng.cuh"

// offset: where to start generating inside the buffer
// length: how many bytes to generate
// selectivity: [0,1] chance for every bit to be a 1, default is 0
void generate_mask_uniform(uint8_t* h_buffer, uint64_t offset, uint64_t length, double selectivity, size_t* out_count = NULL)
{
    uint64_t p_adjusted = selectivity * UINT32_MAX;
    fast_prng rng(42);
    for (uint64_t i = offset; i < offset + length; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (rng.rand() < p_adjusted) {
                acc |= (1 << j);
                if (out_count) {
                    (*out_count)++;
                }
            }
        }
        h_buffer[i] = acc;
    }
}

// count: number of bytes on the total mask
// offset: where to start generating inside the buffer
// length: how many bytes to generate
void generate_mask_zipf(uint8_t* h_buffer, uint64_t count, uint64_t offset, uint64_t length, size_t* out_count = NULL)
{
    fast_prng rng(42);
    // probably r = a * (c * x)^-k
    // empirical:
    uint64_t n = count * 8;
    double a = 1.2; // TODO this spawns too many 1 bits overall, and the end is not as thinned as it should be, same for kernel
    double c = log10(static_cast<double>(n)) / static_cast<double>(n);
    double k = 1.43;

    for (uint64_t i = offset; i < offset + length; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            double ev = a * (1 / (pow((c * (i * 8 + (7 - j))), k)));
            double rv = static_cast<double>(rng.rand()) / static_cast<double>(UINT32_MAX);
            if (rv < ev) {
                acc |= (1 << j);
                if (out_count) {
                    (*out_count)++;
                }
            }
        }
        h_buffer[i] = acc;
    }
}

// count: number of bytes on the total mask
// offset: where to start generating inside the buffer
// length: how many bytes to generate
// segment_sizer: [0,1] percentage of count to treat as segment size, lower values cause more segments
void generate_mask_burst(uint8_t* h_buffer, uint64_t count, uint64_t offset, uint64_t length, double segment_sizer)
{
    // TODO rework so that it works with multithreading, stiching chunks properly together
    fast_prng rng(42);
    // segment_sizer sets pseudo segment distance, can be modified by up to +/-50% in size and is randomly 1/0
    double segment = static_cast<double>(count) * segment_sizer;
    double rv = static_cast<double>(rng.rand()) / static_cast<double>(UINT32_MAX);
    uint64_t current_length = static_cast<uint64_t>(segment * (rv + 0.5));
    bool is_one = false;
    for (uint64_t i = offset; i < offset + length; i++) {
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
        h_buffer[i] = acc;
    }
}

// offset: where to start generating inside the buffer
// length: how many bytes to generate
// spacing: every bit at index n%marg==0 is 1 and others 0, inverted mask if spacing<0
void generate_mask_offset(uint8_t* h_buffer, uint64_t offset, uint64_t length, int64_t spacing)
{
    fast_prng rng(42);
    bool invert = spacing < 0;
    spacing = (spacing == 0) ? 1 : spacing;
    for (uint64_t i = offset; i < offset + length; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if ((i * 8 + (7 - j)) % spacing == 0) {
                acc |= (1 << j);
            }
        }
        h_buffer[i] = (invert ? ~acc : acc);
    }
}

// offset: where to start generating inside the buffer
// length: how many bytes to generate
// pattern: bit pattern to be repeated, starting at most significant bit
// pattern_length: how many of the bits of the given 32-bit pattern are actually used in the pattern
void generate_mask_pattern(
    uint8_t* h_buffer, uint64_t offset, uint64_t length, uint32_t pattern = 0, uint32_t pattern_length = 0, size_t* out_count = NULL)
{
    for (uint64_t i = offset; i < offset + length; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if ((pattern >> (31 - ((i * 8 + (7 - j)) % pattern_length))) & 0b1) {
                acc |= (1 << j);
                if (out_count) {
                    (*out_count)++;
                }
            }
        }
        h_buffer[i] = acc;
    }
}

std::vector<uint8_t> generate_mask_clustering(float selectivity, size_t cluster_count, size_t total_elements, size_t* out_count = NULL)
{
    std::vector<bool> bitset;
    bitset.resize(total_elements);
    size_t total_set_one = selectivity * total_elements;
    size_t cluster_size = total_set_one / cluster_count;
    size_t slice = bitset.size() / cluster_count;
    // start by setting all to zero
    for (int i = 0; i < bitset.size(); i++) {
        bitset[i] = 0;
    }
    for (int i = 0; i < cluster_count; i++) {
        for (int k = 0; k < cluster_size; k++) {
            size_t cluster_offset = i * slice;
            bitset[k + cluster_offset] = 1;
        }
    }
    std::vector<uint8_t> final_bitmask_cpu;
    final_bitmask_cpu.resize(total_elements / 8);
    for (int i = 0; i < total_elements / 8; i++) {
        final_bitmask_cpu[i] = 0;
    }
    for (int i = 0; i < bitset.size(); i++) {
        // set bit of uint8
        if (bitset[i]) {
            if (out_count) {
                (*out_count)++;
            }
            uint8_t current = final_bitmask_cpu[i / 8];
            int location = i % 8;
            current = 1 << (7 - location);
            uint8_t add_res = final_bitmask_cpu[i / 8];
            add_res = add_res | current;
            final_bitmask_cpu[i / 8] = add_res;
        }
    }
    return final_bitmask_cpu;
}

// count: number of bits in the mask
// returns count of 1 bits in the mask
template <typename T> uint64_t generate_validation(T* h_data, uint8_t* h_mask, T* h_validation, uint64_t count)
{
    uint32_t onecount = 0;
    uint64_t val_idx = 0;
    for (uint64_t i = 0; i < count / 8; i++) {
        uint32_t acc = h_mask[i];
        for (int j = 7; j >= 0; j--) {
            uint64_t idx = i * 8 + (7 - j);
            bool v = 0b1 & (acc >> j);
            if (v) {
                onecount++;
                h_validation[val_idx++] = h_data[idx];
            }
        }
    }
    memset(&(h_validation[val_idx]), 0x00, (count - val_idx) * sizeof(T)); // set unused validation space to 0x00
    return onecount;
}

// offset: number of elements to skip
// length: number of elements to check
template <typename T> bool check_validation(T* h_validation, T* h_output, uint64_t offset, uint64_t length)
{
    for (uint64_t i = offset; i < offset + length; i++) {
        if (h_validation[i] != h_output[i]) {
            printf("validation failed @ %lu\n", i);
            return false;
        }
    }
    return true;
    // return memcmp(reinterpret_cast<uint8_t*>(h_validation+offset), reinterpret_cast<uint8_t*>(h_output+offset), length*sizeof(T)) == 0;
}

#endif
