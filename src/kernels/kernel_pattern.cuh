#ifndef KERNEL_PATTERN_CUH
#define KERNEL_PATTERN_CUH

#include <bit>
#include <cstdint>
#include <stdio.h> // debugging

#include "cuda_time.cuh"
#include "cuda_try.cuh"
#include "utils.cuh"

#define CUDA_WARP_SIZE 32
template <typename T>
__global__ void kernel_pattern_proc(
    T* input,
    T* output,
    uint64_t N,
    uint32_t pattern,
    int pattern_length,
    uint32_t* thread_offset_initials,
    uint32_t* readin_offset_increments,
    uint32_t patterns_per_chunk)
{
    // algo:
    // every warp processes a chunk of length (pattern_length * patterns_per_chunk) of the input together
    // every chunk starts at a clean boundary between patterns, every threads starts readin at its offset_initial
    // after every write a thread increments its readin offset by the offset_increment of the one bit at that position in the pattern
    uint32_t warp_offset = threadIdx.x % CUDA_WARP_SIZE;
    uint32_t warp_index = threadIdx.x / CUDA_WARP_SIZE;
    uint8_t pattern_popc = __popc(pattern);
    __shared__ uint32_t smem_thread_offset_initials[32];
    __shared__ uint32_t smem_readin_offset_increments[32];
    if (warp_index == 0) {
        smem_thread_offset_initials[warp_offset] = thread_offset_initials[warp_offset];
        smem_readin_offset_increments[warp_offset] = readin_offset_increments[warp_offset];
    }
    __syncthreads();
    // loop through chunks
    uint64_t chunk_length = pattern_length * patterns_per_chunk;
    uint64_t chunk_idx = warp_index + blockIdx.x * (blockDim.x / CUDA_WARP_SIZE);
    uint64_t chunk_stride = gridDim.x * (blockDim.x / CUDA_WARP_SIZE);
    for (; chunk_idx < ceil2mult(N, chunk_length); chunk_idx += chunk_stride) {
        // determine base writout offset of this chunk using number of patterns before this chunk
        uint64_t base_offset_readin = pattern_length * chunk_idx * patterns_per_chunk; // pattern_len for input offset
        uint64_t base_offset_writeout = pattern_popc * chunk_idx * patterns_per_chunk; // pattern_popc for output offset
        uint64_t thread_offset = smem_thread_offset_initials[warp_offset];
        uint64_t in_chunk_step = 0;
        //if (chunk_idx < 20) printf("N %lu; id %lu; in %lu; out %lu\n", N, chunk_idx, base_offset_readin, base_offset_writeout);
        while ((thread_offset < chunk_length) && (base_offset_readin + thread_offset < N)) {
            output[base_offset_writeout + warp_offset + CUDA_WARP_SIZE * in_chunk_step++] = input[base_offset_readin + thread_offset];
            thread_offset += smem_readin_offset_increments[thread_offset % pattern_length];
        }
        __syncwarp();
    }
}

// processing for patterned bitmasks
// do not call with 0 pattern
// all unused pattern bits MUST be 0
// pattern starts at msb
template <typename T>
float launch_pattern_proc(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint64_t N,
    uint32_t pattern,
    int pattern_length,
    uint32_t chunk_length)
{
    float time = 0;
    if (blockcount == 0) {
        blockcount = N / 1024;
    }
    uint32_t thread_offset_initials[32];
    uint32_t readin_offset_increments[32];
    // calculate first 32 start indices, determine 0 based offset for first 1 bit in every thread
    // calculate for every 1 bit the start of the next pattern block extended to 32 threads
    // determine thread based table entry for writeout offset increment
    int pattern_popc = std::popcount(pattern);
    for (int one_count = 0, thread_offset = 0; one_count < 32+pattern_popc; thread_offset++) {
        if ((pattern >> (31-((thread_offset) % pattern_length))) & 0b1) {
            if (one_count < 32) {
                thread_offset_initials[one_count] = thread_offset;
                readin_offset_increments[one_count] = 0;
            } else {
                int in_pattern_offset = thread_offset_initials[one_count-32];
                readin_offset_increments[in_pattern_offset] = thread_offset - in_pattern_offset;
            }
            one_count++;
        }
    }
    // print for checks
    // for (int i = 0; i < 32; i++) {
    //     std::cout << "[" << i << "] = " << thread_offset_initials[i];
    //     if (readin_offset_increments[i] > 0) {
    //         std::cout << " + " << readin_offset_increments[i];
    //     }
    //     std::cout << "\n";
    // }
    // calculate chunk length as multiple of pattern length, such that every chunk processed by a warp writes out at least 1024 one bits
    uint32_t patterns_per_chunk = ceildiv(chunk_length, pattern_popc);
    // copy const arrays to device
    uint32_t* d_thread_offset_initials;
    uint32_t* d_readin_offset_increments;
    CUDA_TRY(cudaMalloc(&d_thread_offset_initials, sizeof(uint32_t) * 32));
    CUDA_TRY(cudaMalloc(&d_readin_offset_increments, sizeof(uint32_t) * 32));
    CUDA_TRY(cudaMemcpy(d_thread_offset_initials, thread_offset_initials, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_readin_offset_increments, readin_offset_increments, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice));
    CUDA_TIME(
        ce_start, ce_stop, 0, &time,
        (kernel_pattern_proc<T><<<blockcount, threadcount>>>(d_input, d_output, N, pattern, pattern_length, d_thread_offset_initials, d_readin_offset_increments, patterns_per_chunk)));
    CUDA_TRY(cudaFree(d_readin_offset_increments));
    CUDA_TRY(cudaFree(d_thread_offset_initials));
    return time;
}

#endif
