#ifndef KERNEL_PATTERN_CUH
#define KERNEL_PATTERN_CUH

#include <bit>
#include <cstdint>
#include <stdio.h> // debugging

#include "cuda_time.cuh"
#include "cuda_try.cuh"

#define CUDA_WARP_SIZE 32
template <typename T>
__global__ void kernel_pattern_proc(
    T* input, T* output, uint64_t N, uint32_t pattern, int pattern_length, uint32_t* thread_offset_initials, uint32_t* readin_offset_increments)
{
    // pattern kernel
    // adapt 1024bit writeout loop for proper gridstride and only always use the pattern as mask
    // since the pattern is known at launch, make sure that there are always 32 threads alive at writeout time
    // pattern skip lookup is computed on cpu side
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
    // loop through 1024bit blocks
    uint32_t chunklength = 1024;
    uint32_t chunk_id = warp_index + blockIdx.x * blockDim.x / CUDA_WARP_SIZE;
    uint32_t chunk_stride = gridDim.x * blockDim.x / CUDA_WARP_SIZE;
    for (; chunk_id < N / chunklength; chunk_id += chunk_stride) {
        // determine base offset using popc per pattern and number of patterns before this chunk
        uint64_t base_offset = pattern_popc * chunk_id * (chunklength / pattern_length);
        uint32_t thread_offset = smem_thread_offset_initials[warp_offset];
        uint32_t in_chunk_step = 0;
        while (thread_offset < chunklength) {
            output[base_offset + warp_offset + CUDA_WARP_SIZE * in_chunk_step++] = input[base_offset + thread_offset];
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
    int pattern_length)
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
    for (int i = 0; i < 32; i++) {
        std::cout << "[" << i << "] = " << thread_offset_initials[i];
        if (readin_offset_increments[i] > 0) {
            std::cout << " + " << readin_offset_increments[i];
        }
        std::cout << "\n";
    }
    // copy const arrays to device
    uint32_t* d_thread_offset_initials;
    uint32_t* d_readin_offset_increments;
    CUDA_TRY(cudaMalloc(&d_thread_offset_initials, sizeof(uint32_t) * 32));
    CUDA_TRY(cudaMalloc(&d_readin_offset_increments, sizeof(uint32_t) * 32));
    CUDA_TRY(cudaMemcpy(d_thread_offset_initials, thread_offset_initials, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_readin_offset_increments, d_readin_offset_increments, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice));
    CUDA_TIME(
        ce_start, ce_stop, 0, &time,
        (kernel_pattern_proc<T>
         <<<blockcount, threadcount>>>(d_input, d_output, N, pattern, pattern_length, d_thread_offset_initials, d_readin_offset_increments)));
    CUDA_TRY(cudaFree(d_readin_offset_increments));
    CUDA_TRY(cudaFree(d_thread_offset_initials));
    return time;
}

#endif
