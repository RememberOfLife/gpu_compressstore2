#ifndef KERNEL_3PASS_CUH
#define KERNEL_3PASS_CUH

#include <cmath>
#include <cstdint>
#include <stdio.h> // debugging

#include "cuda_time.cuh"

#define CUDA_WARP_SIZE 32
__global__ void kernel_3pass_popc_none_monolithic(uint8_t* mask, uint32_t* pss, uint32_t chunk_length32, uint32_t element_count)
{
    size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    size_t idx = chunk_length32 * 4 * tid; // index for 1st 8bit-element of this chunk
    size_t bit_idx = idx * 8;
    if (bit_idx > element_count) return;
    // assuming chunk_length to be multiple of 32
    size_t remaining_bytes_for_grid = (element_count - bit_idx) / 8;
    size_t bytes_to_process = chunk_length32 * 4;
    if (remaining_bytes_for_grid < bytes_to_process) {
        bytes_to_process = remaining_bytes_for_grid;
    }
    size_t popcount = 0;
    int i = 0;
    for (; i < bytes_to_process / 4 * 4; i += 4) {
        popcount += __popc(*reinterpret_cast<uint32_t*>(mask + idx + i));
    }
    if (i < bytes_to_process / 2 * 2) {
        popcount += __popc(*reinterpret_cast<uint16_t*>(mask + idx + i));
        i += 2;
    }
    if (i < bytes_to_process) popcount += __popc(*reinterpret_cast<uint8_t*>(mask + idx + i));
    pss[tid] = popcount;
}

__global__ void kernel_3pass_popc_none_striding(uint8_t* mask, uint32_t* pss, uint32_t chunk_length32, uint32_t element_count)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    uint32_t chunk_count = ceildiv(element_count, chunk_length32 * 32);
    for (; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t idx = chunk_length32 * 4 * tid; // index for 1st 8bit-element of this chunk
        uint32_t bit_idx = idx * 8;
        // assuming chunk_length to be multiple of 32
        uint32_t remaining_bytes_for_grid = (element_count - bit_idx) / 8;
        uint32_t bytes_to_process = chunk_length32 * 4;
        if (remaining_bytes_for_grid < bytes_to_process) {
            bytes_to_process = remaining_bytes_for_grid;
        }
        // assuming chunk_length to be multiple of 32
        uint32_t popcount = 0;
        int i = 0;
        for (; i < bytes_to_process / 4 * 4; i += 4) {
            popcount += __popc(*reinterpret_cast<uint32_t*>(mask + idx + i));
        }
        if (i < bytes_to_process / 2 * 2) {
            popcount += __popc(*reinterpret_cast<uint16_t*>(mask + idx + i));
            i += 2;
        }
        if (i < bytes_to_process) popcount += __popc(*reinterpret_cast<uint8_t*>(mask + idx + i));
        pss[tid] = popcount;
    }
}

float launch_3pass_popc_none(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    uint8_t* d_mask,
    uint32_t* d_pss,
    uint32_t chunk_length,
    uint32_t element_count)
{
    uint32_t chunk_count = ceildiv(element_count, chunk_length);
    float time = 0;
    uint32_t chunk_length32 = chunk_length / 32;
    if (blockcount == 0) {
        blockcount = (chunk_count / threadcount) + 1;
        CUDA_TIME(
            ce_start, ce_stop, 0, &time,
            (kernel_3pass_popc_none_monolithic<<<blockcount, threadcount>>>(d_mask, d_pss, chunk_length32, element_count)));
    }
    else {
        CUDA_TIME(
            ce_start, ce_stop, 0, &time,
            (kernel_3pass_popc_none_striding<<<blockcount, threadcount>>>(d_mask, d_pss, chunk_length32, element_count)));
    }
    return time;
}

__global__ void kernel_3pass_pssskip(uint32_t* pss, uint32_t* pss_total, uint32_t chunk_count)
{
    *pss_total += pss[chunk_count - 1];
}
void launch_3pass_pssskip(cudaStream_t stream, uint32_t* d_pss, uint32_t* d_pss_total, uint32_t chunk_count)
{
    if (chunk_count == 0) return;
    kernel_3pass_pssskip<<<1, 1, 0, stream>>>(d_pss, d_pss_total, chunk_count);
}

__global__ void kernel_3pass_pss_gmem_monolithic(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t stride = (1 << depth);
    tid = 2 * tid * stride + stride - 1;
    // tid is element id

    // thread loads element at tid and tid+stride
    if (tid >= chunk_count) {
        return;
    }
    uint32_t left_e = pss[tid];
    if (tid + stride < chunk_count) {
        pss[tid + stride] += left_e;
    }
    else {
        (*out_count) += left_e;
    }
}

__global__ void kernel_3pass_pss_gmem_striding(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    uint64_t stride = (1 << depth);
    for (uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint64_t cid = 2 * tid * stride + stride - 1; // calc chunk id
        if (cid >= chunk_count) {
            return;
        }
        uint32_t left_e = pss[cid];
        if (cid + stride < chunk_count) {
            pss[cid + stride] += left_e;
        }
        else {
            (*out_count) = left_e;
        }
    }
}

float launch_3pass_pss_gmem(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    uint32_t* d_pss,
    uint32_t chunk_count,
    uint32_t* d_out_count)
{
    float time = 0;
    float ptime;
    uint32_t max_depth = 0;
    for (uint32_t chunk_count_p2 = 1; chunk_count_p2 < chunk_count; max_depth++) {
        chunk_count_p2 *= 2;
    }
    if (blockcount == 0) {
        // reduce blockcount every depth iteration
        for (int i = 0; i < max_depth; i++) {
            blockcount = ((chunk_count >> i) / (threadcount * 2)) + 1;
            CUDA_TIME(
                ce_start, ce_stop, 0, &ptime, (kernel_3pass_pss_gmem_monolithic<<<blockcount, threadcount>>>(d_pss, i, chunk_count, d_out_count)));
            time += ptime;
        }
        // last pass forces result into d_out_count
        CUDA_TIME(
            ce_start, ce_stop, 0, &ptime,
            (kernel_3pass_pss_gmem_monolithic<<<1, 1>>>(d_pss, static_cast<uint8_t>(max_depth), chunk_count, d_out_count)));
        time += ptime;
    }
    else {
        for (int i = 0; i < max_depth; i++) {
            uint32_t req_blockcount = ((chunk_count >> i) / (threadcount * 2)) + 1;
            if (blockcount > req_blockcount) {
                blockcount = req_blockcount;
            }
            CUDA_TIME(
                ce_start, ce_stop, 0, &ptime, (kernel_3pass_pss_gmem_striding<<<blockcount, threadcount>>>(d_pss, i, chunk_count, d_out_count)));
            time += ptime;
        }
        // last pass forces result into d_out_count
        CUDA_TIME(
            ce_start, ce_stop, 0, &ptime,
            (kernel_3pass_pss_gmem_monolithic<<<1, 1>>>(d_pss, static_cast<uint8_t>(max_depth), chunk_count, d_out_count)));
        time += ptime;
    }
    return time;
}

__device__ uint32_t d_3pass_pproc_pssidx(uint32_t thread_idx, uint32_t* pss, uint32_t chunk_count_p2)
{
    chunk_count_p2 /= 2; // start by trying the subtree with length of half the next rounded up power of 2 of chunk_count
    uint32_t consumed = 0; // length of subtrees already fit inside idx_acc
    uint32_t idx_acc = 0; // assumed starting position for this chunk
    while (chunk_count_p2 >= 1) {
        if (thread_idx >= consumed + chunk_count_p2) {
            // partial tree [consumed, consumed+chunk_count_p2] fits into left side of thread_idx
            idx_acc += pss[consumed + chunk_count_p2 - 1];
            consumed += chunk_count_p2;
        }
        chunk_count_p2 /= 2;
    }
    return idx_acc;
}

__global__ void kernel_3pass_pss2_gmem_monolithic(uint32_t* pss_in, uint32_t* pss_out, uint32_t chunk_count, uint32_t chunk_count_p2)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    if (tid >= chunk_count) {
        return;
    }
    pss_out[tid] = d_3pass_pproc_pssidx(tid, pss_in, chunk_count_p2);
}

__global__ void kernel_3pass_pss2_gmem_striding(uint32_t* pss_in, uint32_t* pss_out, uint32_t chunk_count, uint32_t chunk_count_p2)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        pss_out[tid] = d_3pass_pproc_pssidx(tid, pss_in, chunk_count_p2);
    }
}

// computes per chunk pss for all chunks
float launch_3pass_pss2_gmem(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    uint32_t* d_pss_in,
    uint32_t* d_pss_out,
    uint32_t chunk_count)
{
    float time = 0;
    uint32_t chunk_count_p2 = 1;
    while (chunk_count_p2 < chunk_count) {
        chunk_count_p2 *= 2;
    }
    if (blockcount == 0) {
        blockcount = (chunk_count / threadcount) + 1;
        CUDA_TIME(
            ce_start, ce_stop, 0, &time,
            (kernel_3pass_pss2_gmem_monolithic<<<blockcount, threadcount>>>(d_pss_in, d_pss_out, chunk_count, chunk_count_p2)));
    }
    else {
        CUDA_TIME(
            ce_start, ce_stop, 0, &time,
            (kernel_3pass_pss2_gmem_striding<<<blockcount, threadcount>>>(d_pss_in, d_pss_out, chunk_count, chunk_count_p2)));
    }
    return time;
}

template <uint32_t BLOCK_DIM, typename T, bool complete_pss>
__device__ void kernel_3pass_proc_true_striding_naive_writeout(
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    uint32_t* popc,
    uint32_t chunk_length,
    uint32_t element_count,
    uint32_t chunk_count_p2,
    uint32_t* offset)
{
    uint32_t mask_byte_count = element_count / 8;
    uint32_t chunk_count = ceildiv(element_count, chunk_length);
    if (offset != NULL) {
        output += *offset;
    }
    constexpr uint32_t WARPS_PER_BLOCK = BLOCK_DIM / CUDA_WARP_SIZE;
    __shared__ uint32_t smem[BLOCK_DIM];
    __shared__ uint32_t smem_out_idx[WARPS_PER_BLOCK];
    uint32_t warp_remainder = WARPS_PER_BLOCK;
    while (warp_remainder % 2 == 0) {
        warp_remainder /= 2;
    }
    if (warp_remainder == 0) {
        warp_remainder = 1;
    }
    uint32_t grid_stride = chunk_length * warp_remainder;
    while (grid_stride % (CUDA_WARP_SIZE * BLOCK_DIM) != 0 || grid_stride * gridDim.x < element_count ||
           grid_stride / WARPS_PER_BLOCK < chunk_length) {
        grid_stride *= 2;
    }
    uint32_t warp_stride = grid_stride / WARPS_PER_BLOCK;
    uint32_t warp_offset = threadIdx.x % CUDA_WARP_SIZE;
    uint32_t warp_index = threadIdx.x / CUDA_WARP_SIZE;
    uint32_t base_idx = blockIdx.x * grid_stride + warp_index * warp_stride;
    if (base_idx >= element_count) return;
    uint32_t stride = 1024;
    if (warp_offset == 0) {
        if (complete_pss) {
            smem_out_idx[warp_index] = pss[base_idx / chunk_length];
        }
        else {
            smem_out_idx[warp_index] = d_3pass_pproc_pssidx(base_idx / chunk_length, pss, chunk_count_p2);
        }
    }
    uint32_t stop_idx = base_idx + warp_stride;
    if (stop_idx > chunk_length * chunk_count) {
        stop_idx = chunk_length * chunk_count;
    }
    while (base_idx < stop_idx) {
        // check chunk popcount at base_idx for potential skipped
        if (popc) {
            if (chunk_length >= stride) {
                if (popc[base_idx / chunk_length] == 0) {
                    base_idx += chunk_length;
                    continue;
                }
            }
            else {
                bool empty_stride = true;
                for (uint32_t cid = base_idx / chunk_length; cid < (base_idx + stride) / chunk_length; cid++) {
                    if (popc[cid] != 0) {
                        empty_stride = false;
                        break;
                    }
                }
                if (empty_stride) {
                    base_idx += stride;
                    continue;
                }
            }
        }
        uint32_t mask_idx = base_idx / 8 + warp_offset * 4;
        if (mask_idx < mask_byte_count) {
            uchar4 ucx = {0, 0, 0, 0};
            if (mask_idx + 4 > mask_byte_count) {
                switch (mask_byte_count - mask_idx) {
                    case 3: ucx.z = *(mask + mask_idx + 2);
                    case 2: ucx.y = *(mask + mask_idx + 1);
                    case 1: ucx.x = *(mask + mask_idx);
                }
            }
            else {
                ucx = *reinterpret_cast<uchar4*>(mask + mask_idx);
            }
            uchar4 uix{ucx.w, ucx.z, ucx.y, ucx.x};
            smem[threadIdx.x] = *reinterpret_cast<uint32_t*>(&uix);
        }
        else {
            smem[threadIdx.x] = 0;
        }
        __syncwarp();
        uint32_t input_index = base_idx + warp_offset;
        for (int i = 0; i < CUDA_WARP_SIZE; i++) {
            uint32_t s = smem[threadIdx.x - warp_offset + i];
            uint32_t out_idx_me = __popc(s >> (CUDA_WARP_SIZE - warp_offset));
            bool v = (s >> ((CUDA_WARP_SIZE - 1) - warp_offset)) & 0b1;
            if (v && input_index < element_count) {
                uint32_t out_idx = smem_out_idx[warp_index] + out_idx_me;
                output[out_idx] = input[input_index];
            }
            __syncwarp();
            if (warp_offset == (CUDA_WARP_SIZE - 1)) {
                smem_out_idx[warp_index] += out_idx_me + v;
            }
            __syncwarp();
            input_index += CUDA_WARP_SIZE;
        }
        base_idx += stride;
    }
}

template <uint32_t BLOCK_DIM, typename T, bool complete_pss>
__device__ void kernel_3pass_proc_true_striding_optimized_writeout(
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    uint32_t* popc,
    uint32_t chunk_length,
    uint32_t element_count,
    uint32_t chunk_count_p2,
    uint32_t* offset)
{
    uint32_t mask_byte_count = element_count / 8;
    uint32_t chunk_count = ceildiv(element_count, chunk_length);
    if (offset != NULL) {
        output += *offset;
    }
    constexpr uint32_t WARPS_PER_BLOCK = BLOCK_DIM / CUDA_WARP_SIZE;
    __shared__ uint32_t smem[BLOCK_DIM];
    __shared__ uint32_t out_indices[BLOCK_DIM];
    __shared__ uint32_t smem_out_idx[WARPS_PER_BLOCK];
    uint32_t warp_remainder = WARPS_PER_BLOCK;
    while (warp_remainder % 2 == 0) {
        warp_remainder /= 2;
    }
    if (warp_remainder == 0) {
        warp_remainder = 1;
    }
    uint32_t grid_stride = chunk_length * warp_remainder;
    while (grid_stride % (CUDA_WARP_SIZE * BLOCK_DIM) != 0 || grid_stride * gridDim.x < element_count ||
           grid_stride / WARPS_PER_BLOCK < chunk_length) {
        grid_stride *= 2;
    }
    uint32_t warp_stride = grid_stride / WARPS_PER_BLOCK;
    uint32_t warp_offset = threadIdx.x % CUDA_WARP_SIZE;
    uint32_t warp_index = threadIdx.x / CUDA_WARP_SIZE;
    uint32_t warp_base_index = threadIdx.x - warp_offset;
    uint32_t base_idx = blockIdx.x * grid_stride + warp_index * warp_stride;
    uint32_t stride = 1024;
    if (base_idx >= element_count) return;
    if (warp_offset == 0) {
        if (complete_pss) {
            smem_out_idx[warp_index] = pss[base_idx / chunk_length];
        }
        else {
            smem_out_idx[warp_index] = d_3pass_pproc_pssidx(base_idx / chunk_length, pss, chunk_count_p2);
        }
    }
    __syncwarp();
    uint32_t warp_output_index = smem_out_idx[warp_index];
    uint32_t stop_idx = base_idx + warp_stride;
    if (stop_idx > element_count) {
        stop_idx = element_count;
    }
    uint32_t elements_aquired = 0;
    while (base_idx < stop_idx) {
        // check chunk popcount at base_idx for potential skipped
        if (popc) {
            if (chunk_length >= stride) {
                if (popc[base_idx / chunk_length] == 0) {
                    base_idx += chunk_length;
                    continue;
                }
            }
            else {
                bool empty_stride = true;
                for (uint32_t cid = base_idx / chunk_length; cid < (base_idx + stride) / chunk_length; cid++) {
                    if (popc[cid] != 0) {
                        empty_stride = false;
                        break;
                    }
                }
                if (empty_stride) {
                    base_idx += stride;
                    continue;
                }
            }
        }
        uint32_t mask_idx = base_idx / 8 + warp_offset * 4;
        if (mask_idx < mask_byte_count) {
            uchar4 ucx = {0, 0, 0, 0};
            if (mask_idx + 4 > mask_byte_count) {
                switch (mask_byte_count - mask_idx) {
                    case 3: ucx.z = *(mask + mask_idx + 2);
                    case 2: ucx.y = *(mask + mask_idx + 1);
                    case 1: ucx.x = *(mask + mask_idx);
                }
            }
            else {
                ucx = *reinterpret_cast<uchar4*>(mask + mask_idx);
            }
            uchar4 uix{ucx.w, ucx.z, ucx.y, ucx.x};
            smem[threadIdx.x] = *reinterpret_cast<uint32_t*>(&uix);
        }
        else {
            smem[threadIdx.x] = 0;
        }
        __syncwarp();
        uint32_t input_index = base_idx + warp_offset;
        for (int i = 0; i < CUDA_WARP_SIZE; i++) {
            uint32_t s = smem[threadIdx.x - warp_offset + i];
            uint32_t out_idx_me = __popc(s >> (CUDA_WARP_SIZE - warp_offset));
            bool v = (s >> ((CUDA_WARP_SIZE - 1) - warp_offset)) & 0b1;
            if (warp_offset == CUDA_WARP_SIZE - 1) {
                smem_out_idx[warp_index] = out_idx_me + v;
            }
            __syncwarp();
            uint32_t out_idx_warp = smem_out_idx[warp_index];
            if (elements_aquired + out_idx_warp >= CUDA_WARP_SIZE) {
                uint32_t out_idx_me_full = out_idx_me + elements_aquired;
                uint32_t out_idices_idx = warp_base_index + out_idx_me_full;
                if (v && out_idx_me_full < CUDA_WARP_SIZE) {
                    out_indices[out_idices_idx] = input_index;
                }
                __syncwarp();
                output[warp_output_index + warp_offset] = input[out_indices[warp_base_index + warp_offset]];
                __syncwarp();
                if (v && out_idx_me_full >= CUDA_WARP_SIZE) {
                    out_indices[out_idices_idx - CUDA_WARP_SIZE] = input_index;
                }
                elements_aquired += out_idx_warp;
                elements_aquired -= CUDA_WARP_SIZE;
                warp_output_index += CUDA_WARP_SIZE;
            }
            else {
                if (v) {
                    out_indices[warp_base_index + elements_aquired + out_idx_me] = input_index;
                }
                elements_aquired += out_idx_warp;
            }
            input_index += CUDA_WARP_SIZE;
        }
        base_idx += stride;
    }
    __syncwarp();
    if (warp_offset < elements_aquired) {
        output[warp_output_index + warp_offset] = input[out_indices[warp_base_index + warp_offset]];
    }
}

template <uint32_t BLOCK_DIM, typename T, bool complete_pss, bool optimized_writeout>
__global__ void kernel_3pass_proc_true_striding(
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    uint32_t* popc,
    uint32_t chunk_length,
    uint32_t element_count,
    uint32_t chunk_count_p2,
    uint32_t* offset)
{
    if constexpr (optimized_writeout) {
        kernel_3pass_proc_true_striding_optimized_writeout<BLOCK_DIM, T, complete_pss>(
            input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
    }
    else {
        kernel_3pass_proc_true_striding_naive_writeout<BLOCK_DIM, T, complete_pss>(
            input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
    }
}

template <typename T, bool complete_pss, bool optimized_writeout>
void switch_3pass_proc_true_striding(
    uint32_t block_count,
    uint32_t block_dim,
    cudaStream_t stream,
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    uint32_t* popc,
    uint32_t chunk_length,
    uint32_t element_count,
    uint32_t chunk_count_p2,
    uint32_t* offset)
{
    switch (block_dim) {
        default:
        case 32: {
            kernel_3pass_proc_true_striding<32, T, complete_pss, optimized_writeout>
                <<<block_count, 32, 0, stream>>>(input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
        } break;
        case 64: {
            kernel_3pass_proc_true_striding<64, T, complete_pss, optimized_writeout>
                <<<block_count, 64, 0, stream>>>(input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
        } break;
        case 128: {
            kernel_3pass_proc_true_striding<128, T, complete_pss, optimized_writeout>
                <<<block_count, 128, 0, stream>>>(input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
        } break;
        case 256: {
            kernel_3pass_proc_true_striding<256, T, complete_pss, optimized_writeout>
                <<<block_count, 256, 0, stream>>>(input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
        } break;
        case 512: {
            kernel_3pass_proc_true_striding<512, T, complete_pss, optimized_writeout>
                <<<block_count, 512, 0, stream>>>(input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
        } break;
        case 1024: {
            kernel_3pass_proc_true_striding<1024, T, complete_pss, optimized_writeout>
                <<<block_count, 1024, 0, stream>>>(input, output, mask, pss, popc, chunk_length, element_count, chunk_count_p2, offset);
        } break;
    }
}

// processing (for complete and partial pss) using optimized memory access pattern
template <typename T, bool optimized_writeout>
float launch_3pass_proc_true(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint8_t* d_mask,
    uint32_t* d_pss,
    bool full_pss,
    uint32_t* d_popc,
    uint32_t chunk_length,
    uint32_t element_count)
{
    uint32_t chunk_count = ceildiv(element_count, chunk_length);
    float time = 0;
    uint32_t chunk_count_p2 = 1;
    while (chunk_count_p2 < chunk_count) {
        chunk_count_p2 *= 2;
    }
    if (blockcount == 0) {
        blockcount = chunk_count / 1024;
    }
    if (blockcount < 1) {
        blockcount = 1;
    }
    if (full_pss) {
        CUDA_TIME(
            ce_start, ce_stop, 0, &time,
            (switch_3pass_proc_true_striding<T, true, optimized_writeout>(
                blockcount, threadcount, 0, d_input, d_output, d_mask, d_pss, d_popc, chunk_length, element_count, chunk_count_p2, NULL)));
    }
    else {
        CUDA_TIME(
            ce_start, ce_stop, 0, &time,
            (switch_3pass_proc_true_striding<T, false, optimized_writeout>(
                blockcount, threadcount, 0, d_input, d_output, d_mask, d_pss, d_popc, chunk_length, element_count, chunk_count_p2, NULL)));
    }
    return time;
}

template <typename T, bool complete_pss>
__global__ void kernel_3pass_proc_none_monolithic(
    T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t* popc, uint32_t chunk_length8, uint32_t element_count, uint32_t chunk_count_p2)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid * chunk_length8 * 8 >= element_count) {
        return;
    }
    // check chunk popcount at chunk tid for potential skipped
    if (popc && popc[tid] == 0) {
        return;
    }
    uint32_t out_idx;
    if (complete_pss) {
        out_idx = pss[tid];
    }
    else {
        out_idx = d_3pass_pproc_pssidx(tid, pss, chunk_count_p2);
    }
    uint32_t element_idx = tid * chunk_length8;
    for (uint32_t i = element_idx; i < element_idx + chunk_length8; i++) {
        uint8_t acc = mask[i];
        for (int j = 7; j >= 0; j--) {
            uint64_t idx = i * 8 + (7 - j);
            bool v = 0b1 & (acc >> j);
            if (v && idx < element_count) {
                output[out_idx++] = input[idx];
            }
        }
    }
}

template <typename T, bool complete_pss>
__global__ void kernel_3pass_proc_none_striding(
    T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t* popc, uint32_t chunk_length8, uint32_t element_count, uint32_t chunk_count_p2)
{
    uint32_t chunk_count = ceildiv(element_count, chunk_length8 * 8);
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        // check chunk popcount at chunk tid for potential skipped
        if (popc && popc[tid] == 0) {
            continue;
        }
        uint32_t out_idx;
        if (complete_pss) {
            out_idx = pss[tid];
        }
        else {
            out_idx = d_3pass_pproc_pssidx(tid, pss, chunk_count_p2);
        }
        uint32_t element_idx = tid * chunk_length8;
        for (uint32_t i = element_idx; i < element_idx + chunk_length8; i++) {
            uint8_t acc = mask[i];
            for (int j = 7; j >= 0; j--) {
                uint64_t idx = i * 8 + (7 - j);
                bool v = 0b1 & (acc >> j);
                if (v && idx < element_count) {
                    output[out_idx++] = input[idx];
                }
            }
        }
    }
}

// processing (for complete and partial pss) using thread-chunk-wise access
template <typename T>
float launch_3pass_proc_none(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint8_t* d_mask,
    uint32_t* d_pss,
    bool full_pss,
    uint32_t* d_popc,
    uint32_t chunk_length,
    uint32_t element_count)
{
    uint32_t chunk_count = ceildiv(element_count, chunk_length);
    float time = 0;
    uint32_t chunk_count_p2 = 1;
    while (chunk_count_p2 < chunk_count) {
        chunk_count_p2 *= 2;
    }
    if (blockcount == 0) {
        blockcount = (chunk_count / threadcount) + 1;
        if (full_pss) {
            CUDA_TIME(
                ce_start, ce_stop, 0, &time,
                (kernel_3pass_proc_none_monolithic<T, true>
                 <<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_popc, chunk_length / 8, element_count, 0)));
        }
        else {
            CUDA_TIME(
                ce_start, ce_stop, 0, &time,
                (kernel_3pass_proc_none_monolithic<T, false>
                 <<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_popc, chunk_length / 8, element_count, chunk_count_p2)));
        }
    }
    else {
        if (full_pss) {
            CUDA_TIME(
                ce_start, ce_stop, 0, &time,
                (kernel_3pass_proc_none_striding<T, true>
                 <<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_popc, chunk_length / 8, element_count, 0)));
        }
        else {
            CUDA_TIME(
                ce_start, ce_stop, 0, &time,
                (kernel_3pass_proc_none_striding<T, false>
                 <<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_popc, chunk_length / 8, element_count, chunk_count_p2)));
        }
    }
    return time;
}

#endif
