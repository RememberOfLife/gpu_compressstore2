#pragma once
#include <src/cub_wraps.cuh>
#include <src/kernels/kernel_3pass.cuh>
#include <src/kernels/kernel_pattern.cuh>
#include <src/kernels/kernel_streaming_add.cuh>
#include "cuda_try.cuh"
#include "utils.cuh"

struct timings {
    float popc;
    float pss1;
    float pss2;
    float proc;
    float total;
    timings():
        popc(-1),
        pss1(-1),
        pss2(-1),
        proc(-1),
        total(-1)
    {}
    static float add_timings(float t1, float t2)
    {
        if (t1 < 0) {
            return t2;
        }
        if (t2 < 0) {
            return t1;
        }
        return t1 + t2;
    }
    void operator+=(timings r)
    {
        popc = add_timings(popc, r.popc);
        pss1 = add_timings(pss1, r.pss1);
        pss2 = add_timings(pss2, r.pss2);
        proc = add_timings(proc, r.proc);
        total = add_timings(total, r.total);
    }
};

struct intermediate_data {
    uint32_t* d_pss;
    uint32_t* d_pss2;
    uint32_t* d_popc;
    uint32_t* d_out_count;
    uint64_t* d_failure_count;
    uint8_t* d_cub_intermediate;
    size_t chunk_length;
    size_t element_count;
    size_t chunk_count;
    size_t max_stream_count;
    size_t cub_intermediate_size;
    size_t intermediate_size_3pass;
    cudaEvent_t dummy_event_1;
    cudaEvent_t dummy_event_2;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaStream_t* streams;
    cudaEvent_t* stream_events;
    intermediate_data(size_t element_count, int chunk_length, int max_stream_count)
    {
        this->chunk_length = chunk_length;
        this->element_count = element_count;
        this->chunk_count = ceildiv(ceil2mult(element_count, 8), chunk_length);
        this->max_stream_count = max_stream_count;
        uint8_t* null = (uint8_t*)NULL;
        intermediate_size_3pass = (chunk_count + 1) * sizeof(uint32_t);
        if (chunk_length > 32) {
            // for the streaming kernel
            intermediate_size_3pass = ceildiv(element_count + 1, 32) * sizeof(uint32_t);
        }
        size_t temp_storage_bytes_pss;
        CUDA_TRY(cub::DeviceScan::ExclusiveSum(null, temp_storage_bytes_pss, null, null, chunk_count));
        size_t temp_storage_bytes_flagged;
        cub::DeviceSelect::Flagged(null, temp_storage_bytes_flagged, null, null, null, null, element_count);
        size_t temp_storage_bytes_exclusive_sum;
        cub::DeviceScan::ExclusiveSum(null, temp_storage_bytes_exclusive_sum, null, null, chunk_count);
        cub_intermediate_size = std::max({temp_storage_bytes_pss, temp_storage_bytes_flagged, temp_storage_bytes_exclusive_sum});
        CUDA_TRY(cudaMalloc(&d_cub_intermediate, cub_intermediate_size));
        CUDA_TRY(cudaMalloc(&d_pss, intermediate_size_3pass));
        CUDA_TRY(cudaMalloc(&d_pss2, intermediate_size_3pass));
        CUDA_TRY(cudaMalloc(&d_popc, intermediate_size_3pass));
        CUDA_TRY(cudaMalloc(&d_out_count, sizeof(uint32_t) * (max_stream_count + 1)));
        d_failure_count = (uint64_t*)d_out_count;
        CUDA_TRY(cudaEventCreate(&dummy_event_1));
        CUDA_TRY(cudaEventCreate(&dummy_event_2));
        CUDA_TRY(cudaEventCreate(&start));
        CUDA_TRY(cudaEventCreate(&stop));
        streams = (cudaStream_t*)malloc(max_stream_count * sizeof(cudaStream_t));
        if (!streams) alloc_failure();
        stream_events = (cudaEvent_t*)malloc(max_stream_count * sizeof(cudaEvent_t));
        if (!stream_events) alloc_failure();
        for (size_t i = 0; i < max_stream_count; i++) {
            CUDA_TRY(cudaStreamCreate(&(streams[i])));
            CUDA_TRY(cudaEventCreate(&(stream_events[i])));
        }
    }
    template <typename T> void prepare_buffers(size_t element_count, size_t chunk_length, T* d_output, uint8_t* d_mask)
    {
        if (d_mask) {
            // make sure unused bits in bitmask are 0
            int unused_bits = overlap(element_count, 8);
            if (unused_bits) {
                uint8_t* last_mask_byte_ptr = d_mask + element_count / 8;
                uint8_t last_mask_byte = gpu_to_val(last_mask_byte_ptr);
                last_mask_byte >>= unused_bits;
                last_mask_byte <<= unused_bits;
                val_to_gpu(last_mask_byte_ptr, last_mask_byte);
            }
        }
        CUDA_TRY(cudaMemset(d_out_count, 0, (max_stream_count + 1) * sizeof(*d_out_count)));
        CUDA_TRY(cudaMemset(d_output, 0, element_count * sizeof(T)));
        if (this->element_count >= element_count || this->chunk_length >= chunk_length) return;
        error("sizes in intermediate data are smaller than the ones "
              "submitted "
              "to the algorithm");
    }
    ~intermediate_data()
    {
        CUDA_TRY(cudaFree(d_pss));
        CUDA_TRY(cudaFree(d_pss2));
        CUDA_TRY(cudaFree(d_popc));
        CUDA_TRY(cudaFree(d_cub_intermediate));
        CUDA_TRY(cudaFree(d_out_count));
        CUDA_TRY(cudaEventDestroy(dummy_event_1));
        CUDA_TRY(cudaEventDestroy(dummy_event_2));
        CUDA_TRY(cudaEventDestroy(start));
        CUDA_TRY(cudaEventDestroy(stop));
        for (int i = 0; i < max_stream_count; i++) {
            CUDA_TRY(cudaStreamDestroy(streams[i]));
            CUDA_TRY(cudaEventDestroy(stream_events[i]));
        }
        free(streams);
        free(stream_events);
    }
};

template <class T>
timings bench1_base_variant(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output, size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output, d_mask);
    timings times{};
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        element_count = ceil2mult(element_count, 8);
        times.popc = launch_3pass_popc_none(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_mask, id->d_pss, chunk_length, element_count);
        times.pss1 = launch_3pass_pss_gmem(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, id->d_pss, id->chunk_count, id->d_out_count);
        times.pss2 = launch_3pass_pss2_gmem(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, id->d_pss, id->d_pss2, id->chunk_count);
        times.proc = launch_3pass_proc_none(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_input, d_output, d_mask, id->d_pss2, true, NULL, chunk_length,
            element_count);
    });
    return times;
}

template <class T>
timings bench2_base_variant_skipping(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output, size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output, d_mask);
    timings times{};
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        element_count = ceil2mult(element_count, 8);
        times.popc = launch_3pass_popc_none(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_mask, id->d_pss, chunk_length, element_count);
        times.pss1 = launch_3pass_pss_gmem(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, id->d_pss, id->chunk_count, id->d_out_count);
        times.pss2 = launch_3pass_pss2_gmem(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, id->d_pss, id->d_pss2, id->chunk_count);
        times.proc = launch_3pass_proc_none(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_input, d_output, d_mask, id->d_pss2, true, id->d_pss, chunk_length,
            element_count);
    });
    return times;
}

template <class T>
timings bench3_3pass_streaming(
    intermediate_data* id,
    T* d_input,
    uint8_t* d_mask,
    T* d_output,
    size_t element_count,
    size_t chunk_length,
    int block_size,
    int grid_size,
    int stream_count = 2)
{
    // TODO: make use of streaming parameters
    id->prepare_buffers(element_count, chunk_length, d_output, d_mask);
    // since bymask bits are padded to zero in the last byte, we can
    // increase element_count to a multiple of 8
    timings times{};
    if (stream_count < 1) {
        error("stream_count must be >= 1");
    }
    int p2_sc = 1;
    while (p2_sc < stream_count) {
        p2_sc *= 2;
    }
    if (stream_count != p2_sc) error("stream_count must be a power of 2");

    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        element_count = ceil2mult(element_count, 8);
        CUDA_TRY(cudaMemset(id->d_out_count, 0x00, sizeof(uint32_t) * (stream_count + 1)));

        const uint64_t skip_block_size = 1024;
        const uint64_t minimum_stream_alignment = std::max(skip_block_size, chunk_length);
        uint64_t unit_count = ceildiv(element_count, minimum_stream_alignment);
        uint64_t units_per_stream = unit_count / stream_count;
        uint64_t units_for_last_stream = unit_count - units_per_stream * (stream_count - 1);

        uint64_t skip_blocks_per_unit = minimum_stream_alignment / skip_block_size;
        uint64_t skip_blocks_per_stream = units_per_stream * skip_blocks_per_unit;

        uint64_t chunks_per_unit = minimum_stream_alignment / chunk_length;
        uint64_t chunks_per_stream = units_per_stream * chunks_per_unit;
        uint64_t chunks_for_last_stream = units_for_last_stream * chunks_per_unit;

        uint64_t elements_per_stream = units_per_stream * minimum_stream_alignment;
        uint64_t elements_for_last_stream = element_count - units_per_stream * (stream_count - 1) * minimum_stream_alignment;
        uint64_t mask_bytes_per_stream = elements_per_stream / 8;

        uint32_t chunk_count_p2 = 1;
        while (chunk_count_p2 < chunks_per_stream) {
            chunk_count_p2 *= 2;
        }

        uint32_t chunk_length32 = chunk_length / 32;
        int popc1_threadcount = block_size;
        int popc1_blockcount = grid_size;
        int popc2_threadcount = block_size;
        int popc2_blockcount = grid_size;
        const int proc_threadcount = block_size;
        int proc_blockcount = grid_size;
        if (proc_blockcount < 1) {
            proc_blockcount = 1;
        }

        for (int i = 0; i < stream_count; i++) {
            bool ls = (i == stream_count - 1);
            uint64_t chunks_to_process = ls ? chunks_for_last_stream : chunks_per_stream;
            uint64_t elements_to_process = ls ? elements_for_last_stream : elements_per_stream;
            // launch popc for i
            kernel_3pass_popc_none_monolithic<<<popc1_blockcount, popc1_threadcount, 0, id->streams[i]>>>(
                d_mask + mask_bytes_per_stream * i, id->d_pss + chunks_per_stream * i, chunk_length32, elements_to_process);
            // launch pss for i
            // TODO these temporary storage allocations are timed
            launch_cub_pss(id->streams[i], 0, 0, id->d_pss + chunks_per_stream * i, id->d_out_count + i + 1, chunks_to_process);
            // if i > 0: launch extra: add previous d->out_count to own output
            if (i > 0) {
                launch_streaming_add_pss_totals(id->streams[i], id->d_out_count + i - 1, id->d_out_count + i);
            }
            // record event i
            CUDA_TRY(cudaEventRecord(id->stream_events[i], id->streams[i]));
            // launch optimization popc 1024 for i
            kernel_3pass_popc_none_monolithic<<<popc2_blockcount, popc2_threadcount, 0, id->streams[i]>>>(
                d_mask + mask_bytes_per_stream * i, id->d_popc + skip_blocks_per_stream * i, skip_block_size / 32, elements_to_process);
            // if i > 0: wait for event i-1
            if (i > 0) {
                CUDA_TRY(cudaStreamWaitEvent(id->streams[i], id->stream_events[i - 1]));
            }
            // launch optimized writeout proc for i using  d->out_count at i as
            // offset from output

            switch_3pass_proc_true_striding<T, true>(
                proc_blockcount, proc_threadcount, id->streams[i], d_input + elements_per_stream * i, d_output, d_mask + mask_bytes_per_stream * i,
                id->d_pss + chunks_per_stream * i, id->d_popc + skip_blocks_per_stream * i, chunk_length, elements_to_process, chunk_count_p2,
                id->d_out_count + i);
        }

        CUDA_TRY(cudaDeviceSynchronize());
    });
    return times;
}

template <class T>
timings bench4_3pass_optimized_read_non_skipping_cub_pss(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output, size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output, d_mask);
    timings times{};
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        element_count = ceil2mult(element_count, 8);
        times.popc = launch_3pass_popc_none(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_mask, id->d_pss, chunk_length, element_count);

        launch_3pass_pssskip(0, id->d_pss, id->d_out_count, id->chunk_count);
        CUDA_TRY(cub::DeviceScan::ExclusiveSum(id->d_cub_intermediate, id->cub_intermediate_size, id->d_pss, id->d_pss2, id->chunk_count));
        launch_3pass_pssskip(0, id->d_pss, id->d_out_count, id->chunk_count);

        times.proc = launch_3pass_proc_true(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_input, d_output, d_mask, id->d_pss2, true, NULL, chunk_length,
            element_count);
    });
    return times;
}

template <class T>
timings bench5_3pass_optimized_read_skipping_partial_pss(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output, size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output, d_mask);
    timings times{};
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        element_count = ceil2mult(element_count, 8);
        times.popc = launch_3pass_popc_none(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_mask, id->d_popc, chunk_length, element_count);
        cudaMemcpy(id->d_pss, id->d_popc, id->chunk_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        times.pss1 = launch_3pass_pss_gmem(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, id->d_pss, id->chunk_count, id->d_out_count);
        times.proc = launch_3pass_proc_true(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_input, d_output, d_mask, id->d_pss, false, id->d_popc, chunk_length,
            element_count);
    });
    return times;
}

template <class T>
timings bench6_3pass_optimized_read_skipping_two_phase_pss(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output, size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output, d_mask);
    timings times{};
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        element_count = ceil2mult(element_count, 8);
        times.popc = launch_3pass_popc_none(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_mask, id->d_popc, chunk_length, element_count);
        cudaMemcpy(id->d_pss, id->d_popc, id->chunk_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        times.pss1 = launch_3pass_pss_gmem(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, id->d_pss, id->chunk_count, id->d_out_count);
        times.pss2 = launch_3pass_pss2_gmem(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, id->d_pss, id->d_pss2, id->chunk_count);
        times.proc = launch_3pass_proc_true(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_input, d_output, d_mask, id->d_pss2, true, id->d_popc, chunk_length,
            element_count);
    });
    return times;
}

template <class T>
timings bench7_3pass_optimized_read_skipping_cub_pss(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output, size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output, d_mask);
    timings times{};
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        element_count = ceil2mult(element_count, 8);
        times.popc = launch_3pass_popc_none(id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_mask, id->d_popc, chunk_length, element_count);
        cudaMemcpy(id->d_pss, id->d_popc, id->chunk_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        launch_3pass_pssskip(0, id->d_pss, id->d_out_count, id->chunk_count);
        CUDA_TRY(cub::DeviceScan::ExclusiveSum(id->d_cub_intermediate, id->cub_intermediate_size, id->d_pss, id->d_pss2, id->chunk_count));
        launch_3pass_pssskip(0, id->d_pss, id->d_out_count, id->chunk_count);

        times.proc = launch_3pass_proc_true(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_input, d_output, d_mask, id->d_pss2, true, id->d_popc, chunk_length,
            element_count);
    });
    return times;
}

template <class T> timings bench8_cub_flagged(intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output, size_t element_count)
{
    id->prepare_buffers(element_count, 0, d_output, d_mask);
    bitstream_iterator bit{d_mask};
    timings times{};
    // determine temporary device storage requirements
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &times.total, {
        cub::DeviceSelect::Flagged(id->d_cub_intermediate, id->cub_intermediate_size, d_input, bit, d_output, id->d_out_count, element_count);
    });
    return times;
}

template <class T>
timings bench9_pattern(
    intermediate_data* id,
    T* d_input,
    uint32_t pattern,
    int pattern_length,
    T* d_output,
    size_t element_count,
    size_t chunk_length,
    int block_size,
    int grid_size)
{
    id->prepare_buffers(element_count, 0, d_output, NULL);
    timings times{};
    // determine temporary device storage requirements
    // CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &time, {
    times.total = launch_pattern_proc(
        id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_input, d_output, element_count, pattern, pattern_length, chunk_length);
    // });
    return times;
}

bool validate(intermediate_data* id, float* d_validation, float* d_output, uint64_t out_length, bool report_failures, uint64_t* failure_count = NULL)
{
    val_to_gpu(id->d_failure_count, 0);
    kernel_check_validation<<<64, 32>>>(d_validation, d_output, out_length, id->d_failure_count, report_failures);
    auto fc = gpu_to_val(id->d_failure_count);
    if (failure_count) *failure_count = fc;
    return (fc == 0);
}
