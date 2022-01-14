#include <__clang_cuda_device_functions.h>
#include <bitset>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <thread>
#include <vector>

#define DISABLE_CUDA_TIME
#include "cuda_time.cuh"
#include "cuda_try.cuh"

#include "csv_loader.hpp"
#include "utils.cuh"
#include "data_generator.cuh"
#include "benchmarks.cuh"
#include "kernels/data_generator.cuh"

int main(int argc, char** argv)
{
    if (argc > 1) {
        int device = atoi(argv[1]);
        printf("setting device numer to %i\n", device);
        CUDA_TRY(cudaSetDevice(device));
    }
    const char* csv_path = "../res/Arade_1.csv";
    if (argc > 2) {
        csv_path = argv[2];
    }
    int iterations = 1;
    if (argc > 3) {
        iterations = atoi(argv[3]);
        if (iterations < 1) iterations = 1;
        printf("setting iterations numer to %i\n", iterations);
    }
    bool report_failures = false;
    if (argc > 4) {
        printf("will report failures %i\n", iterations);
        report_failures = true;
    }
    // load data
    std::vector<float> col;
    printf("parsing %s\n", csv_path);
    load_csv(csv_path, {3}, col);

    float* d_input = vector_to_gpu(col);
    float* d_output = alloc_gpu<float>(col.size() + 1);

    // gen predicate mask
    size_t one_count;
    auto pred = gen_predicate(
        col, +[](float f) { return f > 200; }, &one_count);
    // mask from pattern instead
    bool use_pattern_mask = true;
    int pattern_length = 8;
    uint32_t pattern = 0b10110010<<(32-pattern_length);
    if (use_pattern_mask) {
        one_count = 0;
        generate_mask_pattern(&pred[0], 0, pred.size(), pattern, pattern_length, &one_count);
    }
    // put predicate mask on gpu
    uint8_t* d_mask = vector_to_gpu(pred);

    printf("line count: %zu, one count: %zu, percentage: %f\n", col.size(), one_count, (double)one_count / col.size());

    // gen cpu side validation
    std::vector<float> validation;
    validation.resize(col.size());
    size_t out_length = generate_validation(&col[0], &pred[0], &validation[0], col.size());
    float* d_validation = vector_to_gpu(validation);

    // prepare candidates for benchmark
    intermediate_data id{col.size(), 1024, 8}; // setup shared intermediate data

    std::vector<std::pair<std::string, std::function<float()>>> benchs;

    benchs.emplace_back("bench1_base_variant", [&]() { return bench1_base_variant(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024); });
    benchs.emplace_back("bench2_base_variant_skipping", [&]() {
         return bench2_base_variant_skipping(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back(
        "bench3_3pass_streaming", [&]() { return bench3_3pass_streaming(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024); });
    benchs.emplace_back("bench4_3pass_optimized_read_non_skipping_cub_pss", [&]() {
         return bench4_3pass_optimized_read_non_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back("bench5_3pass_optimized_read_skipping_partial_pss", [&]() {
        return bench5_3pass_optimized_read_skipping_partial_pss(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back("bench6_3pass_optimized_read_skipping_two_phase_pss", [&]() {
        return bench6_3pass_optimized_read_skipping_two_phase_pss(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back("bench7_3pass_optimized_read_skipping_cub_pss", [&]() {
        return bench7_3pass_optimized_read_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back("bench8_cub_flagged", [&]() { return bench8_cub_flagged(&id, d_input, d_mask, d_output, col.size()); });

    if (use_pattern_mask) {
        benchs.emplace_back("bench9_pattern", [&]() { return bench9_pattern(&id, d_input, pattern, pattern_length, d_output, col.size(), 1024, 2048, 64); });
    }

    // run benchmark
    std::vector<float> timings(benchs.size(), 0.0f);
    for (int it = 0; it < iterations; it++) {
        for (size_t i = 0; i < benchs.size(); i++) {
            timings[i] += benchs[i].second();
            size_t failure_count;
            if (!validate(&id, d_validation, d_output, out_length, report_failures, &failure_count)) {
                fprintf(stderr, "validation failure in bench %s, run %i: %zu failures\n", benchs[i].first.c_str(), it, failure_count);
                // exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < benchs.size(); i++) {
        std::cout << "benchmark " << benchs[i].first << " time (ms): " << (double)timings[i] / iterations << std::endl;
    }
    return 0;
}
