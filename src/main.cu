#include <__clang_cuda_device_functions.h>
#include <bitset>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <thread>
#include "data_generator.cuh"
#include <vector>

#define DISABLE_CUDA_TIME
#include "cuda_time.cuh"
#include "cuda_try.cuh"

#include "csv_loader.hpp"
#include "utils.cuh"
#include "data_generator.cuh"
#include "benchmarks.cuh"
#include "kernels/data_generator.cuh"
#include <unistd.h>

static float threshold = 200;
bool predicate_function(float f)
{
    return f > threshold;
}
int main(int argc, char** argv)
{
    int lines = 0;
    const char* csv_path = "../res/Arade_1.csv";
    int iterations = 100;
    bool report_failures = false;

    bool use_pattern_mask = true;
    int pattern_length = 8;
    uint32_t pattern;
    float selectivity = 0.5;
    bool use_selectivity = true;
    int option;
    while ((option = getopt(argc, argv, ":d:l:i:f:p:s:t:r")) != -1) {
        switch (option) {
            case 'd': {
                int device = atoi(optarg);
                printf("setting device numer to %i\n", device);
                CUDA_TRY(cudaSetDevice(device));
            } break;
            case 'l': {
                lines = atoi(optarg);
                printf("setting line count to %i\n", lines);
            } break;
            case 'i': {
                iterations = atoi(optarg);
                printf("setting iteration count to %i\n", iterations);
            } break;
            case 'r': {
                printf("will report failures\n");
                report_failures = true;
            } break;
            case 'f': {
                csv_path = optarg;
                lines = 0;
            } break;
            case 'p': {
                pattern_length = atoi(optarg);
                if (pattern_length > 32 || pattern_length < 1) pattern_length = 8;
                printf("setting pattern length to %i\n", pattern_length);
                use_pattern_mask = true;
            } break;
            case 's': {
                selectivity = atof(optarg);
                printf("setting selectivity to %f\n", selectivity);
            } break;
            case 't': {
                threshold = atof(optarg);
                printf("setting value threshold to%f\n", threshold);
                use_pattern_mask = false;
                use_selectivity = false;
            } break;
            case ':': {
                printf("-%c needs a value\n", optopt);
            } break;
            case '?': { // used for some unknown options
                printf("unknown option: %c\n", optopt);
            } break;
        }
    }
    generate_mask_uniform((uint8_t*)&pattern, 0, 4, selectivity);
    pattern = pattern << (32 - pattern_length);
    std::bitset<32> pattern_bitset(pattern); // load data
    std::stringstream ss;
    ss << pattern_bitset;
    std::cout << "pattern: " << ss.str().substr(0, pattern_length) << "\n";
    std::vector<float> col;
    if (lines != 0) {
        printf("generating %i lines of input\n", lines);
        col.resize(lines);
        generate_mask_uniform((uint8_t*)&col[0], 0, lines * 4, 0.5);
    }
    else {
        printf("parsing %s\n", csv_path);
        load_csv(csv_path, {3}, col);
    }
    float* d_input = vector_to_gpu(col);
    float* d_output = alloc_gpu<float>(col.size() + 1);

    // gen predicate mask
    size_t one_count;
    std::vector<uint8_t> pred;
    if (!use_pattern_mask && !use_selectivity) {
        pred = gen_predicate(col, predicate_function, &one_count);
    }
    else {
        pred.resize(ceildiv(col.size(), 8));
        // mask from pattern instead
        one_count = 0;
        generate_mask_pattern(&pred[0], 0, pred.size(), pattern, pattern_length, &one_count);
        // make sure unused bits in bitmask are 0
        int unused_bits = overlap(col.size(), 8);
        if (unused_bits) {
            pred.back() >>= unused_bits;
            pred.back() <<= unused_bits;
        }
    }

    // put predicate mask on gpu
    uint8_t* d_mask = vector_to_gpu(pred);

    printf("line count: %zu, one count: %zu, percentage: %f\n", col.size(), one_count, (double)one_count / col.size());

    // gen cpu side validation
    std::vector<float> validation;
    validation.resize(col.size());
    size_t out_length = generate_validation(&col[0], &pred[0], &validation[0], col.size());
    float* d_validation = vector_to_gpu(validation);

    puts("starting benchmark");

    // prepare candidates for benchmark
    intermediate_data id{col.size(), 1024, 8}; // setup shared intermediate data

    std::vector<std::pair<std::string, std::function<float()>>> benchs;

    benchs.emplace_back("bench1_base_variant", [&]() { return bench1_base_variant(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024); });
    benchs.emplace_back(
        "bench2_base_variant_skipping", [&]() { return bench2_base_variant_skipping(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024); });
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
        benchs.emplace_back(
            "bench9_pattern", [&]() { return bench9_pattern(&id, d_input, pattern, pattern_length, d_output, col.size(), 1024, 256, 1024); });
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
