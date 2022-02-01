#include <algorithm>
#include <bitset>
#include <bit>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <thread>
#include "data_generator.cuh"
#include <vector>

// #define DISABLE_CUDA_TIME
#include "cuda_time.cuh"
#include "cuda_try.cuh"

#include "csv_loader.hpp"
#include "utils.cuh"
#include "data_generator.cuh"
#include "benchmarks.cuh"
#include "kernels/data_generator.cuh"
#include <unistd.h>

typedef uint64_t input_data_type;
static float threshold = 200;
bool predicate_function(input_data_type f)
{
    return f > threshold;
}
int main(int argc, char** argv)
{
    int lines = 1000;
    const char* csv_path = "../res/Arade_1.csv";
    int iterations = 100;
    bool report_failures = false;
    int chunk_length_max = 8192;
    int chunk_length_min = 32;
    int grid_size_max = 2048;
    int grid_size_min = 32;
    int block_size_max = 1024;
    int block_size_min = 32;
    bool use_csv = false;
    bool use_pattern_mask = true;
    int pattern_length = 8;
    uint32_t pattern;
    float selectivity = 0.5;
    bool use_uniform = false;
    bool use_zipf = false;
    int option;
    while ((option = getopt(argc, argv, ":zurd:l:i:f:p:s:t:g:m:c:k:b:n:")) != -1) {
        switch (option) {
            case 'g': {
                grid_size_max = atoi(optarg);
                fprintf(stderr, "using max grid size %i\n", grid_size_max);
            } break;
            case 'm': {
                grid_size_min = atoi(optarg);
                fprintf(stderr, "using min grid size %i\n", grid_size_min);
                if (std::popcount((uint32_t)grid_size_min) != 1) {
                    error("min grid size has to be a power of two\n");
                }
            } break;
            case 'b': {
                block_size_max = atoi(optarg);
                fprintf(stderr, "using max block size %i\n", block_size_max);
                if (std::popcount((uint32_t)block_size_max) != 1 || block_size_max > 1024 || block_size_max < 32) {
                    error("max block size has to be a power of two >= 32 and <= 1024 \n");
                }
            } break;
            case 'n': {
                block_size_min = atoi(optarg);
                fprintf(stderr, "using min block size %i\n", block_size_min);
                if (std::popcount((uint32_t)block_size_min) != 1 || block_size_min > 1024 || block_size_min < 32) {
                    error("min block size has to be a power of two >= 32 and <= 1024 \n");
                }
            } break;
            case 'd': {
                int device = atoi(optarg);
                fprintf(stderr, "setting device numer to %i\n", device);
                CUDA_TRY(cudaSetDevice(device));
            } break;
            case 'c': {
                chunk_length_max = atoi(optarg);
                fprintf(stderr, "setting max chunk length  to %i\n", chunk_length_max);
                if (std::popcount((uint32_t)chunk_length_max) != 1 || chunk_length_max < 32) {
                    error("max chunk length has to be a power of two >= 32\n");
                }
            } break;
            case 'k': {
                chunk_length_min = atoi(optarg);
                fprintf(stderr, "setting min chunk length  to %i\n", chunk_length_min);
                if (std::popcount((uint32_t)chunk_length_min) != 1 || chunk_length_min < 32) {
                    error("min chunk length has to be a power of two >= 32\n");
                }
            } break;
            case 'l': {
                lines = atoi(optarg);
                fprintf(stderr, "setting line count to %i\n", lines);
            } break;
            case 'i': {
                iterations = atoi(optarg);
                fprintf(stderr, "setting iteration count to %i\n", iterations);
            } break;
            case 'r': {
                fprintf(stderr, "will report failures\n");
                report_failures = true;
            } break;
            case 'f': {
                csv_path = optarg;
                lines = 0;
            } break;
            case 'p': {
                pattern_length = atoi(optarg);
                if (pattern_length > 32 || pattern_length < 1) pattern_length = 8;
                fprintf(stderr, "setting pattern length to %i\n", pattern_length);
            } break;
            case 's': {
                selectivity = atof(optarg);
                fprintf(stderr, "setting selectivity to %f\n", selectivity);
            } break;
            case 't': {
                threshold = atof(optarg);
                fprintf(stderr, "setting value threshold to%f\n", threshold);
                use_csv = true;
            } break;
            case 'z': {
                fprintf(stderr, "using zipf mask\n");
                use_zipf = true;
            } break;
            case 'u': {
                fprintf(stderr, "using uniform mask\n");
                use_uniform = true;
            } break;
            case ':': {
                fprintf(stderr, "-%c needs a value\n", optopt);
                exit(-1);
            } break;
            case '?': { // used for some unknown options
                fprintf(stderr, "unknown option: %c\n", optopt);
                exit(-1);
            } break;
        }
    }
    if (use_zipf || use_uniform || use_csv) {
        use_pattern_mask = false;
    }
    if (use_zipf + use_uniform + use_csv + use_pattern_mask != 1) {
        error("can only use one mask type\n");
    }
    if (use_pattern_mask) {
        int pattern_one_count = pattern_length * selectivity;
        fast_prng rng(42);
        pattern = 0;
        while (pattern_one_count > 0) {
            int i = rng.rand() % pattern_length;
            if (((pattern >> (31 - i)) & 0x1) == 0) {
                pattern_one_count--;
                pattern |= 1 << (31 - i);
            }
        }
        // generate_mask_uniform((uint8_t*)&pattern, 0, 4, selectivity);
        // pattern = pattern << (32 - pattern_length);
        std::bitset<32> pattern_bitset(pattern); // load data
        std::stringstream ss;
        ss << pattern_bitset;
        std::cerr << "pattern: " << ss.str().substr(0, pattern_length) << "\n";
    }
    std::vector<input_data_type> col;
    if (!use_csv) {
        fprintf(stderr, "generating %i lines of input\n", lines);
        col.resize(lines);
        generate_mask_uniform((uint8_t*)&col[0], 0, lines * sizeof(input_data_type), 0.5);
    }
    else {
        fprintf(stderr, "parsing %s\n", csv_path);
        load_csv(csv_path, {3}, col);
        if (lines > 0) {
            col.resize(std::min(col.size(), static_cast<size_t>(lines)));
        }
    }
    input_data_type* d_input = vector_to_gpu(col);
    input_data_type* d_output = alloc_gpu<input_data_type>(col.size() + 1);

    // gen predicate mask
    size_t one_count = 0;
    std::vector<uint8_t> pred;
    if (use_csv) {
        pred = gen_predicate(col, predicate_function, &one_count);
    }
    if (use_uniform) {
        pred.resize(ceildiv(col.size(), 8));
        generate_mask_uniform(&pred[0], 0, pred.size(), selectivity, &one_count);
    }
    if (use_zipf) {
        pred.resize(ceildiv(col.size(), 8));
        generate_mask_zipf(&pred[0], pred.size(), 0, pred.size(), &one_count);
    }
    if (use_pattern_mask) {
        pred.resize(ceildiv(col.size(), 8));
        // mask from pattern instead
        generate_mask_pattern(&pred[0], 0, pred.size(), pattern, pattern_length, &one_count);
    }
    // make sure unused bits in bitmask are 0
    int unused_bits = overlap(col.size(), 8);

    if (unused_bits) {
        one_count -= std::popcount(((uint32_t)pred.back() << (8 - unused_bits)) & 0xFF);
        pred.back() >>= unused_bits;
        pred.back() <<= unused_bits;
    }

    // put predicate mask on gpu
    uint8_t* d_mask = vector_to_gpu(pred);

    fprintf(stderr, "line count: %zu, one count: %zu, percentage: %f\n", col.size(), one_count, (double)one_count / col.size());

    // gen cpu side validation
    std::vector<input_data_type> validation;
    validation.resize(col.size());
    size_t out_length = generate_validation(&col[0], &pred[0], &validation[0], col.size());
    input_data_type* d_validation = vector_to_gpu(validation);

    fprintf(stderr, "starting benchmark\n");

    // prepare candidates for benchmark
    intermediate_data id{col.size(), chunk_length_min, 8, (input_data_type*)NULL}; // setup shared intermediate data

    std::vector<std::pair<std::string, std::function<timings(int, int, int)>>> benchs;

    benchs.emplace_back(
        "bench1_base_variant", [&](int cs, int bs, int gs) { return bench1_base_variant(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs); });
    benchs.emplace_back("bench2_base_variant_skipping", [&](int cs, int bs, int gs) {
        return bench2_base_variant_skipping(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    // benchs.emplace_back(
    //     "bench3_3pass_streaming", [&](int cs, int bs, int gs) { return bench3_3pass_streaming(&id, d_input, d_mask, d_output, col.size(), 1024, bs,
    //     gs);
    //     });
    benchs.emplace_back("bench4_3pass_optimized_read_non_skipping_cub_pss", [&](int cs, int bs, int gs) {
        return bench4_3pass_optimized_read_non_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench5_3pass_optimized_read_skipping_partial_pss", [&](int cs, int bs, int gs) {
        return bench5_3pass_optimized_read_skipping_partial_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench6_3pass_optimized_read_skipping_two_phase_pss", [&](int cs, int bs, int gs) {
        return bench6_3pass_optimized_read_skipping_two_phase_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench7_3pass_optimized_read_skipping_cub_pss", [&](int cs, int bs, int gs) {
        return bench7_3pass_optimized_read_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench8_cub_flagged", [&](int cs, int bs, int gs) { return bench8_cub_flagged(&id, d_input, d_mask, d_output, col.size()); });
    if (use_pattern_mask) {
        benchs.emplace_back("bench9_pattern", [&](int cs, int bs, int gs) {
            return bench9_pattern(&id, d_input, pattern, pattern_length, d_output, col.size(), cs, bs, gs);
        });
    }
    benchs.emplace_back("bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss", [&](int cs, int bs, int gs) {
        return bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });

    std::cout << "benchmark;chunk_length;block_size;grid_size;time_popc;time_pss1;time_pss2;time_proc;time_total" << std::endl;
    // run benchmark
    for (int grid_size = grid_size_min; grid_size <= grid_size_max; grid_size *= 2) {
        for (int block_size = block_size_min; block_size <= block_size_max; block_size *= 2) {
            for (int chunk_length = chunk_length_min; chunk_length <= chunk_length_max; chunk_length *= 2) {
                std::vector<timings> timings(benchs.size());
                for (int it = 0; it < iterations; it++) {
                    for (size_t i = 0; i < benchs.size(); i++) {
                        timings[i] += benchs[i].second(chunk_length, block_size, grid_size);
                        size_t failure_count;
                        if (!validate(&id, d_validation, d_output, out_length, report_failures, &failure_count)) {
                            fprintf(
                                stderr, "validation failure in bench %s (%d, %d, %d), run %i: %zu failures\n", benchs[i].first.c_str(), chunk_length,
                                block_size, grid_size, it, failure_count);
                            // exit(EXIT_FAILURE);
                        }
                    }
                }
                for (int i = 0; i < benchs.size(); i++) {
                    std::cout << benchs[i].first << ";" << chunk_length << ";" << block_size << ";" << grid_size << ";"
                              << timings[i] / static_cast<float>(iterations) << std::endl;
                }
            }
        }
    }
    return 0;
}
