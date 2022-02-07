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

static float threshold = 200;
template <typename T> bool predicate_function(T f)
{
    return f > threshold;
}

struct opts {
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
    bool use_clustering = false;
    bool use_8byte_data = false;
};

template <typename input_data_type> void bench(opts o)
{
    if (o.use_pattern_mask) {
        int pattern_one_count = o.pattern_length * o.selectivity;
        fast_prng rng(42);
        o.pattern = 0;
        while (pattern_one_count > 0) {
            int i = rng.rand() % o.pattern_length;
            if (((o.pattern >> (31 - i)) & 0x1) == 0) {
                pattern_one_count--;
                o.pattern |= 1 << (31 - i);
            }
        }
        // generate_mask_uniform((uint8_t*)&pattern, 0, 4, selectivity);
        // pattern = pattern << (32 - pattern_length);
        std::bitset<32> pattern_bitset(o.pattern); // load data
        std::stringstream ss;
        ss << pattern_bitset;
        std::cerr << "pattern: " << ss.str().substr(0, o.pattern_length) << "\n";
    }
    std::vector<input_data_type> col;
    if (!o.use_csv) {
        fprintf(stderr, "generating %i lines of input\n", o.lines);
        col.resize(o.lines);
        generate_mask_uniform((uint8_t*)&col[0], 0, o.lines * sizeof(input_data_type), 0.5);
    }
    else {
        fprintf(stderr, "parsing %s\n", o.csv_path);
        load_csv(o.csv_path, {3}, col);
        if (o.lines > 0) {
            col.resize(std::min(col.size(), static_cast<size_t>(o.lines)));
        }
    }
    input_data_type* d_input = vector_to_gpu(col);
    input_data_type* d_output = alloc_gpu<input_data_type>(col.size() + 1);

    // gen predicate mask
    size_t one_count = 0;
    std::vector<uint8_t> pred;
    if (o.use_csv) {
        pred = gen_predicate(col, predicate_function<input_data_type>, &one_count);
    }
    if (o.use_uniform) {
        pred.resize(ceildiv(col.size(), 8));
        generate_mask_uniform(&pred[0], 0, pred.size(), o.selectivity, &one_count);
    }
    if (o.use_zipf) {
        pred.resize(ceildiv(col.size(), 8));
        generate_mask_zipf(&pred[0], pred.size(), 0, pred.size(), &one_count);
    }
    if (o.use_clustering) {
        pred = generate_mask_clustering(o.selectivity, 1, ceil2mult(col.size(), 8), &one_count);
    }
    if (o.use_pattern_mask) {
        pred.resize(ceildiv(col.size(), 8));
        // mask from pattern instead
        generate_mask_pattern(&pred[0], 0, pred.size(), o.pattern, o.pattern_length, &one_count);
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
    intermediate_data id{col.size(), o.chunk_length_min, 8, (input_data_type*)NULL}; // setup shared intermediate data

    std::vector<std::pair<std::string, std::function<timings(int, int, int)>>> benchs;

    benchs.emplace_back(
        "bench1_base_variant", [&](int cs, int bs, int gs) { return bench1_base_variant(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs); });
    benchs.emplace_back("bench2_base_variant_skipping", [&](int cs, int bs, int gs) {
        return bench2_base_variant_skipping(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench3_3pass_streaming", [&](int cs, int bs, int gs) {
        return bench3_3pass_streaming(&id, d_input, d_mask, d_output, col.size(), 1024, bs, gs);
    });
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
    if (o.use_pattern_mask) {
        benchs.emplace_back("bench9_pattern", [&](int cs, int bs, int gs) {
            return bench9_pattern(&id, d_input, o.pattern, o.pattern_length, d_output, col.size(), cs, bs, gs);
        });
    }
    benchs.emplace_back("bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss", [&](int cs, int bs, int gs) {
        return bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });

    std::cout << "benchmark;chunk_length;block_size;grid_size;time_popc;time_pss1;time_pss2;time_proc;time_total" << std::endl;
    // run benchmark
    for (int grid_size = o.grid_size_min; grid_size <= o.grid_size_max; grid_size *= 2) {
        for (int block_size = o.block_size_min; block_size <= o.block_size_max; block_size *= 2) {
            for (int chunk_length = o.chunk_length_min; chunk_length <= o.chunk_length_max; chunk_length *= 2) {
                std::vector<timings> timings(benchs.size());
                for (int it = 0; it < o.iterations; it++) {
                    for (size_t i = 0; i < benchs.size(); i++) {
                        timings[i] += benchs[i].second(chunk_length, block_size, grid_size);
                        size_t failure_count;
                        if (!validate(&id, d_validation, d_output, out_length, o.report_failures, &failure_count)) {
                            fprintf(
                                stderr, "validation failure in bench %s (%d, %d, %d), run %i: %zu failures\n", benchs[i].first.c_str(), chunk_length,
                                block_size, grid_size, it, failure_count);
                            // exit(EXIT_FAILURE);
                        }
                    }
                }
                for (int i = 0; i < benchs.size(); i++) {
                    std::cout << benchs[i].first << ";" << chunk_length << ";" << block_size << ";" << grid_size << ";"
                              << timings[i] / static_cast<float>(o.iterations) << std::endl;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    opts o{};
    int option;
    while ((option = getopt(argc, argv, ":zuer8d:l:i:f:p:s:t:g:m:c:k:b:n:")) != -1) {
        switch (option) {
            case '8': {
                fprintf(stderr, "using 8 byte data size\n");
                o.use_8byte_data = true;
            } break;
            case 'g': {
                o.grid_size_max = atoi(optarg);
                fprintf(stderr, "using max grid size %i\n", o.grid_size_max);
            } break;
            case 'm': {
                o.grid_size_min = atoi(optarg);
                fprintf(stderr, "using min grid size %i\n", o.grid_size_min);
                if (std::popcount((uint32_t)o.grid_size_min) != 1) {
                    error("min grid size has to be a power of two\n");
                }
            } break;
            case 'b': {
                o.block_size_max = atoi(optarg);
                fprintf(stderr, "using max block size %i\n", o.block_size_max);
                if (std::popcount((uint32_t)o.block_size_max) != 1 || o.block_size_max > 1024 || o.block_size_max < 32) {
                    error("max block size has to be a power of two >= 32 and <= 1024 \n");
                }
            } break;
            case 'n': {
                o.block_size_min = atoi(optarg);
                fprintf(stderr, "using min block size %i\n", o.block_size_min);
                if (std::popcount((uint32_t)o.block_size_min) != 1 || o.block_size_min > 1024 || o.block_size_min < 32) {
                    error("min block size has to be a power of two >= 32 and <= 1024 \n");
                }
            } break;
            case 'd': {
                int device = atoi(optarg);
                fprintf(stderr, "setting device numer to %i\n", device);
                CUDA_TRY(cudaSetDevice(device));
            } break;
            case 'c': {
                o.chunk_length_max = atoi(optarg);
                fprintf(stderr, "setting max chunk length  to %i\n", o.chunk_length_max);
                if (std::popcount((uint32_t)o.chunk_length_max) != 1 || o.chunk_length_max < 32) {
                    error("max chunk length has to be a power of two >= 32\n");
                }
            } break;
            case 'k': {
                o.chunk_length_min = atoi(optarg);
                fprintf(stderr, "setting min chunk length  to %i\n", o.chunk_length_min);
                if (std::popcount((uint32_t)o.chunk_length_min) != 1 || o.chunk_length_min < 32) {
                    error("min chunk length has to be a power of two >= 32\n");
                }
            } break;
            case 'l': {
                o.lines = atoi(optarg);
                fprintf(stderr, "setting line count to %i\n", o.lines);
            } break;
            case 'i': {
                o.iterations = atoi(optarg);
                fprintf(stderr, "setting iteration count to %i\n", o.iterations);
            } break;
            case 'r': {
                fprintf(stderr, "will report failures\n");
                o.report_failures = true;
            } break;
            case 'f': {
                o.csv_path = optarg;
                o.lines = 0;
            } break;
            case 'p': {
                o.pattern_length = atoi(optarg);
                if (o.pattern_length > 32 || o.pattern_length < 1) o.pattern_length = 8;
                fprintf(stderr, "setting pattern length to %i\n", o.pattern_length);
            } break;
            case 's': {
                o.selectivity = atof(optarg);
                fprintf(stderr, "setting selectivity to %f\n", o.selectivity);
            } break;
            case 't': {
                threshold = atof(optarg);
                fprintf(stderr, "setting value threshold to%f\n", threshold);
                o.use_csv = true;
            } break;
            case 'z': {
                fprintf(stderr, "using zipf mask\n");
                o.use_zipf = true;
            } break;
            case 'u': {
                fprintf(stderr, "using uniform mask\n");
                o.use_uniform = true;
            } break;
            case 'e': {
                fprintf(stderr, "using clustEring mask\n");
                o.use_clustering = true;
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
    if (o.use_zipf || o.use_uniform || o.use_csv || o.use_clustering) {
        o.use_pattern_mask = false;
    }
    if (o.use_zipf + o.use_uniform + o.use_csv + o.use_pattern_mask + o.use_clustering != 1) {
        error("can only use one mask type\n");
    }
    if (o.use_8byte_data) {
        bench<uint64_t>(o);
    }
    else {
        bench<uint32_t>(o);
    }
    return 0;
}
