#pragma once
#include <vector>
#include <cstdint>
#include <src/kernels/data_generator.cuh>

void error(const char* error)
{
    fputs(error, stderr);
    fputs("\n", stderr);
    assert(false);
    exit(EXIT_FAILURE);
}
void alloc_failure()
{
    error("memory allocation failed");
}

template <typename T>
void cpu_buffer_print(T* h_buffer, uint32_t offset, uint32_t length)
{
    for (int i = offset; i < offset + length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
}

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length * sizeof(T)));
    CUDA_TRY(cudaMemcpy(
        h_buffer, d_buffer + offset, length * sizeof(T),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

template <typename T> T* vector_to_gpu(const std::vector<T>& vec)
{
    T* buff;
    const auto size = vec.size() * sizeof(T);
    CUDA_TRY(cudaMalloc(&buff, size));
    CUDA_TRY(cudaMemcpy(buff, &vec[0], size, cudaMemcpyHostToDevice));
    return buff;
}

template <typename T> std::vector<T> gpu_to_vector(T* buff, size_t length)
{
    std::vector<T> vec;
    const size_t size = length * sizeof(T);
    vec.resize(size);
    CUDA_TRY(
        cudaMemcpy(&vec[0], buff, length * sizeof(T), cudaMemcpyDeviceToHost));
    return vec;
}

template <class T> struct dont_deduce_t {
    using type = T;
};

template <typename T> T gpu_to_val(T* d_val)
{
    T val;
    CUDA_TRY(cudaMemcpy(&val, d_val, sizeof(T), cudaMemcpyDeviceToHost));
    return val;
}

template <typename T>
void val_to_gpu(T* d_val, typename dont_deduce_t<T>::type val)
{
    CUDA_TRY(cudaMemcpy(d_val, &val, sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> T* alloc_gpu(size_t length)
{
    T* buff;
    CUDA_TRY(cudaMalloc(&buff, length * sizeof(T)));
    return buff;
}

template <typename T>
std::vector<uint8_t> gen_predicate(
    const std::vector<T>& col, bool (*predicate)(T value),
    size_t* one_count = NULL)
{
    std::vector<uint8_t> predicate_bitmask{};
    predicate_bitmask.reserve(col.size() / 8);
    auto it = col.begin();
    size_t one_count_loc = 0;
    for (int i = 0; i < col.size() / 8; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (it == col.end()) break;
            if (predicate(*it++)) {
                acc |= (1 << j);
                one_count_loc++;
            }
        }
        predicate_bitmask.push_back(acc);
    }
    if (one_count) *one_count = one_count_loc;
    return predicate_bitmask;
}
