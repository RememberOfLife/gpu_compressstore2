#pragma once
#include <vector>
#include <cstdint>

template <typename T>
void cpu_buffer_print(T* h_buffer, uint32_t offset, uint32_t length)
{
    for (int i = offset; i < offset+length; i++) {
        std::bitset<sizeof(T)*8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
}

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length*sizeof(T)));
    CUDA_TRY(cudaMemcpy(h_buffer, d_buffer+offset, length*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < length; i++) {
        std::bitset<sizeof(T)*8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

template <typename T>
T* vector_to_gpu(const std::vector<T>& vec){
    T* buff;
    const auto size = vec.size() * sizeof(T);
    CUDA_TRY(cudaMalloc(&buff, size));
    CUDA_TRY(cudaMemcpy(buff, &vec[0], size, cudaMemcpyHostToDevice)); 
    return buff;
}

template <typename T>
std::vector<T> gpu_to_vector(T* buff, size_t length){
    std::vector<T> vec;
    const size_t size = length * sizeof(T);
    vec.resize(size);
    CUDA_TRY(cudaMemcpy(&vec[0], buff, length * sizeof(T), cudaMemcpyDeviceToHost));
    return vec;
}

template <typename T> T* alloc_gpu(size_t length){
    T* buff;
    CUDA_TRY(cudaMalloc(&buff, length * sizeof(T)));
    return buff;
}

template<typename T>
std::vector<uint8_t> gen_predicate(const std::vector<T>& col, bool (*predicate)(T value)){
    std::vector<uint8_t> predicate_bitmask{};
    predicate_bitmask.reserve(col.size() / 8);
    auto it = col.begin();
    for (int i = 0; i < col.size() / 8; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (it == col.end()) break;
            if (predicate(*it++)) {
                acc |= (1<<j);
            }
        }
        predicate_bitmask.push_back(acc);
    }
    return predicate_bitmask;
}
