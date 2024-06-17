
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <cublasXt.h>
#include <nvToolsExt.h>

inline void check(cudaError_t error, int line) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << ":L" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void check(cublasStatus_t status, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << cublasGetStatusString(status) << ":L" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, __LINE__)

class CpuTimer {
public:
    CpuTimer(const char str[]) : msg_(str) {
        start_ = std::chrono::high_resolution_clock::now();
        nvtxRangePush(str);
    }
    ~CpuTimer() {
        nvtxRangePop();
        auto stop = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start_);
        std::cout << msg_ << " time: " << time_span.count() << " milliseconds" << std::endl;
    }
private:
    std::string msg_;
    std::chrono::high_resolution_clock::time_point start_;
};

enum class HostMemoryType {
    LOCK_FREE,
    PINNED,
    MANAGED
};

enum class MemoryInitType {
    EMPTY,
    ZERO,
    RANDOM
};

class Mat {
public:
    Mat(int w, int h, HostMemoryType type, MemoryInitType init) : w_(w), h_(h), type_(type) {
        size_t count = static_cast<size_t>(w) * h;
        size_t sz = count * sizeof(float);
        switch(type) {
            case HostMemoryType::LOCK_FREE:
                ptr_ = new float[count];
                break;
            case HostMemoryType::PINNED:
                CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), sz));
                break;
            case HostMemoryType::MANAGED:
                CHECK(cudaMallocManaged(reinterpret_cast<void**>(&ptr_), sz));
                break;
            default:
                std::cerr << "Invalid Memory Type:L" << __LINE__ << std::endl;
                exit(EXIT_FAILURE);
        }

        switch(init) {
            case MemoryInitType::EMPTY:
                break;
            case MemoryInitType::ZERO:
                std::fill(ptr_, ptr_ + count, 0.f);
                break;
            case MemoryInitType::RANDOM:
                std::generate(ptr_, ptr_ + count, [=](){return static_cast<float>(std::rand() % 2);});
                break;
            default:
                std::cerr << "Invalid Init Type:L" << __LINE__ << std::endl;
                exit(EXIT_FAILURE);
        }
    }
    ~Mat() {
        switch(type_) {
            case HostMemoryType::LOCK_FREE:
                delete [] ptr_;
                break;
            case HostMemoryType::PINNED:
                CHECK(cudaFreeHost(ptr_));
                break;
            case HostMemoryType::MANAGED:
                CHECK(cudaFree(ptr_));
                break;
            default:
                std::cerr << "Invalid Memory Type:L" << __LINE__ << std::endl;
                exit(EXIT_FAILURE);
        }
    }
    float* data() const {
        return ptr_;
    }
    HostMemoryType type() const {
        return type_;
    }
    int width() const {
        return w_;
    }
    int height() const {
        return h_;
    }
private:
    float* ptr_ = nullptr;
    int w_ = 0;
    int h_ = 0;
    HostMemoryType type_{};
};

extern "C" {
cublasStatus_t sgemm_single_gpu(const float* a, const float* b, float* c, int m, int n, int k, HostMemoryType mem_type);
cublasStatus_t sgemm_multi_gpu(float* a, float* b, float* c, int m, int n, int k, HostMemoryType mem_type);
}
