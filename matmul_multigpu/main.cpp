#include <cmath>

#include <cuda_profiler_api.h>

#include "sgemm.h"

void sgemm_cpu(float* a, float* b, float* c, int m, int n, int k) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            float c_value = 0.f;
            #pragma omp seq
            for (int kk = 0; kk < k; ++kk) {
                c_value += a[j * k + kk] * b[kk * n + i];
            }
            c[j * n + i] = c_value;
        }
    }
}

void expect_allclose(float* actual, float* desired, size_t count, double rtol=1e-3, double atol=1e-3) {
    auto result = std::equal(actual, actual + count, desired, [=](float actual_value, float desired_value){
        auto delta = std::abs(actual_value - desired_value);
        auto threshold = atol + rtol * std::abs(desired_value);
        if (delta > threshold) { 
            std::cerr << "desired " << desired_value << ", " << "actual " << actual_value << std::endl;
            return false;
        }
        return true;
    });

    if (result) std::cout << "PASSED" << std::endl;
    else std::cout << "FAILURE" << std::endl;
}

int main () {
    int sz = 32768;
    std::cout << "matrics size: " << sz << "^2" << std::endl;
    int m = sz;
    int n = sz;
    int k = sz;
    auto mem_type = HostMemoryType::PINNED;
    auto verify = false;
    auto init_mem = verify ? MemoryInitType::RANDOM : MemoryInitType::EMPTY;

    Mat mat_a(k, m, mem_type, init_mem);
    Mat mat_b(n, k, mem_type, init_mem);
    Mat mat_c(n, m, mem_type, MemoryInitType::EMPTY);
    Mat mat_ref(n, m, mem_type, MemoryInitType::EMPTY);

    CHECK(cudaProfilerStart());
    sgemm_multi_gpu(mat_a.data(), mat_b.data(), mat_c.data(), m, n, k, mem_type);
    sgemm_single_gpu(mat_a.data(), mat_b.data(), mat_ref.data(), m, n, k, mem_type);
    CHECK(cudaProfilerStop());
    if (verify) {
        expect_allclose(mat_c.data(), mat_ref.data(), n * m);
    }

    /*
    // too slow if matrics are large, skip
    Mat mat_cpu_ref(n, m, HostMemoryType::LOCK_FREE, MemoryInitType::EMPTY);
    sgemm_cpu(mat_a.data(), mat_b.data(), mat_cpu_ref.data(), m, n, k);
    expect_allclose(mat_c.data(), mat_cpu_ref.data(), n * m);
    */
    return 0;
}
