#include "sgemm.h"

extern "C" {

cublasStatus_t sgemm_single_gpu(const float* a, const float* b, float* c, int m, int n, int k, HostMemoryType mem_type) {
    CHECK(cudaSetDevice(0));
    size_t mat_a_count = static_cast<size_t>(k) * m;
    size_t mat_b_count = static_cast<size_t>(n) * k;
    size_t mat_c_count = static_cast<size_t>(n) * m;

    auto need_copy = mem_type != HostMemoryType::MANAGED;
    float* gpu_a = nullptr;
    float* gpu_b = nullptr;
    float* gpu_c = nullptr;
    if (need_copy) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_a), mat_a_count * sizeof(float)));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_b), mat_b_count * sizeof(float)));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_c), mat_c_count * sizeof(float)));
    }
    else {
        gpu_a = const_cast<float*>(a);
        gpu_b = const_cast<float*>(b);
        gpu_c = c;
        CHECK(cudaMemPrefetchAsync(gpu_a, mat_a_count * sizeof(float), 0));
        CHECK(cudaMemAdvise(gpu_a, mat_a_count * sizeof(float), cudaMemAdviseSetReadMostly, 0));
        CHECK(cudaMemAdvise(gpu_a, mat_a_count * sizeof(float), cudaMemAdviseSetPreferredLocation, 0));
        CHECK(cudaMemPrefetchAsync(gpu_b, mat_b_count * sizeof(float), 0));
        CHECK(cudaMemAdvise(gpu_b, mat_b_count * sizeof(float), cudaMemAdviseSetReadMostly, 0));
        CHECK(cudaMemAdvise(gpu_b, mat_b_count * sizeof(float), cudaMemAdviseSetPreferredLocation, 0));
        CHECK(cudaMemPrefetchAsync(gpu_c, mat_c_count * sizeof(float), 0));
        CHECK(cudaMemAdvise(gpu_c, mat_c_count * sizeof(float), cudaMemAdviseSetPreferredLocation, 0));
        CHECK(cudaStreamSynchronize(0));
    }

    cublasHandle_t handle{};
    CHECK(cublasCreate(&handle));
    auto transa = CUBLAS_OP_N;
    auto transb = CUBLAS_OP_N;
    auto alpha = 1.f;
    auto beta = 0.f;
    int lda = k;
    int ldb = n;
    int ldc = n;
    {
        CpuTimer timer("cublasSgemm");
        if (need_copy) {
            CHECK(cudaMemcpy(gpu_a, a, mat_a_count * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(gpu_b, b, mat_b_count * sizeof(float), cudaMemcpyHostToDevice));
        }
        CHECK(cublasSgemm(handle,
            transa, transb,
            n, m, k,
            &alpha,
            gpu_b, ldb,
            gpu_a, lda,
            &beta,
            gpu_c, ldc));
        if (need_copy) {
            CHECK(cudaMemcpy(c, gpu_c, mat_c_count * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cublasDestroy(handle));
    if (need_copy) {
        CHECK(cudaFree(gpu_a));
        CHECK(cudaFree(gpu_b));
        CHECK(cudaFree(gpu_c));
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t sgemm_multi_gpu(float* a, float* b, float* c, int m, int n, int k, HostMemoryType mem_type) {
    int nb_gpu = 0;
    int gpu_id[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(cudaGetDeviceCount(&nb_gpu));
    std::cout << "number of GPUs: " << nb_gpu << std::endl;

    if (mem_type == HostMemoryType::MANAGED) {
        size_t mat_a_count = static_cast<size_t>(k) * m;
        size_t mat_b_count = static_cast<size_t>(n) * k;
        size_t mat_c_count = static_cast<size_t>(n) * m;
        auto* copy_a = a;
        auto* copy_b = b;
        auto* copy_c = c;
        for (int i = 0; i < nb_gpu; ++i) {
            CHECK(cudaMemAdvise(copy_a, mat_a_count * sizeof(float), cudaMemAdviseSetAccessedBy, i));
            CHECK(cudaMemAdvise(copy_b, mat_b_count * sizeof(float), cudaMemAdviseSetAccessedBy, i));
            CHECK(cudaMemAdvise(copy_c, mat_c_count * sizeof(float), cudaMemAdviseSetAccessedBy, i));
        }

        size_t prefetch_count_a = mat_a_count / nb_gpu;
        size_t prefetch_count_b = mat_b_count / nb_gpu;
        size_t prefetch_count_c = mat_c_count / nb_gpu;
        for (int i = 0; i < nb_gpu; ++i) {
            CHECK(cudaMemPrefetchAsync(copy_a, prefetch_count_a * sizeof(float), i));
            CHECK(cudaMemAdvise(copy_a, prefetch_count_a * sizeof(float), cudaMemAdviseSetReadMostly, i));
            CHECK(cudaMemAdvise(copy_a, prefetch_count_a * sizeof(float), cudaMemAdviseSetPreferredLocation, i));
            CHECK(cudaMemPrefetchAsync(copy_b, prefetch_count_b * sizeof(float), i));
            CHECK(cudaMemAdvise(copy_b, prefetch_count_b * sizeof(float), cudaMemAdviseSetReadMostly, i));
            CHECK(cudaMemAdvise(copy_b, prefetch_count_b * sizeof(float), cudaMemAdviseSetPreferredLocation, i));
            CHECK(cudaMemPrefetchAsync(copy_c, prefetch_count_c * sizeof(float), i));
            CHECK(cudaMemAdvise(copy_c, prefetch_count_c * sizeof(float), cudaMemAdviseSetPreferredLocation, i));
            copy_a += prefetch_count_a;
            copy_b += prefetch_count_b;
            copy_c += prefetch_count_c;
        }
        CHECK(cudaStreamSynchronize(0));
    }

    cublasXtHandle_t handle{};
    CHECK(cublasXtCreate(&handle));
    CHECK(cublasXtDeviceSelect(handle, nb_gpu, gpu_id));

    int new_block_dim = 8192;
    CHECK(cublasXtSetBlockDim(handle, new_block_dim));
    int block_dim = 0;
    CHECK(cublasXtGetBlockDim(handle, &block_dim));
    std::cout << "blockDim: " << block_dim << std::endl;

    auto transa = CUBLAS_OP_N;
    auto transb = CUBLAS_OP_N;
    auto alpha = 1.f;
    auto beta = 0.f;
    int lda = k;
    int ldb = n;
    int ldc = n;
    {
        CpuTimer timer("cublasXtSgemm");
        CHECK(cublasXtSgemm(handle,
            transa, transb,
            n, m, k,
            &alpha,
            b, ldb,
            a, lda,
            &beta,
            c, ldc));
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cublasXtDestroy(handle));

    return CUBLAS_STATUS_SUCCESS;
}

}
