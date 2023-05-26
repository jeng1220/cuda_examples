#include "util.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <omp.h>
#include <cstdio>

void info(size_t bytes)
{
    char unit[3];
    double total_lf;

    if (GB(bytes) >= 1.) {
        total_lf = GB(bytes);
        sscanf("GB", "%s", unit);
    }
    else if (MB(bytes) >= 1.) {
        total_lf = MB(bytes);
        sscanf("MB", "%s", unit);
    }
    else if (KB(bytes) >= 1.) {
        total_lf = KB(bytes);
        sscanf("KB", "%s", unit);
    }
    else {
        total_lf = (bytes);
        sscanf("B", "%s", unit);
    }

    printf("%lf, %s\n", total_lf, unit);
}

void** device_alloc_2d(int num, size_t size)
{
    void** ptr_h = reinterpret_cast<void**>(calloc(num, sizeof(void*)));
    unsigned char* ptr_d;
    CK(cudaMalloc(reinterpret_cast<void**>(&ptr_d), num * size));
    for (int i = 0; i < num; ++i) {
        ptr_h[i] = reinterpret_cast<void*>(ptr_d + i * size);
    }
    return ptr_h;
}

void device_free_2d(void** ptr)
{
    CK(cudaFree(ptr[0]));
    free(ptr);
}

void init(float* src, float value, size_t count)
{
    thrust::host_vector<float> h(count);
    for (size_t i = 0; i < count; ++i) {
        h[i] = static_cast<float>(i % 10);
    }
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(src);
    thrust::copy(h.begin(), h.end(), dev_ptr);
}

void init(fp16_t* src, fp16_t value, size_t count)
{
    thrust::host_vector<fp16_t> h(count);

    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        h[i] = __float2half(static_cast<float>(i % 10) * __half2float(value));
    }
    thrust::device_ptr<fp16_t> dev_ptr = thrust::device_pointer_cast(src);
    thrust::copy(h.begin(), h.end(), dev_ptr);
}

void verify(fp16_t* ptr, size_t count)
{
    thrust::host_vector<fp16_t> fp16_h(count);
    thrust::device_ptr<fp16_t> dev_ptr = thrust::device_pointer_cast(ptr);
    thrust::copy(dev_ptr, dev_ptr + count, fp16_h.begin());

    int err_count = 0;

    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        float result = __half2float(fp16_h[i]);
        float ans = static_cast<float>((i % 10));
        if (std::fabs(result - ans) > 1e-6) {
            printf("failure: result[%zu] is %f but should be %f\n",
                i, result, ans);
            fflush(stdout);
            err_count++;
            if (err_count == 10) break;
        }
    }
}

void profile(ProfileBuffer& buff, int itr, std::string name,
    void(*hub)(void* dst, void* src0, void* src1, size_t count))
{
    CK(cudaMemset(buff.a[0], 0, sizeof(fp16_t) * buff.count));

    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    CK(cudaEventRecord(start));

    for (int i = 0; i < itr; ++i) {
        int buff_id = i % buff.num;
        auto  d = buff.a[buff_id];
        auto s0 = buff.b[buff_id];
        auto s1 = buff.c[buff_id];        
        hub(d, s0, s1, buff.count);
    }

    CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop));
    CK(cudaDeviceSynchronize());

    float delta;
    CK(cudaEventElapsedTime(&delta, start, stop));
    double delta_lf = static_cast<double>(delta) / itr;
    double total = MB(buff.count * sizeof(fp16_t) * 3);
    double bw = total / delta_lf;
    printf("%s, %lf, GB/s\n", name.c_str(), bw);

    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));

    verify(reinterpret_cast<fp16_t*>(buff.a[0]), buff.count);
}
