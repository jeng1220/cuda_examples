#include "cast.h"
#include "dtype.h"
#include "util.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define BLOCK 256
#define L(x) __launch_bounds__(x)

__global__ void L(BLOCK) cast1(fp16_t* __restrict__ dst, const float* __restrict__ src, size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;

    #pragma unroll
    for (; i < size; i += total_threads) {
    //if (i < size) {
        dst[i] = static_cast<fp16_t>(src[i]);
    }
}

void cast1_hub(void* dst, void* src0, void* src1, size_t count)
{
    int block = BLOCK;
    int grid = (count + block - 1) / block;
    cast1<<<grid, block>>>(reinterpret_cast<fp16_t*>(dst),
        reinterpret_cast<float*>(src0), count);
}

__global__ void L(BLOCK) cast2(fp16x2_t* __restrict__ dst, const float2* __restrict__ src, size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;

    #pragma unroll
    for (; i < size; i += total_threads) {
    //if (i < size) {
        float2 s = src[i];
        fp16x2_t d;
        d.x = static_cast<fp16_t>(s.x);
        d.y = static_cast<fp16_t>(s.y);
        dst[i] = d;
    }
}

void cast2_hub(void* dst, void* src0, void* src1, size_t count)
{
    int block = BLOCK;
    int vec_count = count / 2;
    int grid = (vec_count + block - 1) / block;
    cast2<<<grid, block>>>(reinterpret_cast<fp16x2_t*>(dst),
        reinterpret_cast<float2*>(src0), vec_count);
}

__global__ void L(BLOCK) cast4(fp16x4_t* __restrict__ dst, const float4* __restrict__ src, size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;

    #pragma unroll
    for (; i < size; i += total_threads) {
    //if (i < size) {
        float4 s = src[i];
        fp16x4_t d;
        d.x = __half_as_ushort(static_cast<fp16_t>(s.x));
        d.y = __half_as_ushort(static_cast<fp16_t>(s.y));
        d.z = __half_as_ushort(static_cast<fp16_t>(s.z));
        d.w = __half_as_ushort(static_cast<fp16_t>(s.w));
        dst[i] = d;
    }
}

void cast4_hub(void* dst, void* src0, void* src1, size_t count)
{
    int block = BLOCK;
    int vec_count = count / 4;
    int grid = (vec_count + block - 1) / block;
    cast4<<<grid, block>>>(reinterpret_cast<fp16x4_t*>(dst),
        reinterpret_cast<float4*>(src0), vec_count);
}

#define PACK(dst, src0, src1)                                                                                \
do {                                                                                                         \
    dst = (__half_as_ushort(static_cast<fp16_t>(src0)) << 16) | __half_as_ushort(static_cast<fp16_t>(src1)); \
} while(0)

__global__ void L(BLOCK) cast8(fp16x8_t* __restrict__ dst, const float4* __restrict__ src, size_t size)
{
#if 1
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    //size_t total_threads = gridDim.x * blockDim.x;

    //for (; i < size; i += total_threads) {
    fp16x8_t d;
    if (i < size) {
        float4 s0 = src[2 * i];
        PACK(d.x, s0.y, s0.x);
        PACK(d.y, s0.w, s0.z);
        float4 s1 = src[2 * i + 1];
        PACK(d.z, s1.y, s1.x);
        PACK(d.w, s1.w, s1.z);
        dst[i] = d;     
    }
#else    
    float4 s0, s1;
    size_t read_i = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    if (read_i < size) {
        s0 = src[read_i];
    }

    read_i += blockDim.x;
    if (read_i < size) {
        s1 = src[read_i];
    }

    uint2 d0, d1;
    PACK(d0.x, s0.y, s0.x);
    PACK(d0.y, s0.w, s0.z);
    PACK(d1.x, s1.y, s1.x);
    PACK(d1.y, s1.w, s1.z);

    __shared__ uint2 smem[2 * BLOCK];

    smem[             threadIdx.x] = d0;
    smem[blockDim.x + threadIdx.x] = d1;

    __syncthreads();

    d0 = smem[2 * threadIdx.x];
    d1 = smem[2 * threadIdx.x + 1];

    fp16x8_t d;
    d.x = d0.x;
    d.y = d0.y;
    d.z = d1.x;
    d.w = d1.y;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size >> 1)) { 
      dst[i] = d;     
    }
#endif
}

void cast8_hub(void* dst, void* src0, void* src1, size_t count)
{
    int block = BLOCK;
    int vec_count = count / 8;
    //int vec_count = count / 4;
    int grid = (vec_count + block - 1) / block;
    cast8<<<grid, block>>>(reinterpret_cast<fp16x8_t*>(dst),
        reinterpret_cast<float4*>(src0), vec_count);
}

void cast_thrust(void* dst, void* src0, void* src1, size_t count)
{
    float* src_ptr = reinterpret_cast<float*>(src0);
    fp16_t* dst_ptr = reinterpret_cast<fp16_t*>(dst);
    thrust::copy(thrust::device, src_ptr, src_ptr + count, dst_ptr);
}

void cast_perf(size_t count, int itr)
{
    size_t read = sizeof(float) * count;
    size_t write = sizeof(fp16_t) * count;
    size_t bytes = read + write;
    info(bytes);

    size_t free, total;
    CK(cudaMemGetInfo(&free, &total));

    int num = itr;
    while (bytes * num >= free) {
        num /= 2;
    }

    ProfileBuffer buff(kCAST, num, count);

    ADD_PROFILE(buff, itr, cast_thrust);
    ADD_PROFILE(buff, itr, cast1_hub);
    ADD_PROFILE(buff, itr, cast2_hub);
    ADD_PROFILE(buff, itr, cast4_hub);
    ADD_PROFILE(buff, itr, cast8_hub);
}
