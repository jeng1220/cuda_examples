#include "add.h"
#include "dtype.h"
#include "util.h"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#if __CUDA_ARCH__  >= 530
#define HADD(d, e, f) do { d = e + f; } while(0)
#define HADD2(d, e, f) do { d = __hadd2(e, f); } while(0)
#else
#define HADD(d, e, f) do { d = __float2half(__half2float(e) + __half2float(f)); } while(0)
#define HADD2(d, e, f)                                         \
do {                                                           \
    d.x = __float2half(__half2float(e.x) + __half2float(f.x)); \
    d.y = __float2half(__half2float(e.y) + __half2float(f.y)); \
} while(0)
#endif

#define BLOCK 256
#define L(x) __launch_bounds__(x)

__global__ void L(BLOCK) add1(fp16_t* c, const fp16_t* __restrict__ a, const fp16_t* __restrict__ b, size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;

    #pragma unroll
    for (; i < size; i += total_threads) {
    //if (i < size) {
        HADD(c[i], a[i], b[i]);
    }
}

void add1_hub(void* dst, void* src0, void* src1, size_t count)
{
    int block = BLOCK;
    int grid = (count + block - 1) / block;
    add1<<<grid, block>>>(reinterpret_cast<fp16_t*>(dst),
        reinterpret_cast<fp16_t*>(src0), reinterpret_cast<fp16_t*>(src1), count);
}

__global__ void L(BLOCK) add2(fp16x2_t* c, const fp16x2_t* __restrict__ a, const fp16x2_t* __restrict__ b, size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;

    #pragma unroll
    for (; i < size; i += total_threads) {
    //if (i < size) {
        fp16x2_t a_f, b_f, c_f;
        a_f = a[i];
        b_f = b[i];
        HADD2(c_f, a_f, b_f);
        c[i] = c_f;
    }
}

void add2_hub(void* dst, void* src0, void* src1, size_t count)
{
    int block = BLOCK;
    int vec_count = count / 2;
    int grid = (vec_count + block - 1) / block;
    add2<<<grid, block>>>(reinterpret_cast<fp16x2_t*>(dst),
        reinterpret_cast<fp16x2_t*>(src0), reinterpret_cast<fp16x2_t*>(src1), vec_count);
}

#define UNPACK(dst, src)                                                        \
do {                                                                            \
    dst.x = __ushort_as_half(static_cast<unsigned short>((src >> 16)));         \
    dst.y = __ushort_as_half(static_cast<unsigned short>((src) & 0x0000FFFF));  \
} while(0)

#define PACK(dst, src)                                               \
do {                                                                 \
    dst = (__half_as_ushort(src.x) << 16) | __half_as_ushort(src.y); \
} while(0)

__global__ void L(BLOCK) add8(uint4* c, const uint4* a, const uint4* b, size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        uint4 a_i, b_i, c_i;
        a_i = a[i];
        b_i = b[i];
        fp16x2_t a_f, b_f, c_f;
        UNPACK(a_f, a_i.x);
        UNPACK(b_f, b_i.x);
        HADD2(c_f, a_f, b_f);
        PACK(c_i.x, c_f);

        UNPACK(a_f, a_i.y);
        UNPACK(b_f, b_i.y);
        HADD2(c_f, a_f, b_f);
        PACK(c_i.y, c_f);

        UNPACK(a_f, a_i.z);
        UNPACK(b_f, b_i.z);
        HADD2(c_f, a_f, b_f);
        PACK(c_i.z, c_f);

        UNPACK(a_f, a_i.w);
        UNPACK(b_f, b_i.w);
        HADD2(c_f, a_f, b_f);
        PACK(c_i.w, c_f);

        c[i] = c_i;     
    } 
}

void add8_hub(void* dst, void* src0, void* src1, size_t count)
{
    int block = BLOCK;
    int vec_count = count / 8;
    int grid = (vec_count + block - 1) / block;
    add8<<<grid, block>>>(reinterpret_cast<uint4*>(dst),
        reinterpret_cast<uint4*>(src0), reinterpret_cast<uint4*>(src1), vec_count);
}

void add_thrust(void* dst, void* src0, void* src1, size_t count)
{
    fp16_t* dst_ptr  = reinterpret_cast<fp16_t*>(dst);
    fp16_t* src0_ptr = reinterpret_cast<fp16_t*>(src0);
    fp16_t* src1_ptr = reinterpret_cast<fp16_t*>(src1);
    thrust::plus<int> op;
    thrust::transform(thrust::device, src0_ptr, src0_ptr + count, src1_ptr, dst_ptr, op);
}

void add_perf(size_t count, int itr)
{
    size_t read = sizeof(fp16_t) * count * 2;
    size_t write = sizeof(fp16_t) * count;
    size_t bytes = read + write;
    info(bytes);

    size_t free, total;
    CK(cudaMemGetInfo(&free, &total));

    int num = itr;
    while (bytes * num >= free) {
        num /= 2;
    }

    ProfileBuffer buff(kADD, num, count);

    ADD_PROFILE(buff, itr, add_thrust);
    ADD_PROFILE(buff, itr, add1_hub);
    ADD_PROFILE(buff, itr, add2_hub);
    ADD_PROFILE(buff, itr, add8_hub);
}