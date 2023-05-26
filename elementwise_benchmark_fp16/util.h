#pragma once
#include "dtype.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <string>

#define KB(x) (static_cast<double>(x) * 1e-3)
#define MB(x) (static_cast<double>(x) * 1e-6)
#define GB(x) (static_cast<double>(x) * 1e-9)

#define CK(x)                                                                         \
do {                                                                                  \
    if (x != cudaSuccess) {                                                           \
        printf("%s:%u, %s(%d)\n", __FILE__, __LINE__, cudaGetErrorString(x), (int)x); \
        exit(-1);                                                                     \
    }                                                                                 \
} while(0)

void info(size_t bytes);
void** device_alloc_2d(int num, size_t size);
void device_free_2d(void** ptr);
void init(float* src, float value, size_t count);
void init(fp16_t* src, fp16_t value, size_t count);

enum ProfileItem {
    kCAST,
    kADD,
};

struct ProfileBuffer {
    ProfileBuffer(ProfileItem item, int num, size_t count) {
        this->num = num;
        this->count = count;
        if (item == kCAST) {
            a = device_alloc_2d(num, count * sizeof(fp16_t));
            b = device_alloc_2d(num, count * sizeof(float));
            c = reinterpret_cast<void**>(calloc(num, sizeof(void*)));
            for (int i = 0 ; i < num; ++i) {
                init(reinterpret_cast<float*>(b[i]), 1.f, count);    
            }
        }
        else if (item == kADD) {
            a = device_alloc_2d(num, count * sizeof(fp16_t));
            b = device_alloc_2d(num, count * sizeof(fp16_t));
            c = device_alloc_2d(num, count * sizeof(fp16_t));
            for (int i = 0; i < num; ++i) {
                init(reinterpret_cast<fp16_t*>(b[i]), __float2half(0.75f), count);
                init(reinterpret_cast<fp16_t*>(c[i]), __float2half(0.25f), count);
            }
        }
    }
    ~ProfileBuffer() {
        device_free_2d(a);
        device_free_2d(b);
        if (c[0]) {
            device_free_2d(c);
        }
        else {
            free(reinterpret_cast<void*>(c));
        }
    }
    void **a;
    void **b;
    void **c;
    int num;
    size_t count;
};

void verify(fp16_t* ptr, size_t count);

void profile(ProfileBuffer& buff, int itr, std::string name, 
    void(*hub)(void* dst, void* src0, void* src1, size_t count));

#define ADD_PROFILE(buff, itr, func) profile(buff, itr, #func, func)
