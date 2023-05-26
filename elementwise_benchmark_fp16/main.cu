#include "dtype.h"
#include "cast.h"
#include "add.h"
#include "util.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, const char** argv)
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d, %d\n", prop.name, prop.major, prop.minor);

    size_t count = 10000000;
    if (argc > 1) {
        sscanf(argv[1], "%zu", &count);
        // padding with 8
        size_t num = (count + 7) / 8;
        count = num * 8;
    }
    printf("number of elements, %zu\n", count);
   
    int itr = 100;
    if (argc > 2) {
        itr = std::atoi(argv[2]);
    }
    printf("iterations, %d\n", itr);

    cast_perf(count, itr);
    add_perf(count, itr);
    return 0;
}
