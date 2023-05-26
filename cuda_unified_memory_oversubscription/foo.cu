#include <cstdio>
#include <string>

#include <cuda_runtime_api.h>

void CheckCudaError(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        printf("LINE: %d, %s\n", line, cudaGetErrorString(err));
    }
}

#define CHECK(x) CheckCudaError((x), __LINE__)

int main (int argc, const char* argv[]) {
    if (argc < 3) {
        printf("usage:\n"
            "$ foo <itr, integer> <size, integer> <use 1)cudaMemset or 0) memset, integer>\n"
            "$ foo 85 1000 1\n");
        return 1;
    }

    int itr = std::stoi(argv[1]);
    printf("%d times allocations\n", itr);

    size_t size = std::stoi(argv[2]);
    size *= 1024; // KB
    size *= 1024; // MB
    printf("allocate %zu B at a time\n", size);
    printf("allocate %zu B totally\n", size * itr);

    int use_cudamemset = std::stoi(argv[3]);
    printf("use %s to access the buffers\n",
        (use_cudamemset)?"cudaMemset":"memset"
    );

    CHECK(cudaSetDevice(0));
    for (int i = 0; i < itr; ++i) {
	void* ptr;
        size_t free, total;
        CHECK(cudaMemGetInfo(&free, &total));
        printf("free: %zu, total: %zu\n", free, total);
        CHECK(cudaMallocManaged((void**)&ptr, size));
        if (use_cudamemset) {
            CHECK(cudaMemset(ptr, 0, size));
        } 
        else {
            memset(ptr, 0, size);
        }
    }
    while(1);
    return 0;
}
