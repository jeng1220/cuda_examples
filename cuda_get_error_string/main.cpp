#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>

int main (int argc, const char* argv[]) {
  if (argc < 2) {
    printf("usage:\n"
      "$ ./main <error code, integer>\n");
    return 0;
  }
  
  auto error = static_cast<cudaError_t>(atoi(argv[1]));
  printf("%d means \"%s\"\n", error, cudaGetErrorString(error));

  return 0;
}
