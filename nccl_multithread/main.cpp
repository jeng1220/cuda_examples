#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nvToolsExt.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


struct ThreadData {
  int num_threads;
  int thread_id;
  ncclUniqueId nccl_id;
};

std::mutex mtx;

void foo(void* ptr) {
  auto* thread_ptr = reinterpret_cast<ThreadData*>(ptr);
  int nRanks = thread_ptr->num_threads;
  int myRank = thread_ptr->thread_id;
  int size = 1024;

  ncclUniqueId id = thread_ptr->nccl_id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  std::vector<float> hostbuff(size);
  for (auto& value : hostbuff) {
    value = myRank;
  }

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(myRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMemcpy(sendbuff, hostbuff.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));


  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  nvtxRangePush("start allreduce test");

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff,
    size, ncclFloat, ncclSum, comm, s));
  //check results
  CUDACHECK(cudaMemcpyAsync(hostbuff.data(), recvbuff, size * sizeof(float),
    cudaMemcpyDeviceToHost, s));
  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  nvtxRangePop();

  //check results
  mtx.lock();
  printf("myRank %d: ", myRank);
  for (int i = 0; i < 10; ++i) {
    printf("%.0f, ", hostbuff[i]);
  }
  printf("\n");
  mtx.unlock();

  nvtxRangePush("enqueue 2 allreduce");

  //enqueue `AllReduce` multiple times to see if it has deadlock
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff,
    size, ncclFloat, ncclSum, comm, s));
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff,
    size, ncclFloat, ncclSum, comm, s));
  CUDACHECK(cudaStreamSynchronize(s));

  nvtxRangePop();

  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  CUDACHECK(cudaStreamDestroy(s));


  //finalizing NCCL
  NCCLCHECK(ncclCommDestroy(comm));
}

int main(int argc, char* argv[])
{
  constexpr int num_dev = 8;
  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));

  std::vector<ThreadData> thread_data(num_dev);
  std::vector<std::thread> threads;

  CUDACHECK(cudaProfilerStart());
  for (int i = 0; i < num_dev; ++i) {
    auto& data = thread_data[i];
    data.num_threads = num_dev;
    data.thread_id = i;
    data.nccl_id = id;
    auto* ptr = &data;
    threads.push_back(std::thread(foo, ptr));
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  CUDACHECK(cudaProfilerStop());

  printf("Success \n");
  return 0;
}

