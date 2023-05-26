#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>
#if USE_CUDA_VMM
#include "multidevicealloc_memmap.hpp"
#endif

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#if USE_CUDA_VMM
void checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
  if (res != CUDA_SUCCESS) {
    const char *errStr = NULL;
    (void)cuGetErrorString(res, &errStr);
    printf("Failed, CUDA Drv error %s:%d '%s'\n",
      file, line, errStr);
  }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

// collect all of the devices whose memory can be mapped from cuDevice.
std::vector<CUdevice> getBackingDevices(CUdevice cuDevice) {
  int num_devices;

  CHECK_DRV(cuDeviceGetCount(&num_devices));

  std::vector<CUdevice> backingDevices;
  backingDevices.push_back(cuDevice);
  for (int dev = 0; dev < num_devices; dev++) {
    int capable = 0;
    int attributeVal = 0;

    // The mapping device is already in the backingDevices vector
    if (dev == cuDevice) {
      continue;
    }

    // Only peer capable devices can map each others memory
    CHECK_DRV(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
    if (!capable) {
      continue;
    }

    // The device needs to support virtual address management for the required
    // apis to work
    CHECK_DRV(cuDeviceGetAttribute(
        &attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        cuDevice));
    if (attributeVal == 0) {
      continue;
    }

    backingDevices.push_back(dev);
  }
  return backingDevices;
}
#endif

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

int main(int argc, char* argv[])
{
  int size = 32*1024*1024;
  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  auto pid = getpid();
  printf("MPI Rank %d, MPI Size %d, PID %u, Host %s\n", myRank, nRanks, pid, hostname);

  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // CUDA virtual memory management
#if USE_CUDA_VMM
  std::vector<CUdevice> mappingDevices;
  std::vector<CUdevice> backingDevices;

  CUDACHECK(cudaSetDevice(localRank));
  cudaFree(0); // force runtime to create a ctx
  CUdevice cuDevice;
  CHECK_DRV(cuDeviceGet(&cuDevice, localRank));
  mappingDevices.push_back(cuDevice);
  //backingDevices = getBackingDevices(cuDevice);
  backingDevices.push_back(cuDevice);

  size_t allocationSize = 0;
#endif

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
#if USE_CUDA_VMM
  CHECK_DRV(simpleMallocMultiDeviceMmap(reinterpret_cast<CUdeviceptr*>(&sendbuff), &allocationSize, size * sizeof(float), backingDevices, mappingDevices));
  CHECK_DRV(simpleMallocMultiDeviceMmap(reinterpret_cast<CUdeviceptr*>(&recvbuff), nullptr, size * sizeof(float), backingDevices, mappingDevices));
#else
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
#endif  
  CUDACHECK(cudaStreamCreate(&s));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  //free device buffers
#if USE_CUDA_VMM
  CHECK_DRV(simpleFreeMultiDeviceMmap(reinterpret_cast<CUdeviceptr>(sendbuff), allocationSize));
  CHECK_DRV(simpleFreeMultiDeviceMmap(reinterpret_cast<CUdeviceptr>(recvbuff), allocationSize));
#else
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
#endif

  //finalizing NCCL
  NCCLCHECK(ncclCommDestroy(comm));

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
