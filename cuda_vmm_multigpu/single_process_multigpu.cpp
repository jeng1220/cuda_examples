#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#if USE_CUDA_VMM
#include "multidevicealloc_memmap.hpp"
#endif

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

int main(int argc, char* argv[])
{
  ncclComm_t comms[2];

  //managing 2 devices
  int nDev = 2;
  int size = 32*1024*1024;
  int devs[2] = { 0, 1 };

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  // CUDA virtual memory management
#if USE_CUDA_VMM
  std::vector<CUdevice> mappingDevices;
  std::vector<CUdevice> backingDevices[2];

  for (int i = 0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    cudaFree(0); // force runtime to create a ctx
    CUdevice cuDevice;
    CHECK_DRV(cuDeviceGet(&cuDevice, i));
    mappingDevices.push_back(cuDevice);
    //backingDevices[i] = getBackingDevices(cuDevice);
    backingDevices[i].push_back(cuDevice);
  }

  size_t allocationSize = 0;
#endif

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));

#if USE_CUDA_VMM
    CHECK_DRV(simpleMallocMultiDeviceMmap(reinterpret_cast<CUdeviceptr*>(sendbuff + i), &allocationSize, size * sizeof(float), backingDevices[i], mappingDevices));
    CHECK_DRV(simpleMallocMultiDeviceMmap(reinterpret_cast<CUdeviceptr*>(recvbuff + i), nullptr, size * sizeof(float), backingDevices[i], mappingDevices));
#else
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
#endif
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
      comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
#if USE_CUDA_VMM
    CHECK_DRV(simpleFreeMultiDeviceMmap(reinterpret_cast<CUdeviceptr>(sendbuff[i]), allocationSize));
    CHECK_DRV(simpleFreeMultiDeviceMmap(reinterpret_cast<CUdeviceptr>(recvbuff[i]), allocationSize));
#else
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
#endif
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }

  printf("Success \n");
  return 0;
}
