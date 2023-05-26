# CUDA Virtual Memory Management with MultiGPU #
Two examples demonstrate the interoperability between CUDA virtual memory management and multi-GPU computing

## Requirement ##
* CUDA 11
* NCCL
* MPI (NVIDIA HPC SDK is recommended)
* Two GPUs

## Build ##
```sh
$ USE_CUDA_VMM=0 make -j`nproc` # without CUDA virtual memory management
$ USE_CUDA_VMM=1 make -j`nproc` # with CUDA virtual memory management
```

## Run ##
```sh
$ ./single_process_multigpu
$ mpirun -n 2 ./multiprocess_multigpu
```