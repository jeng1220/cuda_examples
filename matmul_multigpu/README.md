# Matrix Multiplication with Multi-GPU #

# Requirement #
- CUDA 8.0+
- NumPy, Python example only
- CuPy, Python example only

## Build ##
```sh
make -j2
```

## Run ##
```
./sgemm
python ./c_binding.py --help
python ./c_binding.py
```

## Reference ##
- [Using the cuBLASXt API](https://docs.nvidia.com/cuda/cublas/#using-the-cublasxt-api)
