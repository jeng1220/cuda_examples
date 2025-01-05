# Simple FFT Cross Correlation (Conv2D) Example with cuFFT

Conv2D (2D convolution) and Cross-Correlation are very similar in CNNs. Both involve sliding a filter over an image and computing the element-wise product and sum. The main difference is that convolution flips the filter, while cross-correlation does not. In practice, Conv2D usually refers to cross-correlation.

Conv2D can be efficiently implemented using FFT (Fast Fourier Transform) if filter size is large, because it converts convolution operations into element-wise multiplications in the frequency domain, significantly speeding up the process for large filters and images.

This example only provides a basic demonstration of cuFFT usage. Users must optimize the process by themselves, such as data copying.

## Build
```sh
make
```

## Run
```sh
python ./foo.py # numpy reference
./basic         # using cuFFT
./advanced      # using cuFFT
```

