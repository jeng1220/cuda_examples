# Thrust Async #

This example demonstrates how to achieve asynchronous computing of new Thrust library. After CUDA 10.1, Thrust behavior was changed a lot. The behavior of old thrust APIs become blocking API, such as `thrust::transform`, `thrust::sort`. That means these APIs will call cudaStreamSynchronize automatically. If developer wants to achieve asynchronous computing, have to replace old APIs with `thrust::async` series APIs, such as `thrust::async::transform`. Read the example to know more detail.

# requirement #

* CUDA 10.1+

# build #

```sh
$ make
```

# run #

```sh
$ nsys profile -c cudaProfilerApi -t cuda,nvtx ./foo # use nsys to see the trace
```
