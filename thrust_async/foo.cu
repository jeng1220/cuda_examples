#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <thrust/async/transform.h>
#include <thrust/device_vector.h>

class NVTX {
  public:
    NVTX(const char str[]) { nvtxRangePush(str); }
    ~NVTX() { nvtxRangePop(); }

  private:
};

thrust::device_event foo(thrust::device_vector<float> &x, cudaStream_t stream);

__global__ void bar() {
    // do something here
    for (int i = 0; i < 100000; ++i)
        ;
}

int main() {
    int N = 5000000;
    thrust::device_vector<float> x(N);
    thrust::negate<float> op;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaProfilerStart();

    // method 1
    {
        NVTX tag("method-1");
        thrust::device_event e; // empty device_event
                                // `thrust::async` APIs returns a handle,
                                // which synchronizes when it is destroyed,
                                // so we need a variable to hold the handle
        for (int i = 0; i < 10; ++i) {
            e = thrust::async::transform(thrust::cuda::par.on(stream), x.begin(), x.end(),
                                         x.begin(), op);
        }
    }
    // dtor of device_event will call cudaStreamSynchronize once

#if __CUDACC_VER_MAJOR__ > 11

    // method 2
    {
        NVTX tag("method-2");
        thrust::device_event e; // empty device_event
                                // `thrust::async` APIs returns a handle,
                                // which synchronizes when it is destroyed,
                                // so we need a variable to hold the handle
        for (int i = 0; i < 10; ++i) {
            e = thrust::async::transform(thrust::cuda::par_nosync.on(stream), x.begin(), x.end(),
                                         x.begin(), op);
        }
    }
    // dtor of device_event will call cudaStreamSynchronize once

    // method 3
    {
        NVTX tag("method-3");
        for (int i = 0; i < 10; ++i) {
            auto e = thrust::async::transform(thrust::cuda::par_nosync.on(stream), x.begin(),
                                              x.end(), x.begin(), op);
            // dtor of device_event will call cudaStreamSynchronize,
            // and it will call 11 times
        }
    }
#endif // __CUDACC_VER_MAJOR__

    // method 4
    {
        NVTX tag("method-4");
        thrust::device_event e(thrust::new_stream); // device_event with stream

        for (int i = 0; i < 10; ++i) {
            e = thrust::async::transform(thrust::device.after(e), x.begin(), x.end(), x.begin(),
                                         op);
        }
        // dtor of device_event will call cudaStreamSynchronize,
        // but it will call 11 times in CUDA 10.1
	// it was fixed after CUDA 11.x
    }

    // method 5
    {
        NVTX tag("method-5");
        {
            thrust::device_event e;
            e = foo(x, stream);
            bar<<<1, 1, 0, stream>>>();
        }
        // dtor of device_event will call cudaStreamSynchronize once
    }

    cudaProfilerStop();
    cudaStreamDestroy(stream);
    return 0;
}

thrust::device_event foo(thrust::device_vector<float> &x, cudaStream_t stream) {
    thrust::negate<float> op;
    thrust::device_event e;

    for (int i = 0; i < 10; ++i) {
        e = thrust::async::transform(thrust::cuda::par.on(stream), x.begin(), x.end(), x.begin(),
                                     op);
    }

    // have to return the handle to avoid destroying it,
    // if you don't want to synchronizes here
    return e;
}
