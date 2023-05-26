#include <thrust/async/transform.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

thrust::device_event foo(thrust::device_vector<float>& x,
  cudaStream_t stream);

__global__ void bar() {
  // do something here
  for (int i = 0; i < 100000; ++i);
}

int main()
{           
  int N = 5000000;
  thrust::device_vector<float> x(N);
  thrust::negate<float> op;

  // method 1
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    {
      thrust::device_event e; // empty device_event
                              // `thrust::async` APIs returns a handle,
                              // which synchronizes when it is destroyed,
                              // so we need a variable to hold the handle
      for (int i = 0; i < 10; ++i) {
        e = thrust::async::transform(thrust::cuda::par.on(stream),
          x.begin(), x.end(), x.begin(), op);
      }
    } // dtor of device_event will call cudaStreamSynchronize once
    cudaStreamDestroy(stream);
  }

  // method 2
  {
    thrust::device_event e(thrust::new_stream); // device_event with stream

    for (int i = 0; i < 10; ++i) {
      e = thrust::async::transform(thrust::device.after(e),
        x.begin(), x.end(), x.begin(), op);
    }
  } // dtor of device_event will call cudaStreamSynchronize,
    // but it will call 11 times


  // method 3
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
      thrust::device_event e;
      e = foo(x, stream);
      bar<<<1, 1, 0, stream>>>();
    } // dtor of device_event will call cudaStreamSynchronize once

    cudaStreamDestroy(stream);
  }
  return 0;
}

thrust::device_event foo(thrust::device_vector<float>& x,
  cudaStream_t stream)
{
  thrust::negate<float> op;
  thrust::device_event e;

  for (int i = 0; i < 10; ++i) {
    e = thrust::async::transform(thrust::cuda::par.on(stream),
      x.begin(), x.end(), x.begin(), op);
  }

  // have to return the handle to avoid destroying it,
  // if you don't want to synchronizes here
  return e;
}
