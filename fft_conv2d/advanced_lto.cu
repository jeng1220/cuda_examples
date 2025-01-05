#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cufftXt.h>
//#include <nvtx3/nvtx3.hpp> // https://github.com/NVIDIA/NVTX
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "nvrtc_helper.h"

void check(cufftResult result, const char* filename, int line) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "ERROR, " << filename << ":"
            << line << ", " << (result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void check(cudaError_t error, const char* filename, int line) {
    if (error != cudaSuccess) {
        std::cerr << "ERROR, " << filename << ":"
            << line << ", " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, __FILE__, __LINE__)

// Helper function to print a matrix
template<typename T>
void print_matrix(const T& matrix, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << "\n";
    }
}

template <>
void print_matrix(const thrust::device_vector<cufftComplex>& matrix, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cufftComplex c = matrix[i * width + j];
            std::cout << c.x << "," << c.y << "j ";
        }
        std::cout << "\n";
    }
}

thrust::host_vector<float> valid_cross_correlation_2d(
    const thrust::host_vector<float>& image, int image_height, int image_width,
    const thrust::host_vector<float>& filter, int filter_height, int filter_width)
{
    int output_height = image_height - filter_height + 1;
    int output_width = image_width - filter_width + 1;
    thrust::host_vector<float> output(output_height * output_width, 0.f);

    for (int oy = 0; oy < output_height; ++oy) {
        for (int ox = 0; ox < output_width; ++ox) {
            float tmp = 0.f;
            for (int fy = 0; fy < filter_height; ++fy) {
                for (int fx = 0; fx < filter_width; ++fx) {
                    auto iv = image[(oy + fy) * image_width + (ox + fx)];
                    auto fv = filter[fy * filter_width + fx];
                    tmp += iv * fv;
                }
            }
            output[oy * output_width + ox] = tmp;
        }
    }
    return output;
}


template<typename T>
thrust::device_ptr<T> async_allocate(size_t n, cudaStream_t stream) {
    T* raw_ptr = nullptr;
    CHECK(cudaMallocAsync(&raw_ptr, n * sizeof(T), stream));
    thrust::device_ptr<T> output_ptr(raw_ptr);
    return output_ptr;
}

template<typename T>
void async_free(thrust::device_ptr<T>& ptr, cudaStream_t stream) {
    auto* raw_ptr = thrust::raw_pointer_cast(ptr);
    CHECK(cudaFreeAsync(raw_ptr, stream));
}


void irfft_epilog(thrust::device_ptr<float>& input, size_t size, cudaStream_t stream) {
    auto divisor = static_cast<float>(size);
    thrust::for_each(thrust::cuda::par_nosync.on(stream), input, input + size,
        [divisor] __device__ (float& v) { v /= divisor; });
}


struct FlipPadParams {
    int image_width;
    int filter_height;
    int filter_width;
};

struct ComplexMultiplyParams {
    cufftComplex* ptr;
};

struct EpilogExtractParams {
    int image_width;
    int offset_y;
    int offset_x;
    int output_width;
    float size_scale;
};

class FFTCrossCorrelation {
public:
    FFTCrossCorrelation(int image_height, int image_width, int filter_height, int filter_width) :
        image_height_(image_height), image_width_(image_width),
        filter_height_(filter_height), filter_width_(filter_width) {

        //nvtx3::scoped_range tag("fft-conv2d::init");

        std::vector<char> callback_buffer;
	    compile_file_to_lto(callback_buffer, "./fft_callback.cu");

        CHECK(cufftCreate(&filter_plan_));
        CHECK(cudaMalloc((void **)&d_flip_params_, sizeof(FlipPadParams)));
        FlipPadParams h_flip_params{image_width, filter_height, filter_width};
        CHECK(cudaMemcpy(d_flip_params_, &h_flip_params,
            sizeof(FlipPadParams), cudaMemcpyHostToDevice));
        CHECK(cufftXtSetJITCallback(
            filter_plan_, 
            "flip_pad_callback",
            (void*)callback_buffer.data(),
            callback_buffer.size(),
            CUFFT_CB_LD_REAL,
            (void **)&d_flip_params_));

        CHECK(cudaMallocHost((void **)&h_cmul_params_, sizeof(ComplexMultiplyParams)));
        CHECK(cudaMalloc((void **)&d_cmul_params_, sizeof(ComplexMultiplyParams)));
        CHECK(cufftXtSetJITCallback(
            filter_plan_, 
            "complex_multiply_callback",
            (void*)callback_buffer.data(),
            callback_buffer.size(),
            CUFFT_CB_ST_COMPLEX,
            (void **)&d_cmul_params_));
        size_t size = 0;
        CHECK(cufftMakePlan2d(filter_plan_, image_height, image_width, CUFFT_R2C, &size));
 
        CHECK(cufftCreate(&plan_inverse_));
        CHECK(cudaMalloc((void **)&d_extract_params_, sizeof(EpilogExtractParams)));
        int output_width = image_width - filter_width + 1;
        float size_scale = static_cast<float>(image_height) * image_width;
        EpilogExtractParams h_extract_params{
            image_width, filter_height - 1, filter_width - 1, output_width, size_scale
        };
        CHECK(cudaMemcpy(d_extract_params_, &h_extract_params,
            sizeof(EpilogExtractParams), cudaMemcpyHostToDevice));
        CHECK(cufftXtSetJITCallback(
            plan_inverse_, 
            "epilog_extract_callback",
            (void*)callback_buffer.data(),
            callback_buffer.size(),
            CUFFT_CB_ST_REAL,
            (void **)&d_extract_params_));
        CHECK(cufftMakePlan2d(plan_inverse_, image_height, image_width, CUFFT_C2R, &size));
 
        CHECK(cufftPlan2d(&image_plan_, image_height, image_width, CUFFT_R2C));
    }
    ~FFTCrossCorrelation() {
        CHECK(cufftDestroy(image_plan_));
        CHECK(cufftDestroy(filter_plan_));
        CHECK(cufftDestroy(plan_inverse_));
        CHECK(cudaFree(d_flip_params_));
        CHECK(cudaFree(d_cmul_params_));
        CHECK(cudaFreeHost(h_cmul_params_));
        CHECK(cudaFree(d_extract_params_));
    }
    thrust::device_ptr<float> operator()(
        thrust::device_vector<float>& image,
        thrust::device_vector<float>& filter,
        cudaStream_t stream)
    {
        //nvtx3::scoped_range tag("fft-conv2d::exec");

        // Perform RFFT on image and filter
        int complex_width = image_width_ / 2 + 1; // Number of complex columns after RFFT
        size_t complex_size = static_cast<size_t>(image_height_) * complex_width;
        auto rfft_image = async_allocate<cufftComplex>(complex_size, stream);
        CHECK(cufftSetStream(image_plan_, stream));
        CHECK(cufftExecR2C(image_plan_, const_cast<float*>(thrust::raw_pointer_cast(image.data())),
            thrust::raw_pointer_cast(rfft_image)));

        // Perform element-wise multiplication in Fourier domain
        h_cmul_params_->ptr = thrust::raw_pointer_cast(rfft_image);
        CHECK(cudaMemcpyAsync(d_cmul_params_, h_cmul_params_, sizeof(ComplexMultiplyParams),
            cudaMemcpyHostToDevice, stream));
        auto rfft_result = async_allocate<cufftComplex>(complex_size, stream);
        CHECK(cufftSetStream(filter_plan_, stream));
        CHECK(cufftExecR2C(filter_plan_, const_cast<float*>(thrust::raw_pointer_cast(filter.data())),
            thrust::raw_pointer_cast(rfft_result)));
        async_free(rfft_image, stream);


        // Perform inverse RFFT to get the spatial domain result
        // Extract the valid region
        int output_height = image_height_ - filter_height_ + 1;
        int output_width = image_width_ - filter_width_ + 1;
        size_t output_size = static_cast<size_t>(output_height) * output_width;
        auto valid_result = async_allocate<float>(output_size, stream);

        CHECK(cufftSetStream(plan_inverse_, stream));
        CHECK(cufftExecC2R(plan_inverse_, thrust::raw_pointer_cast(rfft_result), thrust::raw_pointer_cast(valid_result)));
        async_free(rfft_result, stream);
        return valid_result;
    }
private:
    int image_height_;
    int image_width_;
    int filter_height_;
    int filter_width_;
    cufftHandle image_plan_;
    cufftHandle filter_plan_;
    cufftHandle plan_inverse_;
    FlipPadParams* d_flip_params_;
    ComplexMultiplyParams* d_cmul_params_;
    ComplexMultiplyParams* h_cmul_params_;
    EpilogExtractParams* d_extract_params_;
};

int fft_cross_correlation_test() {
    int image_height = 7;
    int image_width = 7;
    int filter_height = 5;
    int filter_width = 5;

    thrust::host_vector<float> image = {
        0.5488135,  0.71518937, 0.60276338, 0.54488318, 0.4236548,  0.64589411, 0.43758721,
        0.891773,   0.96366276, 0.38344152, 0.79172504, 0.52889492, 0.56804456, 0.92559664,
        0.07103606, 0.0871293,  0.0202184,  0.83261985, 0.77815675, 0.87001215, 0.97861834,
        0.79915856, 0.46147936, 0.78052918, 0.11827443, 0.63992102, 0.14335329, 0.94466892,
        0.52184832, 0.41466194, 0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898,
        0.6176355,  0.61209572, 0.616934,   0.94374808, 0.6818203,  0.3595079, 0.43703195,
        0.6976312,  0.06022547, 0.66676672, 0.67063787, 0.21038256, 0.1289263, 0.31542835,
    };
    thrust::host_vector<float> filter = {
        0.36371077, 0.57019677, 0.43860151, 0.98837384, 0.10204481,
        0.20887676, 0.16130952, 0.65310833, 0.2532916,  0.46631077,
        0.24442559, 0.15896958, 0.11037514, 0.65632959, 0.13818295,
        0.19658236, 0.36872517, 0.82099323, 0.09710128, 0.83794491,
        0.09609841, 0.97645947, 0.4686512,  0.97676109, 0.60484552,
    };

    auto output = valid_cross_correlation_2d(
        image, image_height, image_width,
        filter, filter_height, filter_width);

    int output_height = image_height - filter_height + 1;
    int output_width = image_width - filter_width + 1;
    std::cout << "\nDirect Cross Correlation:\n";
    print_matrix(output, output_height, output_width);

    thrust::device_vector<float> d_image = image;
    thrust::device_vector<float> d_filter = filter;

    FFTCrossCorrelation op(image_height, image_width, filter_height, filter_width);
    auto d_output = op(d_image, d_filter, 0);

    std::cout << "\nFFT Cross Correlation:\n";
    print_matrix(d_output, output_height, output_width);
    async_free(d_output, 0);
    CHECK(cudaStreamSynchronize(0));
    return 0;
}

int benchmark() {
    int image_height = 4096;
    int image_width = 4096;
    int filter_height = 41;
    int filter_width = 41;

    thrust::device_vector<float> image(image_height * image_width);
    thrust::device_vector<float> filter(filter_height * filter_width);
    FFTCrossCorrelation op(image_height, image_width, filter_height, filter_width);

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (int i = 0; i < 10; ++i) {
        auto output = op(image, filter, stream);
        async_free(output, stream);
    }
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaStreamDestroy(stream));
    return 0;
}

int main() {
    fft_cross_correlation_test();
    benchmark();
    return 0;
}
