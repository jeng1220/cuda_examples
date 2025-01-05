#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

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
#define PADD_DEV(x, y) ((x + y - 1) / y)

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

thrust::device_vector<cufftComplex> rfft2(const thrust::device_vector<float>& input, int height, int width, cudaStream_t stream) {
    int complex_width = width / 2 + 1; // Number of complex columns after RFFT
    // Allocate device memory
    thrust::device_vector<cufftComplex> output(height * complex_width);
    // Create a cuFFT plan
    cufftHandle plan_forward;
    CHECK(cufftPlan2d(&plan_forward, height, width, CUFFT_R2C));
    CHECK(cufftSetStream(plan_forward, stream));
    // Execute forward FFT
    CHECK(cufftExecR2C(plan_forward, const_cast<float*>(thrust::raw_pointer_cast(input.data())), thrust::raw_pointer_cast(output.data())));
    // Cleanup
    CHECK(cufftDestroy(plan_forward));
    return output;
}


void irfft_epilog(thrust::device_vector<float>& input, cudaStream_t stream) {
    auto divisor = static_cast<float>(input.size());
    thrust::for_each(thrust::cuda::par_nosync.on(stream), input.begin(), input.end(),
        [divisor] __device__ (float& v) { v /= divisor; });
}

thrust::device_vector<float> irfft2(const thrust::device_vector<cufftComplex>& input, int height, int width, cudaStream_t stream) {
    // Allocate device memory
    thrust::device_vector<float> output(height * width);
    // Create a cuFFT plan for inverse transform
    cufftHandle plan_inverse;
    CHECK(cufftPlan2d(&plan_inverse, height, width, CUFFT_C2R));
    CHECK(cufftSetStream(plan_inverse, stream));
    // Execute inverse FFT
    CHECK(cufftExecC2R(plan_inverse, const_cast<cufftComplex*>(thrust::raw_pointer_cast(input.data())), thrust::raw_pointer_cast(output.data())));
    irfft_epilog(output, stream);
    // Cleanup
    CHECK(cufftDestroy(plan_inverse));
    return output;
}

// Perform 2D RFFT and IRFFT
thrust::device_vector<float> rfft2_irfft2(const thrust::device_vector<float>& input, int height, int width) {
    auto c = rfft2(input, height, width, 0);
    auto r = irfft2(c, height, width, 0);
    return r;
}

int rfft2_irfft2_test() {
    // Input data (e.g., a 4x4 matrix)
    thrust::device_vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Perform RFFT2 and IRFFT2
    int height = 4;
    int width = 4;
    auto output = rfft2_irfft2(input, height, width);

    // Print the original input and output
    std::cout << "Input Matrix:\n";
    print_matrix(input, height, width);
    std::cout << "\nOutput Matrix (after IRFFT2):\n";
    print_matrix(output, height, width);
    return 0;
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

thrust::device_vector<float> fft_valid_cross_correlation_rfft_2d(    
    const thrust::device_vector<float>& image, int image_height, int image_width,
    const thrust::device_vector<float>& filter, int filter_height, int filter_width,
    cudaStream_t stream)
{
    // Flip and Pad the filter to match the image size
    thrust::device_vector<float> flipped_padded_filter(image_height * image_width);
    auto* filter_ptr = thrust::raw_pointer_cast(filter.data());
    auto* padded_filter_ptr = thrust::raw_pointer_cast(flipped_padded_filter.data());
    thrust::counting_iterator<int> itr(0);
    thrust::for_each(thrust::cuda::par_nosync.on(stream), itr, itr + flipped_padded_filter.size(),
        [filter_ptr, filter_height, filter_width, padded_filter_ptr, image_width] __device__ (int i) {
        int x = i % image_width;
        int y = i / image_width;
        float v = 0.f;
        if (x < filter_width && y < filter_height) {
            int fx = filter_width - x - 1;
            int fy = filter_height - y - 1;
            v = filter_ptr[fy * filter_width + fx];
        }
        padded_filter_ptr[i] = v;
    });

    // Perform RFFT on image and filter
    auto rfft_image = rfft2(image, image_height, image_width, stream);
    auto rfft_filter = rfft2(flipped_padded_filter, image_height, image_width, stream);

    // Perform element-wise multiplication in Fourier domain
    thrust::device_vector<cufftComplex> rfft_result(rfft_image.size());
    thrust::transform(thrust::cuda::par_nosync.on(stream), rfft_image.begin(), rfft_image.end(), rfft_filter.begin(), rfft_result.begin(),
        [] __device__ (cufftComplex z1, cufftComplex z2) {
        return cufftComplex{(z1.x * z2.x - z1.y * z2.y), (z1.x * z2.y + z1.y * z2.x)};
    });

    // Perform inverse RFFT to get the spatial domain result
    auto full_result = irfft2(rfft_result, image_height, image_width, stream);

    // Extract the valid region
    auto* offset_ptr = thrust::raw_pointer_cast(full_result.data()) + (filter_height - 1) * image_width + (filter_width - 1);
    int output_height = image_height - filter_height + 1;
    int output_width = image_width - filter_width + 1;
    thrust::device_vector<float> valid_result(output_height * output_width);
    auto* output_ptr = thrust::raw_pointer_cast(valid_result.data());
    thrust::for_each(thrust::cuda::par_nosync.on(stream), itr, itr + valid_result.size(),
        [offset_ptr, image_width, output_ptr, output_width] __device__ (int i) {
        int x = i % output_width;
        int y = i / output_width;
        output_ptr[i] = offset_ptr[y * image_width + x];
    });
    return valid_result;
}

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
    auto d_output = fft_valid_cross_correlation_rfft_2d(
        d_image, image_height, image_width,
        d_filter, filter_height, filter_width, 0);
    std::cout << "\nFFT Cross Correlation:\n";
    print_matrix(d_output, output_height, output_width);
    return 0;
}

int main() {
    rfft2_irfft2_test();
    fft_cross_correlation_test();
    return 0;
}
