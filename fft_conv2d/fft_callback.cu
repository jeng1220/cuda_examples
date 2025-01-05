//#include <cufftXt.h>
typedef float2 cufftComplex;

struct FlipPadParams {
    int image_width;
    int filter_height;
    int filter_width;
};

__device__ float flip_pad_callback(
    void* input,
    unsigned long long idx,
    void* info,
    void* sharedmem) 
{  
	auto* params = static_cast<const FlipPadParams*>(info);
    int image_width = params->image_width;
    int filter_height = params->filter_height;
    int filter_width = params->filter_width;
	auto* filter_ptr = static_cast<float*>(input);

    int x = idx % image_width;
    int y = idx / image_width;
    float v = 0.f;
    if (x < filter_width && y < filter_height) {
        int fx = filter_width - x - 1;
        int fy = filter_height - y - 1;
        v = filter_ptr[fy * filter_width + fx];
    }
    return v;
}


struct ComplexMultiplyParams {
    cufftComplex* ptr;
};

__device__ void complex_multiply_callback(
    void *output, 
    unsigned long long idx,
    cufftComplex z2, 
    void *info, 
    void *sharedmem) 
{
    auto* params = static_cast<const ComplexMultiplyParams*>(info);
    auto z1 = params->ptr[idx];
    cufftComplex v{(z1.x * z2.x - z1.y * z2.y), (z1.x * z2.y + z1.y * z2.x)};
    auto* output_ptr = static_cast<cufftComplex*>(output);
    output_ptr[idx] = v;
}


struct EpilogExtractParams {
    int image_width;
    int offset_y;
    int offset_x;
    int output_width;
    float size_scale;
};

__device__ void epilog_extract_callback(
    void *output, 
    unsigned long long idx,
    float element, 
    void *info, 
    void *sharedmem) 
{
    auto* params = static_cast<const EpilogExtractParams*>(info);
    int image_width = params->image_width;
    int offset_y = params->offset_y;
    int offset_x = params->offset_x;
    int output_width = params->output_width;

    int x = idx % image_width;
    int y = idx / image_width;
    if (x >= offset_x && y >= offset_y) {
        int offset = (y - offset_y) * output_width + (x - offset_x);
	auto* output_ptr = static_cast<float*>(output);
        output_ptr[offset] = element / params->size_scale;
    }
}
