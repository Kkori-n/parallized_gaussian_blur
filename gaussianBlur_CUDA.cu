#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


//Allows to check for CUDA errors, since they can be hard to debug otherwise.
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define MAX_KERNEL_SIZE 1025
__constant__ float d_kernel[MAX_KERNEL_SIZE];

__global__ void blur_horizontal(const unsigned char* img, float* temp, int w, int h, int channels, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= w || y >= h || c >= channels) return;

    float sum = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        int xx = max(0, min(w - 1, x + dx));
        sum += d_kernel[dx + r] * img[(size_t(y) * w + xx) * channels + c];
    }

    temp[(size_t(y) * w + x) * channels + c] = sum;
}

__global__ void blur_vertical(const float* temp, unsigned char* out, int w, int h, int channels, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= w || y >= h || c >= channels) return;

    float sum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        int yy = max(0, min(h - 1, y + dy));
        sum += d_kernel[dy + r] * temp[(size_t(yy) * w + x) * channels + c];
    }

    out[(size_t(y) * w + x) * channels + c] = (unsigned char)(sum + 0.5f);
}

int main(int argc, char** argv) {
    const char* input_path = "test_image.jpg";
    const char* output_path = "test_output.jpg";
    if(argc < 4) {
        printf("Usage: %s <input_image> <output_image> <sigma> <--devlog>", argv[0]);
        return 1;
    }
    input_path = argv[1];
    output_path = argv[2];
    float sigma = (argc > 3) ? atof(argv[3]) : 5.0;
    bool devlog = (argc > 4) && (strcmp(argv[4], "--devlog") == 0);

    int w, h, channels;
    unsigned char* h_img = stbi_load(input_path, &w, &h, &channels, 0);
    if (!h_img) {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        exit(EXIT_FAILURE);
    }

    int radius = (int)ceil(3 * sigma);
    if (radius > MAX_KERNEL_SIZE / 2 - 1) {
        fprintf(stderr, "Radius too large for kernel size\n");
        exit(EXIT_FAILURE);
    }

    int kernal_size = 2 * radius + 1;
    if (kernal_size > MAX_KERNEL_SIZE) {
        fprintf(stderr, "Kernel size exceeds maximum allowed\n");
        exit(EXIT_FAILURE);
    }

    std::vector<float> h_kernel(kernal_size);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        h_kernel[i + radius] = exp(-(i * i) / (2.0f * sigma * sigma));
        sum += h_kernel[i + radius];
    }

    for (float& v : h_kernel) v /= sum;
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, h_kernel.data(), kernal_size * sizeof(float)));

    if (devlog) printf("Loaded image: %s (Width: %d, Height: %d, Channels: %d)\n", input_path, w, h, channels);

    float* d_temp;
    unsigned char* d_img, *d_out, *d_out;
    size_t img_size = (size_t)w * h * channels;

    CUDA_CHECK(cudaMalloc(&d_img, img_size));
    CUDA_CHECK(cudaMalloc(&d_temp, img_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, img_size));

    CUDA_CHECK(cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, channels);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (devlog) printf("Launching CUDA kernels with block size (%d, %d) and grid size (%d, %d, %d)\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y, gridSize.z);

    cudaEventRecord(start);

    blur_horizontal<<<gridSize, blockSize>>>(d_img, d_temp, w, h, channels, radius);
    CUDA_CHECK(cudaGetLastError());
    blur_vertical<<<gridSize, blockSize>>>(d_temp, d_out, w, h, channels, radius);
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    if (devlog) printf("CUDA Blurring Time: %.2f ms\n", ms);

    std::vector<unsigned char> h_out(img_size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, img_size, cudaMemcpyDeviceToHost));

    stbi_write_jpg(output_path, w, h, channels, h_out.data(), 90);

    cudaFree(d_img);
    cudaFree(d_temp);
    cudaFree(d_out);
    stbi_image_free(h_img);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("blurred %dx%d (%d ch), sigma=%.2f, GPU: %s, %.3f s\n", w, h, channels, sigma, prop.name, ms / 1000.0f);

    return 0;
}