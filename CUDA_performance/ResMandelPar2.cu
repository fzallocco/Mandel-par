#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Mandelbrot.hpp"
#include <ctime>    // for time()
#include <chrono>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

__device__ int mandelbrot(float real, float imag, int max_iter) {
    float z_real = real;
    float z_imag = imag;
    int n;
    for (n = 0; n < max_iter; ++n) {
        float r2 = z_real * z_real;
        float i2 = z_imag * z_imag;
        if (r2 + i2 > 4.0f) break;
        z_imag = 2.0f * z_real * z_imag + imag;
        z_real = r2 - i2 + real;
    }
    return n;
}

__global__ void mandelbrot_kernel(int *output, int width, int height, float real_min, float real_max, float imag_min, float imag_max, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float real = real_min + (real_max - real_min) * x / width;
        float imag = imag_min + (imag_max - imag_min) * y / height;
        int index = y * width + x;
        output[index] = mandelbrot(real, imag, max_iter);
    }
}

void write_pgm(const char *filename, int *data, int width, int height) {
    ofstream file(filename, ios::out | ios::binary);
    file << "P5\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file.put(static_cast<unsigned char>(data[i] * 255 / 1000));
    }
    file.close();
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const float real_min = -2.0f;
    const float real_max = 1.0f;
    const float imag_min = -1.5f;
    const float imag_max = 1.5f;
    const int max_iter = 1000;

    int *h_output = new int[width * height];
    int *d_output;
    cudaMalloc(&d_output, width * height * sizeof(int));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    auto start = chrono::high_resolution_clock::now();
    mandelbrot_kernel<<<gridDim, blockDim>>>(d_output, width, height, real_min, real_max, imag_min, imag_max, max_iter);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    cudaMemcpy(h_output, d_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    write_pgm("mandelbrot.pgm", h_output, width, height);

    cudaFree(d_output);
    delete[] h_output;

    chrono::duration<double> duration = end - start;
    cout << "Time taken: " << duration.count() << " seconds" << endl;

    return 0;
}