#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Mandelbrot.hpp"
#include <ctime>    // for time()
#include <chrono>

#define CxMin -1.27
#define CxMax -1.24
#define CyMin 0.01
#define CyMax 0.03
#define EscapeRadius 2
#define IDim_max 700
#define MaxGrayComponentValue 255
#define IterationMax 1275
#define output_row_width 20

float PixelWidth;
float PixelHeight;
float ER2;

__global__ void mandelKernel(float CxMin, float PixelWidth, float CyMin, float PixelHeight, float ER2, int IterationMax, int *image_out) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < IDim_max && j < IDim_max) {
        float Zx, Zy;
        float Zx2, Zy2;
        float Cx, Cy;
        int Iteration;

        Cx = CxMin + j * PixelWidth;
        Cy = CyMin + i * PixelHeight;

        Zx = 0.0;
        Zy = 0.0;
        Zx2 = 0.0;
        Zy2 = 0.0;

        for (Iteration = 0; Iteration < IterationMax && ((Zx2 + Zy2) < ER2); Iteration++) {
            Zy = 2 * Zx * Zy + Cy;
            Zx = Zx2 - Zy2 + Cx;
            Zx2 = Zx * Zx;
            Zy2 = Zy * Zy;
        }

        if (Iteration == IterationMax)
            image_out[i * IDim_max + j] = 0;
        else
            image_out[i * IDim_max + j] = (int)Iteration / 5;
    }
}

__host__ void write_image_to_file(int *image_out) {
    int i, j;
    int OUTPUT_COUNTER = 1;
    printf("P2\n");
    printf("%d %d\n", IDim_max, IDim_max);
    printf("%d\n", MaxGrayComponentValue);

    for (i = 0; i < IDim_max; i++) {
        for (j = 0; j < IDim_max; j++) {
            if (OUTPUT_COUNTER == output_row_width) {
                printf("\n");
                OUTPUT_COUNTER = 1;
            }
            printf("%d ", image_out[i * IDim_max + j]);
            OUTPUT_COUNTER += 1;
        }
    }
}

int main() {
    PixelWidth = (CxMax - CxMin) / IDim_max;
    PixelHeight = (CyMax - CyMin) / IDim_max;
    ER2 = EscapeRadius * EscapeRadius;

    int *d_image_out;
    int *h_image_out = (int *)malloc(IDim_max * IDim_max * sizeof(int)); //allocate memory in GPU/Device during Kernel execution
    /*malloc is used the GPU kernel code and memory allocated is typically local or shared memory. The lifetime of the memory is limited to the execution of the kernel, and it must be freed within the kernel using free*/
    cudaMalloc((void **)&d_image_out, IDim_max * IDim_max * sizeof(int)); //allocate arrays in device memory
    /*function is used in the CPU/Host and memory is allocated in the GPU for use in CUDA kernels. Memory is typically allocated globally in the device and can be accessed by kernel running on the GPU*/
    /*Data must be transferred between host and device memory using functions like cudaMemcpy*/

    /*cudaMalloc allocates memory on the GPU from the CPU vs malloc allocates memory within kernel's execution*/
    /*Use cudaMalloc for memory that persists across multiple kernel launches. Use malloc for temporary allocations that only exist during the kernel's execution*/
    dim3 threadsPerBlock(16, 16); //grid dimentions
    dim3 numBlocks((IDim_max + threadsPerBlock.x - 1) / threadsPerBlock.x, (IDim_max + threadsPerBlock.y - 1) / threadsPerBlock.y); //number of blocks

    mandelKernel<<<numBlocks, threadsPerBlock>>>(CxMin, PixelWidth, CyMin, PixelHeight, ER2, IterationMax, d_image_out);

    cudaMemcpy(h_image_out, d_image_out, IDim_max * IDim_max * sizeof(int), cudaMemcpyDeviceToHost); //copy memory from Host/CPU to Device/GPU

    write_image_to_file(h_image_out);

    cudaFree(d_image_out); //frees memory on the GPU
    free(h_image_out); //freeing memory after kernel execution

    return 0;
}