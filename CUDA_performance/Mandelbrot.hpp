
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>

using namespace cv;
using namespace std;

__device__ __constant__ int d_rows;
__device__ __constant__ int d_columns;

__global__ void mandelKernel(float CxMin, float PixelWidth, float CyMin, float PixelHeight, float ER2, int IterationMax, int *image_out);
__host__ void write_image_to_file(int *image_out);

/*__host__ uchar *cpuConvertToGray(std::string inputImage); replace
__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[]); we need this also from memory analysis
__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile); we need this
__host__ void copyFromDeviceToHost(uchar *d_gray, uchar *gray, int rows, int columns); we need this
__host__ std::tuple<uchar *, uchar *, uchar *, uchar *> allocateDeviceMemory(int rows, int columns); we need this also from memory analysis
__host__ void deallocateMemory(uchar *d_r, uchar *d_g, uchar *d_b, int *d_image_num_pixels); we need this
__host__ void cleanUpDevice(); we need this also from memory analysis
__host__ void copyFromHostToDevice(uchar *h_r, uchar *d_r, uchar *h_g, uchar *d_g, uchar *h_b, uchar *d_b, int rows, int cols); we need this also from memory analysis
__host__ void executeKernel(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray, int rows, int columns, int threadsPerBlock); we need this also from memory analysis
__global__ void convert(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray); replace
__host__ float compareGrayImages(uchar *gray, uchar *test_gray, int rows, int columns); replace*/
