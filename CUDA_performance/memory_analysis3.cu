//NAT version
#include "memory_analysis.h"

__global__ void globalMemorySearch(int *input, int elementsPerThread, int threadsPerBlock)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < threadsPerBlock)
    {
      int totalFound = 0;
      for(int i = 0; i < numElements; i++) {
        if (input[i] == constant_search_value)
          totalFound++;
      }

        // Create a for loop that handles the fact that each thread needs to search for n values in input
            // Increment the input array at index i by 1, update in memory prior to testing if it is equal to constant_search_value
    }
}

__global__ void sharedMemorySearch(int *input, int elementsPerThread, int threadsPerBlock)
{
    extern __shared__ int sharedInput[];
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i = 0; i < numElements; i++)
        sharedInput[i] = input[i];

    if (threadId < threadsPerBlock)
    {
        int totalFound = 0;
        for(int i = 0; i < numElements; i++) {
        if (sharedInput[i] == constant_search_value)
          totalFound++;
        }
        __syncthreads();
        // Create a for loop that handles the fact that each thread needs to search for n values in input
            // Increment the input array at index i by 1, update in memory prior to testing if it is equal to constant_search_value
    }
}

__global__ void constantMemorySearch()
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < constant_num_threads)
    {
        // Create a for loop that handles the fact that each thread needs to search for n values in input
            // Increment the input array at index i by 1, update in memory prior to testing if it is equal to constant_search_value
    }
}

__global__ void registerMemorySearch(int numElements, int numThreads)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int register_array[numElements];
    if (threadId < threadsPerBlock) {
        for (int i = 0; i < numElements)
            register_array[i] += threadId + i;

        int totalFound = 0;
        for(int i = 0; i < numElements; i++) {
        if (register_array[i] == constant_search_value)
          totalFound++;
        }
        // Create a for loop that handles the fact that each thread needs to search for n values in input
        // Increment the input array at index i by 1, update in memory prior to testing if it is equal to constant_search_value
    }
}

// This will generate an array of size numElements of random integers from 0 to 255 in pageable host memory
__host__ int * allocatePageableRandomHostMemory(int numElements)
{
    srand(time(0));
    size_t size = numElements * sizeof(int);

    // Allocate the host pinned memory input pointer B
    int *data;
    cudaHostAlloc((void**)&data, size, cudaHostAllocDefault);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        data[i] = rand() % 255;
    }

    return data;
}

__host__ int * allocateDeviceMemory(int numElements)
{
    size_t size = numElements * sizeof(int);

    int *d_input = NULL;
    cudaError_t err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return d_input;
}

__host__ void copyFromHostToDevice(std::string kernelType, int *input, int numElements, int numThreads, int *d_input)
{
    size_t size = numElements * sizeof(int);

    if(!strcmp(kernelType.c_str(), "constant"))
    {
      //__device__ __constant__ int *constant_input;
      err = cudaMemcpyToSymbol(constant_input, &input, sizeof(int)*numElements, 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      //__device__ __constant__ int constant_num_elements;
      err = cudaMemcpyToSymbol(constant_num_elements,&numElements, sizeof(int), 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
      //__device__ __constant__ int constant_num_threads;
      err = cudaMemcpyToSymbol(constant_num_threads,&threadsPerBlock, sizeof(int), 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

    } else
    {
        // Copy input to d_input
        cudaError_t err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);//added by NAT
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array input from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
}

__host__ void executeKernel(int *d_input, int numElements, int threadsPerBlock, std::string kernelType)
{
    //int totalFound = 0;
    int numThreads = 256;
    // Launch the search CUDA Kernel
    if (!strcmp(kernelType.c_str(), "global"))
    {
        globalMemorySearch<<<1,threadsPerBlock>>>(d_input, numElements, numThreads); // you will need to fill in function arguments appropriately
        //void globalMemorySearch(int *input, int numElements, int numThreads)
    } else if (!strcmp(kernelType.c_str(), "constant"))
    {
       constantMemorySearch<<<1,threadsPerBlock>>>();  // you will need to fill in function arguments appropriately
    } else if (!strcmp(kernelType.c_str(), "shared"))
    {
        unsigned int_array_size = numElements * sizeof(int);
        sharedMemorySearch<<<1,threadsPerBlock, int_array_size>>>(d_input, numElements, numThreads); // you will need to fill in function arguments appropriately
    } else {
        registerMemorySearch<<<1,threadsPerBlock>>>(d_input, numElements, numThreads);  // you will need to fill in function arguments appropriately
    }

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch %s kernel (error code %s)!\n", kernelType.c_str(), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
}

// Free device global memory
__host__ void deallocateMemory(int *d_input)
{

    cudaError_t err = cudaFree(d_input);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__host__ std::tuple<int, std::string, int, std::string> parseCommandLineArguments(int argc, char *argv[])
{
    int elementsPerThread = 2;
    int threadsPerBlock = 256;
    std::string currentPartId = "test";
    std::string kernelType = "global";

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if(option.compare("-m") == 0)
        {
            elementsPerThread = atoi(value.c_str());
        }
        else if(option.compare("-p") == 0)
        {
            currentPartId = value;
        }
        else if(option.compare("-k") == 0)
        {
            kernelType = value;
        }
    }

    return {elementsPerThread, currentPartId, threadsPerBlock, kernelType};
}

__host__ int * setUpInput(int numElements, int threadsPerBlock)
{
    srand(time(0));
    int *input;
    input = allocatePageableRandomHostMemory(numElements); //array with Random data

    int searchValue = rand() % 255;

    //__device__ __constant__ int constant_search_value;
    cudaError_t err = cudaMemcpyToSymbol(constant_search_value, &searchValue, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return input;
}

/*
 * Host main routine
 * -m elementsPerThread - the number of elements that a thread will search for a random value in
 * -p currentPartId - the Coursera Part ID
 * -t threadsPerBlock - the number of threads to schedule for concurrent processing
 * -k the kernel type - global, constant, shared, register
 */
int main(int argc, char *argv[])
{
    auto[elementsPerThread, currentPartId, threadsPerBlock, kernelType] = parseCommandLineArguments(argc, argv);

    //int numElements = elementsPerThread * threadsPerBlock;

    int *input = setUpInput(elementsPerThread);
    int *d_input = allocateDeviceMemory(elementsPerThread);
    //int total;
    //int *totalFound;
    /*cudaError_t err = cudaMalloc(&totalFound, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
*/
    copyFromHostToDevice(kernelType, input, numElements, threadsPerBlock, d_input);

    // Start time including kernel processing time
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    executeKernel(d_input, elementsPerThread, threadsPerBlock, kernelType);

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);

    deallocateMemory(d_input);
    cleanUpDevice();

    ofstream outputfile;
    outputfile.open ("output.csv", std::ios_base::app);
    outputfile << currentPartId.c_str() << "," << kernelType.c_str() << "," << threadsPerBlock << "," << elementsPerThread << "," << elapsedTime << "\n";
    outputfile.close();

    return 0;
}
