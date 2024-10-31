# Specify the compiler
CC = g++
NVCC = nvcc

# Compiler flags for C++ and CUDA
CFLAGS = -Wall -std=c++17
CUDA_FLAGS = --std=c++17 -Wno-deprecated-gpu-targets

# Add OpenCV include and library paths
INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = -L/usr/lib/x86_64-linux-gnu

# Libraries to link
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

# Target executable
TARGET = ResMandelPar2.exe

# Source files
SRC_CUDA = ResMandelPar2.cu

# Build rule for the target
$(TARGET): $(SRC_CUDA)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) -o $(TARGET) $(SRC_CUDA)

# Clean up object files and executables
clean:
	rm -f $(TARGET)
