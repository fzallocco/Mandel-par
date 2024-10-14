#include <iostream>
#include <chrono>

int main() {
    float a, b = 0
    // Using chrono for high-resolution timing
    auto start = std::chrono::high_resolution_clock::now();

    // Code to be timed
    for (int i = 0; i < 1000000; ++i) {
        // Perform some operation
        a= srand(time());
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}