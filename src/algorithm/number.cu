#include <iostream>
#include "number.h"

// Kernel definition
__global__ void g_singleAnswer(long* answer){ *answer = 2; }

void getNumber(long *aValue) {
    long h_answer = 256;
    std::cout << "CUDA bef: " << h_answer << std::endl;
    long* d_answer;
    cudaMalloc(&d_answer, sizeof(long));
    std::cout << "Ask cuda for number." << std::endl;
    g_singleAnswer<<<1,1>>>(d_answer);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error: " << cudaGetErrorString(err) << "\n";
    cudaMemcpy(&h_answer, d_answer, sizeof(long), cudaMemcpyDeviceToHost);
    cudaFree(d_answer);


    std::cout << "CUDA returned: " << h_answer << std::endl;
    *aValue = h_answer;
}
