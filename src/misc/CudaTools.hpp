//
// Created by Krzysztof Gonciarz on 4/12/18.
//

#ifndef LIBAPR_CUDATOOLS_HPP
#define LIBAPR_CUDATOOLS_HPP


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include <iostream>
#include <chrono>
#include "data_structures/Mesh/PixelData.hpp"


inline void waitForCuda() {
//    cudaDeviceSynchronize();
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

inline void printCudaDims(const dim3 &threadsPerBlock, const dim3 &numBlocks) {
    std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
}

template<typename ImgType>
inline void getDataFromKernel(PixelData<ImgType> &input, size_t inputSize, ImgType *cudaInput) {
    cudaMemcpy(input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaInput);
}

class CudaTimer {
    std::vector<std::chrono::system_clock::time_point> iStartTimes;
    std::vector<std::string> names;
    bool iUseTimer;
    std::string iTimerName;

public:
    CudaTimer(bool aUseTimer, const std::string &aTimerName = "NO_NAME") : iUseTimer(aUseTimer), iTimerName(aTimerName) {
        if (iUseTimer) {
            std::cout << "--TIME-- " << iTimerName << " started!" << "\n";
        }
    }


    void start_timer(const std::string &timing_name) {
        if (iUseTimer) {
            for (int i = 0; i < iStartTimes.size(); ++i) std::cout << "    ";
            std::cout << "--TIME-- " << iTimerName << " [" << timing_name << "]\n";
            names.push_back(timing_name);
            std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
            iStartTimes.push_back(startTime);
        }
    }

    void stop_timer() {
        if (iUseTimer) {
            waitForCuda();
            std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
            if (iStartTimes.size() == 0) {
                std::cout << "ERROR: you have stopeed timer too many times..." << std::endl;
                return;
            }
            std::chrono::system_clock::time_point startTime = iStartTimes.back();
            iStartTimes.pop_back();
            std::chrono::duration<double> elapsedSeconds = endTime - startTime;
            auto name = names.back();
            names.pop_back();
            for (int i = 0; i < iStartTimes.size(); ++i) std::cout << "    ";
            std::cout << "--TIME-- " << iTimerName << " [" << name << "] = " << elapsedSeconds.count() << "\n";
        }
    }
};

#endif //LIBAPR_CUDATOOLS_HPP
