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

// --------------- ScopedMemHandler ------------------------------------------------------------------------------------
// Helper to send data to/from GPU
// Creates CUDA memory and sends data (if requested) from PixelData container on creation and
// copies back data (if requested) and frees CUDA memory when destroyed

using CopyDir = int;
constexpr CopyDir H2D = 1;
constexpr CopyDir D2H = 2;

template <typename T>
using is_pixel_data = std::is_same<typename std::remove_cv<T>::type, PixelData<typename T::value_type>>;

template <typename PIXEL_DATA_T, bool CHECK_TYPE = is_pixel_data<PIXEL_DATA_T>::value>
class ScopedMemHandler {
    static_assert(is_pixel_data<PIXEL_DATA_T>::value == true, "Wrong data type provided, only PixelData is valid.");

    using data_type = typename PIXEL_DATA_T::value_type;
    using is_pixel_data_const = std::is_const<PIXEL_DATA_T>;

    PIXEL_DATA_T &data;
    const CopyDir direction;
    data_type *cudaMem = nullptr;
    size_t size = 0;

public:
    ScopedMemHandler(PIXEL_DATA_T &aData, const CopyDir aDirection) : data(aData), direction(aDirection) {
        size = data.mesh.size() * sizeof(typename PIXEL_DATA_T::value_type);
        cudaMalloc(&cudaMem, size);
        if (direction & H2D) {
            cudaMemcpy(cudaMem, data.mesh.get(), size, cudaMemcpyHostToDevice);
        }
    }

    ~ScopedMemHandler() {
        if (is_pixel_data_const::value) {
            const bool isCopyToHostRequested = direction & D2H;
            assert(!isCopyToHostRequested); // abort if wrong direction (works in debug mode)
            if (isCopyToHostRequested) throw std::invalid_argument("Device to host not possible for const PixelData!");
        }
        else if (direction & D2H) {
            cudaMemcpy((void*)data.mesh.get(), cudaMem, size, cudaMemcpyDeviceToHost);
        }
        cudaFree(cudaMem);
    }

    data_type* get() {return cudaMem;}
};

// Incomplete specialization to prevent using ScopedMemHandler with types different than PixelData
template <typename PIXEL_DATA_T> class ScopedMemHandler<PIXEL_DATA_T, false>;

#endif //LIBAPR_CUDATOOLS_HPP
