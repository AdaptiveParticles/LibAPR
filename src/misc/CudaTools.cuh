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

// --------------- ScopedCudaMemHandler ------------------------------------------------------------------------------------
// Helper to send data to/from GPU
// Creates CUDA memory and sends data (if requested) from PixelData container on creation and
// copies back data (if requested) and frees CUDA memory when destroyed


template <typename T, typename D=decltype(&cudaFree)>
struct CudaMemoryUniquePtr : public std::unique_ptr<T[], D> {
    using std::unique_ptr<T[],D>::unique_ptr; // inheriting other constructors
    explicit CudaMemoryUniquePtr(T *aMemory = nullptr) : std::unique_ptr<T[], D>(aMemory, &cudaFree) {}
};

using CopyDir = int;
constexpr CopyDir JUST_ALLOC = 0;
constexpr CopyDir H2D = 1;
constexpr CopyDir D2H = 2;

template <typename T>
constexpr typename std::enable_if<std::is_pod<T>::value, bool>::type
is_pixel_data() {return false;}

template <typename T>
constexpr typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, PixelData<typename T::value_type>>::value, bool>::type
is_pixel_data() {return true;}



template <typename PIXEL_DATA_T, bool CHECK_TYPE = is_pixel_data<PIXEL_DATA_T>()>
class ScopedCudaMemHandler {
    using DataType = typename PIXEL_DATA_T::value_type;
    static constexpr bool IsPixelDataConst = std::is_const<PIXEL_DATA_T>::value;

    PIXEL_DATA_T &iData;

    const CopyDir iDirection;
    CudaMemoryUniquePtr<DataType> iCudaMemory;
    size_t iSize = 0;
    cudaStream_t iStream;

public:
    explicit ScopedCudaMemHandler(PIXEL_DATA_T &aData, const CopyDir aDirection = JUST_ALLOC, const cudaStream_t aStream = nullptr) : iData(aData), iDirection(aDirection), iStream(aStream) {
        iSize = iData.mesh.size() * sizeof(typename PIXEL_DATA_T::value_type);
        DataType *mem = nullptr;
        cudaMalloc(&mem, iSize);
        iCudaMemory.reset(mem);
        if (iDirection & H2D) {
            cudaMemcpyAsync(iCudaMemory.get(), iData.mesh.get(), iSize, cudaMemcpyHostToDevice, iStream);
        }
    }

    ~ScopedCudaMemHandler() {
        if (IsPixelDataConst) {
            const bool isCopyToHostRequested = iDirection & D2H;
            if (isCopyToHostRequested) throw std::invalid_argument("Device to host not possible for const PixelData!");
        }
        else if (iDirection & D2H) {
            cudaMemcpyAsync((void*)iData.mesh.get(), iCudaMemory.get(), iSize, cudaMemcpyDeviceToHost, iStream);
        }
    }

    DataType* get() {return iCudaMemory.get();}

private:
    ScopedCudaMemHandler(const ScopedCudaMemHandler&) = delete; // make it noncopyable
    ScopedCudaMemHandler& operator=(const ScopedCudaMemHandler&) = delete; // make it not assignable

};

// Incomplete specialization to prevent using ScopedCudaMemHandler with types different than PixelData
//template <typename PIXEL_DATA_T> class ScopedCudaMemHandler<PIXEL_DATA_T, false>;

// =----------------------------

template <typename DATA_TYPE>
class ScopedCudaMemHandler<DATA_TYPE, false> {
    using DataType = DATA_TYPE;
    static constexpr bool IsDataConst = std::is_const<DATA_TYPE>::value;

    DataType *iData;
    const CopyDir iDirection;
    std::unique_ptr<DataType, decltype(&cudaFree)> iCudaMemory = {nullptr, &cudaFree};
    size_t iSize = 0;
    cudaStream_t iStream;

public:
    explicit ScopedCudaMemHandler(DataType *aData, size_t aSize, const CopyDir aDirection = JUST_ALLOC, const cudaStream_t aStream = nullptr) : iData(aData), iDirection(aDirection), iStream(aStream) {
        iSize = aSize * sizeof(DataType);
        DataType *mem = nullptr;
        cudaMalloc(&mem, iSize);
        iCudaMemory.reset(mem);
        if (iDirection & H2D) {
            cudaMemcpyAsync(iCudaMemory.get(), iData, iSize, cudaMemcpyHostToDevice, iStream);
        }
    }

    ~ScopedCudaMemHandler() {
        if (IsDataConst) {
            const bool isCopyToHostRequested = iDirection & D2H;
            if (isCopyToHostRequested) throw std::invalid_argument("Device to host not possible for const PixelData!");
        }
        else if (iDirection & D2H) {
            cudaMemcpyAsync((void*)iData, iCudaMemory.get(), iSize, cudaMemcpyDeviceToHost, iStream);
        }
    }

    DataType* get() {return iCudaMemory.get();}

private:
    ScopedCudaMemHandler(const ScopedCudaMemHandler&) = delete; // make it noncopyable
    ScopedCudaMemHandler& operator=(const ScopedCudaMemHandler&) = delete; // make it not assignable

};


#endif //LIBAPR_CUDATOOLS_HPP
