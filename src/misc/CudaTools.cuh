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


// Useful type for keeping CUDA allocated memory (which is released with cudaFree)
template <typename T, typename D=decltype(&cudaFree)>
struct CudaMemoryUniquePtr : public std::unique_ptr<T[], D> {
    using std::unique_ptr<T[],D>::unique_ptr; // inheriting other constructors
    explicit CudaMemoryUniquePtr(T *aMemory = nullptr) : std::unique_ptr<T[], D>(aMemory, &cudaFree) {}
};

/**
 * Directions for sending data between Host and Device
 */
using CopyDirType = unsigned int;
enum CopyDir : CopyDirType {
    JUST_ALLOC = 0, // Just allocate memory on GPU(Device)
    H2D = 1,        // Send from Host to Device
    D2H = 2,        // Send from Device to Host
    INVALID = 4     // Just wrong/last value keeper for validating settings
};


/**
 * Checks if provided type is a PixelData container
 * @tparam T - type to check
 * @return true if PixelData and false otherwise
 */
template <typename T>
constexpr typename std::enable_if<std::is_pod<T>::value, bool>::type
is_pixel_data() {return false;}
template <typename T>
constexpr typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, PixelData<typename T::value_type>>::value, bool>::type
is_pixel_data() {return true;}

/**
 * Gets data type from  PixelData so:
 * - in case if provided wiht PixelData<T> it gives T
 * - in case if provided with const PixelData<T> it gives 'const T'
 * - in case of any other type T it gives back T
 */
template <typename T, bool B = is_pixel_data<T>()>
struct get_type {
    static constexpr bool IsDataConst = std::is_const<T>::value;
    using type = typename std::conditional<IsDataConst, typename std::add_const<typename T::value_type>::type, typename T::value_type>::type;
};
template <typename T>
struct get_type<T, false> {
    using type = T;
};

template <typename T>
using QualifiedBaseElementType = typename std::remove_pointer<typename get_type<T>::type>::type;
template <typename T>
using BaseElementType = typename std::remove_cv<QualifiedBaseElementType<T>>::type;

/**
 * ScopedCudaMemHandler is responsible allows allocating, copying to device for both - PixelData and pure memory pointed data.
 * After going out of scope it first copies back data to host from GPU and deallocating cuda memory.
 * Some examples:
 *
 * - on a host we have 'const float* mem;' data of lenght 10, to be copied to GPU:
 *     ScopedCudaMemHandler<const float*, H2D> m(mem, 10);
 * - on a host we have 'int* mem;' data of lenght 5, send to and from device:
 *     ScopedCudaMemHandler<int*, H2D | D2H> m(mem, 5);
 * - same situatino as above but with PixelData<int>:
 *     ScopedCudaMemHandler<PixelData<int>, H2D | D2H> m(mem);
 *
 * @tparam DATA_TYPE
 * @tparam DIRECTION - of type CopyDirType
 */
template <typename DATA_TYPE, CopyDirType DIRECTION>
class ScopedCudaMemHandler {
    using QualifiedElementType = QualifiedBaseElementType<DATA_TYPE>; // it preserves 'const' if in DATA_TYPE
    using ElementType = BaseElementType<DATA_TYPE>; // pure element type for GPU side

    static constexpr bool IsDataConst = std::is_const<typename std::remove_pointer<DATA_TYPE>::type>::value;
    static constexpr size_t DataSize = sizeof(ElementType);

    // Do some compile time checks
    static_assert(!(IsDataConst && (DIRECTION & D2H)), "Input is const, copying data from device back to host not possible!");
    static_assert(DIRECTION < INVALID, "Wrong value provided for DIRECTION template parameter");

    QualifiedElementType *iData;
    const size_t iSize;
    const size_t iBytes;
    CudaMemoryUniquePtr<ElementType> iCudaMemory;
    const cudaStream_t iStream;

public:
    /**
     * Constructor for pointer POD types
     */
    template <typename T = DATA_TYPE, typename std::enable_if<std::is_pointer<T>::value, int>::type = 0>
    explicit ScopedCudaMemHandler(DATA_TYPE aData, size_t aSize, const cudaStream_t aStream = nullptr) : iData(aData), iSize(aSize), iBytes(iSize * DataSize), iStream(aStream) {
        initialize();
    }

    /**
     * Constructor for PixelData<T> types
     */
    template<typename X = DATA_TYPE, typename std::enable_if<is_pixel_data<X>(), int>::type = 0>
    explicit ScopedCudaMemHandler(DATA_TYPE &aData, const cudaStream_t aStream = nullptr) : iData(aData.mesh.get()), iSize(aData.mesh.size()), iBytes(iSize * DataSize), iStream(aStream) {
        initialize();
    }

    ~ScopedCudaMemHandler() {
        copyD2H();
    }

    ElementType* get() {return iCudaMemory.get();}

private:
    ScopedCudaMemHandler(const ScopedCudaMemHandler&) = delete; // make it noncopyable
    ScopedCudaMemHandler& operator=(const ScopedCudaMemHandler&) = delete; // make it not assignable

    void initialize() {
        ElementType *mem = nullptr;
        cudaMalloc(&mem, iBytes);
        iCudaMemory.reset(mem);
        if (DIRECTION & H2D) {
            copyH2D();
        }
    }

    void copyH2D() {
        cudaMemcpyAsync(iCudaMemory.get(), iData, iBytes, cudaMemcpyHostToDevice, iStream);
    }

    void copyD2H() {
        if (DIRECTION & D2H) {
            cudaMemcpyAsync((void*)iData, iCudaMemory.get(), iBytes, cudaMemcpyDeviceToHost, iStream);
        }
    }
};

#endif //LIBAPR_CUDATOOLS_HPP
