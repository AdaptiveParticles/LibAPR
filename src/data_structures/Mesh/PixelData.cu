#include "PixelDataCuda.h"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "misc/CudaTools.cuh"

#include "downsample.cuh"

// explicit instantiation of handled types
template void downsampleMeanCuda(const PixelData<float>&, PixelData<float>&);
template void downsampleMaxCuda(const  PixelData<float>&, PixelData<float>&);

template <typename T, typename S>
void downsampleMeanCuda(const PixelData<T> &input, PixelData<S> &output) {
    APRTimer timer(true);

    timer.start_timer("cuda: memory alloc + data transfer to device");

    size_t inputSize = input.mesh.size() * sizeof(T);
    T *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);

    size_t outputSize = output.mesh.size() * sizeof(float);
    float *cudaOutput;
    cudaMalloc(&cudaOutput, outputSize);
    cudaMemcpy(cudaOutput, output.mesh.get(), outputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    runDownsampleMean(cudaInput, cudaOutput, input.x_num, input.y_num, input.z_num, 0);

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaFree(cudaInput);
    cudaMemcpy((void*)output.mesh.get(), cudaOutput, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaOutput);
    timer.stop_timer();
};


enum CopyDir {
    H2D = 1,
    D2H = 2
};


template <typename T>
using is_pixel_data = std::is_same<typename std::remove_const<T>::type, PixelData<typename T::value_type> >;

template <typename T, bool IS_CONST = std::is_const<T>::value, bool CHECK_TYPE = is_pixel_data<T>::value>
class ScopedMemHandler {
    
    T &data;
    using data_type = typename T::value_type;
    CopyDir direction;
    data_type *cudaMem = nullptr;
    size_t size = 0;
public:
    ScopedMemHandler(T &aData, CopyDir aDirection) : data(aData), direction(aDirection) {
        size = data.mesh.size() * sizeof(typename T::value_type);
        cudaMalloc(&cudaMem, size);
        if (direction & H2D) {
            cudaMemcpy(cudaMem, data.mesh.get(), size, cudaMemcpyHostToDevice);
        }
    }

    ~ScopedMemHandler() {
        if (direction & D2H) {
            cudaMemcpy((void*)data.mesh.get(), cudaMem, size, cudaMemcpyDeviceToHost);
        }
        cudaFree(cudaMem);
    }

    data_type* get() {return cudaMem;}
};

template <typename T>
class ScopedMemHandler<T, true, true> {
    T &data;
    using data_type = typename T::value_type;
    CopyDir direction;
    data_type *cudaMem = nullptr;
    size_t size = 0;
public:
    ScopedMemHandler(T &aData, CopyDir aDirection) : data(aData), direction(aDirection) {
        size = data.mesh.size() * sizeof(typename T::value_type);
        cudaMalloc(&cudaMem, size);
        if (direction & H2D) {
            cudaMemcpy(cudaMem, data.mesh.get(), size, cudaMemcpyHostToDevice);
        }
    }

    ~ScopedMemHandler() {
        assert(!(direction & D2H));
        cudaFree(cudaMem);
    }

    const data_type* get() const {return cudaMem;}
};

template <typename T, bool B> class ScopedMemHandler<T, B, false>;

template <typename T, typename S>
void downsampleMaxCuda(const PixelData<T> &input, PixelData<S> &output) {
//    APRTimer timer(true);
//
//    timer.start_timer("cuda: memory alloc + data transfer to device");
//    size_t inputSize = input.mesh.size() * sizeof(T);
//    T *cudaInput;
//    cudaMalloc(&cudaInput, inputSize);
//    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
//    size_t outputSize = output.mesh.size() * sizeof(float);
//    float *cudaOutput;
//    cudaMalloc(&cudaOutput, outputSize);
//    cudaMemcpy(cudaOutput, output.mesh.get(), outputSize, cudaMemcpyHostToDevice);
//    timer.stop_timer();

    ScopedMemHandler<const PixelData<T>> in(input, H2D);
    ScopedMemHandler<PixelData<S>> out(output, D2H);
    runDownsampleMax(in.get(), out.get(), input.x_num, input.y_num, input.z_num, 0);

//    runDownsampleMax(cudaInput, cudaOutput, input.x_num, input.y_num, input.z_num, 0);
//
//
//
//    timer.start_timer("cuda: transfer data from device and freeing memory");
//    cudaFree(cudaInput);
//    cudaMemcpy((void*)output.mesh.get(), cudaOutput, outputSize, cudaMemcpyDeviceToHost);
//    cudaFree(cudaOutput);
//    timer.stop_timer();
};
