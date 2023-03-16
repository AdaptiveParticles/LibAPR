#include "PixelDataCuda.h"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "misc/CudaTools.cuh"

#include "downsample.cuh"
#include "paddPixelData.cuh"


// explicit instantiation of handled types
template void downsampleMeanCuda(const PixelData<float>&, PixelData<float>&);
template void downsampleMaxCuda(const  PixelData<float>&, PixelData<float>&);
template void paddPixelsCuda(const PixelData<float> &input, PixelData<float> &output, const PixelDataDim &padSize);
template void unpaddPixelsCuda(const PixelData<float> &input, PixelData<float> &output, const PixelDataDim &padSize);

template <typename T, typename S>
void downsampleMeanCuda(const PixelData<T> &input, PixelData<S> &output) {
    ScopedCudaMemHandler<const PixelData<T>, H2D> in(input);
    ScopedCudaMemHandler<PixelData<S>, D2H> out(output);

    runDownsampleMean(in.get(), out.get(), input.x_num, input.y_num, input.z_num, 0);
};

template <typename T, typename S>
void downsampleMaxCuda(const PixelData<T> &input, PixelData<S> &output) {
    ScopedCudaMemHandler<const PixelData<T>, H2D> in(input);
    ScopedCudaMemHandler<PixelData<S>, D2H> out(output);

    runDownsampleMax(in.get(), out.get(), input.x_num, input.y_num, input.z_num, 0);
};

template <typename T>
void paddPixelsCuda(const PixelData<T> &input, PixelData<T> &output, const PixelDataDim &padSize) {
    ScopedCudaMemHandler<const PixelData<T>, H2D> inputData(input);
    ScopedCudaMemHandler<PixelData<T>, D2H> outputData(output);

    runPaddPixels(inputData.get(), outputData.get(), input.getDimension(), output.getDimension(), padSize, 0);
};

template <typename T>
void unpaddPixelsCuda(const PixelData<T> &input, PixelData<T> &output, const PixelDataDim &padSize) {
    ScopedCudaMemHandler<const PixelData<T>, H2D> inputData(input);
    ScopedCudaMemHandler<PixelData<T>, D2H> outputData(output);

    runUnpaddPixels(inputData.get(), outputData.get(), input.getDimension(), output.getDimension(), padSize, 0);
};
