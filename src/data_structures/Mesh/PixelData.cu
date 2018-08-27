#include "PixelDataCuda.h"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "misc/CudaTools.cuh"

#include "downsample.cuh"
#include <vector>

// explicit instantiation of handled types
template void downsampleMeanCuda(const PixelData<float>&, PixelData<float>&);
template void downsampleMaxCuda(const  PixelData<float>&, PixelData<float>&);

template <typename T, typename S>
void downsampleMeanCuda(const PixelData<T> &input, PixelData<S> &output) {
    ScopedCudaMemHandler<const PixelData<T>> in(input, H2D);
    ScopedCudaMemHandler<PixelData<S>> out(output, D2H);

    runDownsampleMean(in.get(), out.get(), input.x_num, input.y_num, input.z_num, 0);
};


template <typename T, typename S>
void downsampleMaxCuda(const PixelData<T> &input, PixelData<S> &output) {
    ScopedCudaMemHandler<const PixelData<T>> in(input, H2D);
    ScopedCudaMemHandler<PixelData<S>> out(output, D2H);

    runDownsampleMax(in.get(), out.get(), input.x_num, input.y_num, input.z_num, 0);
};
