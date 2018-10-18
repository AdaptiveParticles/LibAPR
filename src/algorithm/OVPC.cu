#include "PullingSchemeCuda.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "misc/CudaTools.cuh"
#include "data_structures/Mesh/downsample.cuh"

// explicit instantiation of handled types
template void computeOVPC(const  PixelData<float>&, PixelData<float>&, int, int);

template <typename T, typename S>
void computeOVPC(const PixelData<T> &input, PixelData<S> &output, int levelMin, int levelMax) {
    ScopedCudaMemHandler<const PixelData<T>, H2D> in(input);
    ScopedCudaMemHandler<PixelData<S>, D2H> out(output);

    runDownsampleMax(in.get(), out.get(), input.x_num, input.y_num, input.z_num, 0);
};