#include "ComputeInverseCubicBsplineCuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "invBspline.cuh"
#include "misc/CudaTools.hpp"

// explicit instantiation of handled types
template void cudaInverseBspline(PixelData<float> &, TypeOfInvBsplineFlags);

template <typename ImgType>
void cudaInverseBspline(PixelData<ImgType> &input, TypeOfInvBsplineFlags flags) {
    APRTimer timer(true), timerFullPipelilne(true);
    size_t inputSize = input.mesh.size() * sizeof(ImgType);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    constexpr int numOfWorkers = 32;
    timerFullPipelilne.start_timer("cuda: calculations on device FULL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ");
    if (flags & INV_BSPLINE_Y_DIR) {
        timer.start_timer("cuda: calculations on device Y ============================================================================ ");
        dim3 threadsPerBlock(1, numOfWorkers, 1);
        dim3 numBlocks((input.x_num + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       1,
                       (input.z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
        printCudaDims(threadsPerBlock, numBlocks);
        invBsplineYdir<ImgType> <<< numBlocks, threadsPerBlock>>> (cudaInput, input.x_num, input.y_num, input.z_num);
        waitForCuda();
        timer.stop_timer();
    }
    if (flags & INV_BSPLINE_X_DIR) {
        timer.start_timer("cuda: calculations on device X ============================================================================ ");
        dim3 threadsPerBlock(1, numOfWorkers, 1);
        dim3 numBlocks(1,
                       (input.y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (input.z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
        printCudaDims(threadsPerBlock, numBlocks);
        invBsplineXdir<ImgType> <<< numBlocks, threadsPerBlock>>> (cudaInput, input.x_num, input.y_num, input.z_num);
        waitForCuda();
        timer.stop_timer();
    }
    if (flags & INV_BSPLINE_Z_DIR) {
        timer.start_timer("cuda: calculations on device Z ============================================================================ ");
        dim3 threadsPerBlock(1, numOfWorkers, 1);
        dim3 numBlocks((input.x_num + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (input.y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       1);
        printCudaDims(threadsPerBlock, numBlocks);
        invBsplineZdir<ImgType> <<< numBlocks, threadsPerBlock>>> (cudaInput, input.x_num, input.y_num, input.z_num);
        waitForCuda();
        timer.stop_timer();
    }
    timerFullPipelilne.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    getDataFromKernel(input, inputSize, cudaInput);
    timer.stop_timer();
}
