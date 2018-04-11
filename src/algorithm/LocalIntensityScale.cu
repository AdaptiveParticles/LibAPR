#include "LocalIntensityScaleCuda.h"

#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {
    void waitForCuda() {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
    }

    void emptyCallForTemplateInstantiation() {
        MeshData<float> f = MeshData<float>(0, 0, 0);
        MeshData<uint16_t> u16 = MeshData<uint16_t>(0, 0, 0);
        MeshData<uint8_t> u8 = MeshData<uint8_t>(0, 0, 0);

        calcMeanYdir(f, 0);
//        calcMeanYdir(u16, 0);
//        calcMeanYdir(u8, 0);
    }

    void printCudaDims(const dim3 &threadsPerBlock, const dim3 &numBlocks) {
        std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
        std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
    }
}

template <typename T>
__global__ void meanYdir(T *image, int offset, size_t x_num, size_t y_num, size_t z_num) {
    // NOTE: Block size in x/z direction must be 1
    const size_t workersOffset = (blockIdx.z * x_num + blockIdx.x) * y_num;
    const int numOfWorkers = blockDim.y;
    const unsigned int active = __activemask();
    const int workerIdx = threadIdx.y;
    int workerOffset = workerIdx;

    int offsetInTheLoop = 0;
    T sum = 0;
    T v = 0;
    bool waitForNextLoop = false;
    int countNumOfSumElements = 1;
    while(workerOffset < y_num) {
        if (!waitForNextLoop) v = image[workersOffset + workerOffset];
        bool waitForNextValues = (workerIdx + offsetInTheLoop) % numOfWorkers >= (numOfWorkers - offset);
        for (int off = 1; off <= offset; ++off) {
            T prevElement = __shfl_sync(active, v, workerIdx + blockDim.y - off, blockDim.y);
            T nextElement = __shfl_sync(active, v, workerIdx + off, blockDim.y);
            // LHS boundary check + don't add previous values if they were added in a previous loop execution
            if (workerOffset >= off && !waitForNextLoop) {sum += prevElement; ++countNumOfSumElements;}
            // RHS boundary check + don't read next values since they are not read yet
            if (!waitForNextValues && workerOffset + off < y_num) {sum += nextElement; ++countNumOfSumElements;}
        }
        waitForNextLoop = waitForNextValues;
        if (!waitForNextLoop) {
            sum += v;
            image[workersOffset + workerOffset] = sum / countNumOfSumElements;

            // workere is done with current element - move to next one
            sum = 0;
            countNumOfSumElements = 1;
            workerOffset += numOfWorkers;
        }
        offsetInTheLoop += offset;
    }
}

template <typename T>
void calcMeanYdir(MeshData<T> &image, int offset) {
    APRTimer timer(true);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t imageSize = image.mesh.size() * sizeof(T);
    T *cudaImage;
    cudaMalloc(&cudaImage, imageSize);
    cudaMemcpy(cudaImage, image.mesh.get(), imageSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    timer.start_timer("cuda: calculations on device");
    dim3 threadsPerBlock(1, 32, 1);
    dim3 numBlocks((image.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   1,
                   (image.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    printCudaDims(threadsPerBlock, numBlocks);
    meanYdir<<<numBlocks,threadsPerBlock>>>(cudaImage, offset, image.x_num, image.y_num, image.z_num);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)image.mesh.get(), cudaImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaImage);
    timer.stop_timer();
}
