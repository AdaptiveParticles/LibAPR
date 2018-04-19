#include "LocalIntensityScaleCuda.h"

#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "misc/CudaTools.hpp"


/**
 *
 * How it works along y-dir (let's suppose offset = 2 and number of workers = 8 for simplicity):
 *
 * image idx: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2
 *
 * loop #1
 * workersIdx 0 1 2 3 4 5 6 7
 * loop #2
 * workersIdx             6 7 0 1 2 3 4 5
 * loop #3
 * workersIdx                         4 5 6 7 0 1 2 3
 * ..............
 *
 * so #offset workers must wait in each loop to have next elements to sum
 *
 * @tparam T
 * @param image
 * @param offset
 * @param x_num
 * @param y_num
 * @param z_num
 */
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

constexpr int NumberOfWorkers = 32; // Cannot be greater than 32 since there is no inter-warp communication implemented.

/**
 * Filter in X-dir moves circular buffer along direction adding to sum of elements newly read element and removing last one.
 * For instance (filter len = 5)
 *
 * idx:               0 1 2 3 4 5 6 7 8 9
 * image elements:    1 2 2 4 5 3 2 1 3 4
 *
 * buffer:                2 3 4 5 2                        current sum = 16  element @idx=4 will be updated to 16/5
 *
 * next step
 * buffer:                  3 4 5 2 1                      sum = sum - 2 + 1 = 15  element @idx=5 = 15 / 5
 *
 * In general circular buffer is kept to speedup operations and to not reach to global memory more than once for
 * read/write operations for given element.
 */
template <typename T>
__global__ void meanXdir(T *image, int offset, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.z * blockDim.z + threadIdx.z) * y_num * x_num;
    const int workerYoffset = blockIdx.y * blockDim.y + threadIdx.y ;
    const int workerIdx = threadIdx.y;
    const int nextElementOffset = y_num;

    extern __shared__ float sharedMem[];
    float (*data)[NumberOfWorkers] = (float (*)[NumberOfWorkers])sharedMem;

    const int divisor = 2 * offset  + 1;
    int currElementOffset = 0;
    int saveElementOffset = 0;

    if (workerYoffset < y_num) {
        // clear shared mem
        for (int i = offset; i < divisor; ++i) data[i][workerIdx] = 0;

        // saturate cache with #offset elements since it will allow to calculate first element value on LHS
        float sum = 0;
        int count = 0;
        while (count < offset) {
            T v = image[workerOffset + currElementOffset];
            sum += v;
            data[count][workerIdx] = v;
            currElementOffset += nextElementOffset;
            ++count;
        }

        // Pointer in circular buffer
        int beginPtr = offset;

        // main loop going through all elements in range [0, x_num-offset)
        for (int x = 0; x < x_num - offset; ++x) {
            // Read new element
            T v = image[workerOffset + currElementOffset];

            // Update sum to cover [-offset, offset] of currently processed element
            sum += v;
            sum -= data[beginPtr][workerIdx];

            // Save and move pointer
            data[beginPtr][workerIdx] = v;
            beginPtr = (beginPtr + 1) % divisor;

            // Update count and save currently processed element
            count = min(count + 1, divisor);
            image[workerOffset + saveElementOffset] = sum / count;

            // Move to next elements
            currElementOffset += nextElementOffset;
            saveElementOffset += nextElementOffset;
        }

        // Handle last #offset elements on RHS
        while (saveElementOffset < currElementOffset) {
            count = count - 1;
            sum -= data[beginPtr][workerIdx];
            image[workerOffset + saveElementOffset] = sum / count;
            beginPtr = (beginPtr + 1) % divisor;
            saveElementOffset += nextElementOffset;
        }
    }
}

/**
 * Filter in Z-dir moves circular buffer along direction adding to sum of elements newly read element and removing last one.
 * For instance (filter len = 5)
 *
 * idx:               0 1 2 3 4 5 6 7 8 9
 * image elements:    1 2 2 4 5 3 2 1 3 4
 *
 * buffer:                2 3 4 5 2                        current sum = 16  element @idx=4 will be updated to 16/5
 *
 * next step
 * buffer:                  3 4 5 2 1                      sum = sum - 2 + 1 = 15  element @idx=5 = 15 / 5
 *
 * In general circular buffer is kept to speedup operations and to not reach to global memory more than once for
 * read/write operations for given element.
 */
template <typename T>
__global__ void meanZdir(T *image, int offset, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.x * blockDim.x + threadIdx.x) * y_num;
    const int workerYoffset = blockIdx.y * blockDim.y + threadIdx.y ;
    const int workerIdx = threadIdx.y;
    const int nextElementOffset = x_num * y_num;

    extern __shared__ float sharedMem[];
    float (*data)[NumberOfWorkers] = (float (*)[NumberOfWorkers])sharedMem;

    const int divisor = 2 * offset  + 1;
    int currElementOffset = 0;
    int saveElementOffset = 0;

    if (workerYoffset < y_num) {
        // clear shared mem
        for (int i = offset; i < divisor; ++i) data[i][workerIdx] = 0;

        // saturate cache with #offset elements since it will allow to calculate first element value on LHS
        float sum = 0;
        int count = 0;
        while (count < offset) {
            T v = image[workerOffset + currElementOffset];
            sum += v;
            data[count][workerIdx] = v;
            currElementOffset += nextElementOffset;
            ++count;
        }

        // Pointer in circular buffer
        int beginPtr = offset;

        // main loop going through all elements in range [0, x_num-offset)
        for (int z = 0; z < z_num - offset; ++z) {
            // Read new element
            T v = image[workerOffset + currElementOffset];

            // Update sum to cover [-offset, offset] of currently processed element
            sum += v;
            sum -= data[beginPtr][workerIdx];

            // Save and move pointer
            data[beginPtr][workerIdx] = v;
            beginPtr = (beginPtr + 1) % divisor;

            // Update count and save currently processed element
            count = min(count + 1, divisor);
            image[workerOffset + saveElementOffset] = sum / count;

            // Move to next elements
            currElementOffset += nextElementOffset;
            saveElementOffset += nextElementOffset;
        }

        // Handle last #offset elements on RHS
        while (saveElementOffset < currElementOffset) {
            count = count - 1;
            sum -= data[beginPtr][workerIdx];
            image[workerOffset + saveElementOffset] = sum / count;
            beginPtr = (beginPtr + 1) % divisor;
            saveElementOffset += nextElementOffset;
        }
    }
}

template <typename T>
void calcMean(MeshData<T> &image, int offset, TypeOfMeanFlags flags) {
    APRTimer timer(true), timerFullPipelilne(true);

    timer.start_timer("GpuMemTransferHostToDevice");
    size_t imageSize = image.mesh.size() * sizeof(T);
    T *cudaImage;
    cudaMalloc(&cudaImage, imageSize);
    cudaMemcpy(cudaImage, image.mesh.get(), imageSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    timerFullPipelilne.start_timer("GpuDeviceTimeFull");
    if (flags & MEAN_Y_DIR) {
        timer.start_timer("GpuDeviceTimeYdir");
        dim3 threadsPerBlock(1, NumberOfWorkers, 1);
        dim3 numBlocks((image.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                       1,
                       (image.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
        printCudaDims(threadsPerBlock, numBlocks);
        meanYdir<<<numBlocks,threadsPerBlock>>>(cudaImage, offset, image.x_num, image.y_num, image.z_num);
        waitForCuda();
        timer.stop_timer();
    }

    // Shared memory size  - it is able to keep filter len elements for each worker.
    const int sharedMemorySize = (offset * 2 + 1) * sizeof(float) * NumberOfWorkers;

    if (flags & MEAN_X_DIR) {
        timer.start_timer("GpuDeviceTimeXdir");
        dim3 threadsPerBlock(1, NumberOfWorkers, 1);
        dim3 numBlocks(1,
                       (image.y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (image.z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
        printCudaDims(threadsPerBlock, numBlocks);
        meanXdir <<< numBlocks, threadsPerBlock, sharedMemorySize >>> (cudaImage, offset, image.x_num, image.y_num, image.z_num);
        waitForCuda();
        timer.stop_timer();
    }
    if (flags & MEAN_Z_DIR) {
        timer.start_timer("GpuDeviceTimeZdir");
        dim3 threadsPerBlock(1, NumberOfWorkers, 1);
        dim3 numBlocks((image.x_num + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (image.y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       1);
        printCudaDims(threadsPerBlock, numBlocks);
        meanZdir <<< numBlocks, threadsPerBlock, sharedMemorySize >>> (cudaImage, offset, image.x_num, image.y_num, image.z_num);
        waitForCuda();
        timer.stop_timer();
    }
    timerFullPipelilne.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)image.mesh.get(), cudaImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaImage);
    timer.stop_timer();
}

namespace {
    void emptyCallForTemplateInstantiation() {
        MeshData<float> f = MeshData<float>(0, 0, 0);
        MeshData<uint16_t> u16 = MeshData<uint16_t>(0, 0, 0);
        MeshData<uint8_t> u8 = MeshData<uint8_t>(0, 0, 0);

        calcMean(f, 0, 0);
        calcMean(u8, 0, 0);
        calcMean(u16, 0, 0);
    }
}

