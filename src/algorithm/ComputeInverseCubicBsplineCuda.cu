#include "ComputeInverseCubicBsplineCuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {
    void waitForCuda() {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
    }

    template<typename ImgType>
    void getDataFromKernel(MeshData<ImgType> &input, size_t inputSize, ImgType *cudaInput) {
        cudaMemcpy(input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
        cudaFree(cudaInput);
    }

    void printCudaDims(const dim3 &threadsPerBlock, const dim3 &numBlocks) {
        std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z
                  << std::endl;
        std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/"
                  << threadsPerBlock.z << std::endl;
    }

    void emptyCallForTemplateInstantiation() {
        MeshData<float> f = MeshData<float>(0, 0, 0);
        MeshData<uint16_t> u16 = MeshData<uint16_t>(0, 0, 0);
        MeshData<uint8_t> u8 = MeshData<uint8_t>(0, 0, 0);


        cudaInverseBsplineYdir(f);
        cudaInverseBsplineYdir(u16);
        cudaInverseBsplineYdir(u8);
    }
}

template<typename T>
__global__ void invBsplineYdir(T *image, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workersOffset = blockIdx.x * blockDim.x * y_num + blockIdx.z * blockDim.z * y_num * x_num;
    const int workerIdx = threadIdx.y;
    const unsigned int active = __activemask();
    int workerOffset = workerIdx;
    int loopNum = 0;

    T oldPrevious = 0;
    T p = 0;
    T v = 0;
    bool notLastInRow = true;
    while (workerOffset < y_num) {
        if (notLastInRow) v = image[workersOffset + workerOffset];
        T temp = __shfl_sync(active, v, workerIdx + blockDim.y - 1, blockDim.y);
        p = notLastInRow ? temp : p;
        T n = __shfl_sync(active, v, workerIdx + 1, blockDim.y);

        // handle boundary (reflective mode)
        if (workerOffset == 0) p = n;
        if (workerOffset == y_num - 1) n = p;

        notLastInRow = (workerIdx + 1 + loopNum) % blockDim.y != 0;
        if (notLastInRow) {
            v = (p + v * 4 + n) / 6.0;
            image[workersOffset + workerOffset] = v;
            workerOffset += blockDim.y;
        }

        loopNum++;
    }
}

template <typename ImgType>
void cudaInverseBsplineYdir(MeshData<ImgType> &input, TypeOfFlags flags) {
    APRTimer timer(true), timerFullPipelilne(true);
    size_t inputSize = input.mesh.size() * sizeof(ImgType);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

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
    timerFullPipelilne.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    getDataFromKernel(input, inputSize, cudaInput);
    timer.stop_timer();
}
