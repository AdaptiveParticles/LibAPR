#ifndef INV_BSPLINE_CUH
#define INV_BSPLINE_CUH

template<typename T>
__global__ void invBsplineYdir(T *image, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workersOffset = blockIdx.x * blockDim.x * y_num + blockIdx.z * blockDim.z * y_num * x_num;
    const int workerIdx = threadIdx.y;
    const unsigned int active = __activemask();
    int workerOffset = workerIdx;
    int loopNum = 0;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    float p = 0;
    float v = 0;
    bool notLastInRow = true;
    while (workerOffset < y_num) {
        if (notLastInRow) v = image[workersOffset + workerOffset];
        float temp = __shfl_sync(active, v, workerIdx + blockDim.y - 1, blockDim.y);
        p = notLastInRow ? temp : p;
        float n = __shfl_sync(active, v, workerIdx + 1, blockDim.y);

        // handle boundary (reflective mode)
        if (workerOffset == 0) p = n;
        if (workerOffset == y_num - 1) n = p;

        notLastInRow = (workerIdx + 1 + loopNum) % blockDim.y != 0;
        if (notLastInRow) {
            v = a1 * p + a2 * v + a3 * n;
            image[workersOffset + workerOffset] = v;
            workerOffset += blockDim.y;
        }

        loopNum++;
    }
}

template <typename T>
void runInvBsplineYdir(T* cudaInput, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream) {
    constexpr int numOfWorkers = 32;
    dim3 threadsPerBlock(1, numOfWorkers, 1);
    dim3 numBlocks((x_num + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   1,
                   (z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
    invBsplineYdir<T> <<<numBlocks, threadsPerBlock, 0, aStream>>> (cudaInput, x_num, y_num, z_num);
}

template<typename T>
__global__ void invBsplineXdir(T *image, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.z * blockDim.z + threadIdx.z) * y_num * x_num;
    const int workerIdx = blockIdx.y * blockDim.y + threadIdx.y ;
    const int nextElementOffset = y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    if (workerIdx < y_num) {
        int currElementOffset = 0;

        T v1 = image[workerOffset + currElementOffset];
        T v2 = image[workerOffset + currElementOffset + nextElementOffset];
        image[workerOffset + currElementOffset] = a1 * v2 + a2 * v1 + a3 * v2;

        for (int x = 2; x < x_num; ++x) {
            T v3 = image[workerOffset + currElementOffset + 2 * nextElementOffset];
            image[workerOffset + currElementOffset + nextElementOffset] = (a1 * v1 + a2 * v2 + a3 * v3);
            v1 = v2;
            v2 = v3;
            currElementOffset += nextElementOffset;
        }
        image[workerOffset + currElementOffset + nextElementOffset] = (a1 + a3) * v1 + a2 * v2;
    }
}

template <typename T>
void runInvBsplineXdir(T* cudaInput, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream) {
    constexpr int numOfWorkers = 32;
    dim3 threadsPerBlock(1, numOfWorkers, 1);
    dim3 numBlocks(1,
                   (y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
    invBsplineXdir<T> <<<numBlocks, threadsPerBlock, 0, aStream>>> (cudaInput, x_num, y_num, z_num);
}

template<typename T>
__global__ void invBsplineZdir(T *image, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.x * blockDim.x + threadIdx.x) * y_num;
    const int workerIdx = blockIdx.y * blockDim.y + threadIdx.y ;
    const int nextElementOffset = x_num * y_num;

    const float a1 = 1.0/6.0;
    const float a2 = 4.0/6.0;
    const float a3 = 1.0/6.0;

    if (workerIdx < y_num) {
        int currElementOffset = 0;

        T v1 = image[workerOffset + currElementOffset];
        T v2 = image[workerOffset + currElementOffset + nextElementOffset];
        image[workerOffset + currElementOffset] = a1 * v2 + a2 * v1 + a1 * v2;

        for (int x = 2; x < z_num; ++x) {
            T v3 = image[workerOffset + currElementOffset + 2 * nextElementOffset];
            image[workerOffset + currElementOffset + nextElementOffset] = a1 * v1 + a2 * v2 + a3 * v3;
            v1 = v2;
            v2 = v3;
            currElementOffset += nextElementOffset;
        }
        image[workerOffset + currElementOffset + nextElementOffset] = (a1 + a3) * v1 + a2 * v2;
    }
}

template <typename T>
void runInvBsplineZdir(T* cudaInput, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream) {
    constexpr int numOfWorkers = 32;
    dim3 threadsPerBlock(1, numOfWorkers, 1);
    dim3 numBlocks((x_num + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   1);
    invBsplineZdir<T> <<<numBlocks, threadsPerBlock, 0, aStream>>> (cudaInput, x_num, y_num, z_num);
}

#endif