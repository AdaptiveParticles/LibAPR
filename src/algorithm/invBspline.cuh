#ifndef INV_BSPLINE_CUH
#define INV_BSPLINE_CUH

template<typename T>
__global__ void invBsplineYdir(T *image, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workersOffset = blockIdx.x * blockDim.x * y_num + blockIdx.z * blockDim.z * y_num * x_num;
    const int workerIdx = threadIdx.y;
    const unsigned int active = __activemask();
    int workerOffset = workerIdx;
    int loopNum = 0;

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

template<typename T>
__global__ void invBsplineXdir(T *image, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.z * blockDim.z + threadIdx.z) * y_num * x_num;
    const int workerIdx = blockIdx.y * blockDim.y + threadIdx.y ;
    const int nextElementOffset = y_num;

    if (workerIdx < y_num) {
        int currElementOffset = 0;

        T v1 = image[workerOffset + currElementOffset];
        T v2 = image[workerOffset + currElementOffset + nextElementOffset];
        image[workerOffset + currElementOffset] = (2 * v2 + 4 * v1) / 6.0;

        for (int x = 2; x < x_num; ++x) {
            T v3 = image[workerOffset + currElementOffset + 2 * nextElementOffset];
            image[workerOffset + currElementOffset + nextElementOffset] = (v1 + 4 * v2 + v3) / 6.0;
            v1 = v2;
            v2 = v3;
            currElementOffset += nextElementOffset;
        }
        image[workerOffset + currElementOffset + nextElementOffset] = (2 * v1 + 4 * v2) / 6.0;
    }
}

template<typename T>
__global__ void invBsplineZdir(T *image, size_t x_num, size_t y_num, size_t z_num) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.x * blockDim.x + threadIdx.x) * y_num;
    const int workerIdx = blockIdx.y * blockDim.y + threadIdx.y ;
    const int nextElementOffset = x_num * y_num;

    if (workerIdx < y_num) {
        int currElementOffset = 0;

        T v1 = image[workerOffset + currElementOffset];
        T v2 = image[workerOffset + currElementOffset + nextElementOffset];
        image[workerOffset + currElementOffset] = (2 * v2 + 4 * v1) / 6.0;

        for (int x = 2; x < z_num; ++x) {
            T v3 = image[workerOffset + currElementOffset + 2 * nextElementOffset];
            image[workerOffset + currElementOffset + nextElementOffset] = (v1 + 4 * v2 + v3) / 6.0;
            v1 = v2;
            v2 = v3;
            currElementOffset += nextElementOffset;
        }
        image[workerOffset + currElementOffset + nextElementOffset] = (2 * v1 + 4 * v2) / 6.0;
    }
}

#endif