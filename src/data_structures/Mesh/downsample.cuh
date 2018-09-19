#ifndef DOWNSAMPLE_CUH
#define DOWNSAMPLE_CUH

template <typename T, typename S>
__global__ void downsampleMean(const T *input, S *output, size_t x_num, size_t y_num, size_t z_num) {
    const size_t xi = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
    const size_t zi = ((blockIdx.z * blockDim.z) + threadIdx.z) * 2;
    if (xi >= x_num || zi >= z_num) return;

    size_t yi = ((blockIdx.y * blockDim.y) + threadIdx.y);
    if (yi == y_num && yi % 2 == 1) {
        // In case when y is odd we need last element to pair with last even y (boundary in y-dir)
        yi = y_num - 1;
    }
    else if (yi >= y_num) {
        return;
    }

    // Handle boundary in x/y direction
    int xs =  xi + 1 > x_num - 1 ? 0 : 1;
    int zs =  zi + 1 > z_num - 1 ? 0 : 1;

    // Index of first element
    size_t idx = (zi * x_num + xi) * y_num + yi;

    // Go through all elements in 2x2
    T v = input[idx];
    v +=  input[idx + xs * y_num];
    v +=  input[idx +              zs * x_num * y_num];
    v +=  input[idx + xs * y_num + zs * x_num * y_num];

    // Get data from odd thread to even one
    const int workerIdx = threadIdx.y;
    T a = __shfl_sync(__activemask(), v, workerIdx + 1);

    // downsampled dimensions twice smaller (rounded up)

    if (workerIdx % 2 == 0) {
        // Finish calculations in even thread completing whole 2x2x2 cube.
        v += a;

        v /= 8.0; // calculate mean by dividing sum by 8

        // store result in downsampled mesh
        const size_t x_num_ds = ceilf(x_num/2.0);
        const size_t y_num_ds = ceilf(y_num/2.0);
        const size_t dsIdx = (zi/2 * x_num_ds + xi/2) * y_num_ds + yi/2;
        output[dsIdx] = v;
    }
}

template <typename T, typename S>
void runDownsampleMean(const T *cudaImage, S *cudalocal_scale_temp, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks(((x_num + threadsPerBlock.x - 1)/threadsPerBlock.x + 1) / 2,
                    (y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   ((z_num + threadsPerBlock.z - 1)/threadsPerBlock.z + 1) / 2);
    downsampleMean<<<numBlocks,threadsPerBlock, 0, aStream>>>(cudaImage, cudalocal_scale_temp, x_num, y_num, z_num);
}

template <typename T, typename S>
__global__ void downsampleMax(const T *input, S *output, size_t x_num, size_t y_num, size_t z_num) {
    const size_t xi = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
    const size_t zi = ((blockIdx.z * blockDim.z) + threadIdx.z) * 2;
    if (xi >= x_num || zi >= z_num) return;

    size_t yi = ((blockIdx.y * blockDim.y) + threadIdx.y);
    if (yi == y_num && yi % 2 == 1) {
        // In case when y is odd we need last element to pair with last even y (boundary in y-dir)
        yi = y_num - 1;
    }
    else if (yi >= y_num) {
        return;
    }

    // Handle boundary in x/y direction
    int xs =  xi + 1 > x_num - 1 ? 0 : 1;
    int zs =  zi + 1 > z_num - 1 ? 0 : 1;

    // Index of first element
    size_t idx = (zi * x_num + xi) * y_num + yi;

    // Go through all elements in 2x2
    T v = input[idx];
    v =  max(input[idx + xs * y_num], v);
    v =  max(input[idx +              zs * x_num * y_num], v);
    v =  max(input[idx + xs * y_num + zs * x_num * y_num], v);

    // Get data from odd thread to even one
    const int workerIdx = threadIdx.y;
    T a = __shfl_sync(__activemask(), v, workerIdx + 1);

    // downsampled dimensions twice smaller (rounded up)

    if (workerIdx % 2 == 0) {
        // Finish calculations in even thread completing whole 2x2x2 cube.
        v = max(a, v);

        // store result in downsampled mesh
        const size_t x_num_ds = ceilf(x_num/2.0);
        const size_t y_num_ds = ceilf(y_num/2.0);
        const size_t dsIdx = (zi/2 * x_num_ds + xi/2) * y_num_ds + yi/2;
        output[dsIdx] = v;
    }
}

template <typename T, typename S>
void runDownsampleMax(const T *cudaImage, S *cudalocal_scale_temp, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks(((x_num + threadsPerBlock.x - 1)/threadsPerBlock.x + 1) / 2,
                   (y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   ((z_num + threadsPerBlock.z - 1)/threadsPerBlock.z + 1) / 2);
    downsampleMax<<<numBlocks,threadsPerBlock, 0, aStream>>>(cudaImage, cudalocal_scale_temp, x_num, y_num, z_num);
}

#endif