#ifndef DS_GRADIENT_CUH
#define DS_GRADIENT_CUH

#include "misc/CudaTools.cuh"

template<typename T>
__global__ void
gradient(const T *input, PixelDataDim inputDim, T *grad, PixelDataDim gradDim, float hx, float hy, float hz) {
    const int xi = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
    const int yi = ((blockIdx.y * blockDim.y) + threadIdx.y) * 2;
    const int zi = ((blockIdx.z * blockDim.z) + threadIdx.z) * 2;
    const auto x_num = inputDim.x;
    const auto y_num = inputDim.y;
    const auto z_num = inputDim.z;

    if (xi >= x_num || yi >= y_num || zi >= z_num) return;

    const size_t xnumynum = x_num * y_num;

    float temp[4][4][4];

    for (int z = 0; z < 4; ++z)
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y) {
                int xc = xi + x - 1;
                if (xc < 0) xc = 0; else if (xc >= x_num) xc = x_num - 1;
                int yc = yi + y - 1;
                if (yc < 0) yc = 0; else if (yc >= y_num) yc = y_num - 1;
                int zc = zi + z - 1;
                if (zc < 0) zc = 0; else if (zc >= z_num) zc = z_num - 1;
                temp[z][x][y] = *(input + zc * xnumynum + xc * y_num + yc);
            }
    float maxGrad = 0;
    for (int z = 1; z <= 2; ++z)
        for (int x = 1; x <= 2; ++x)
            for (int y = 1; y <= 2; ++y) {
                float xd = (temp[z][x - 1][y] - temp[z][x + 1][y]) / (2 * hx);
                xd = xd * xd;
                float zd = (temp[z - 1][x][y] - temp[z + 1][x][y]) / (2 * hz);
                zd = zd * zd;
                float yd = (temp[z][x][y - 1] - temp[z][x][y + 1]) / (2 * hy);
                yd = yd * yd;
                float gm = sqrtf(xd + zd + yd);
                if (gm > maxGrad) maxGrad = gm;
            }

    const size_t idx = zi / 2 * gradDim.x * gradDim.y + xi / 2 * gradDim.y + yi / 2;
    grad[idx] = maxGrad;
}

template<typename T>
void runKernelGradient(const T *cudaInput, T *cudaGrad,
                       PixelDataDim inputDim,
                       PixelDataDim gradDim,
                       float hx, float hy, float hz, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks((inputDim.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (inputDim.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (inputDim.z + threadsPerBlock.z - 1) / threadsPerBlock.z);
    gradient <<<numBlocks, threadsPerBlock, 0, aStream>>> (cudaInput, inputDim, cudaGrad, gradDim, hx, hy, hz);
}


#endif
