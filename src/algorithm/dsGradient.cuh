#ifndef DS_GRADIENT_CUH
#define DS_GRADIENT_CUH

#include "misc/CudaTools.cuh"

template<typename T>
__global__ void
gradient(const T *input, size_t x_num, size_t y_num, size_t z_num, T *grad, size_t x_num_ds, size_t y_num_ds,
         float hx, float hy, float hz) {
    const int xi = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
    const int yi = ((blockIdx.y * blockDim.y) + threadIdx.y) * 2;
    const int zi = ((blockIdx.z * blockDim.z) + threadIdx.z) * 2;
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
                float yd = (temp[z - 1][x][y] - temp[z + 1][x][y]) / (2 * hy);
                yd = yd * yd;
                float zd = (temp[z][x][y - 1] - temp[z][x][y + 1]) / (2 * hz);
                zd = zd * zd;
                float gm = __fsqrt_rn(xd + yd + zd);
                if (gm > maxGrad) maxGrad = gm;
            }

    const size_t idx = zi / 2 * x_num_ds * y_num_ds + xi / 2 * y_num_ds + yi / 2;
    grad[idx] = maxGrad;
}

template<typename T>
void runKernelGradient(const T *cudaInput, T *cudaGrad,
                       size_t xLenInput, size_t yLenInput, size_t zLenInput, size_t xLenGradient,
                       size_t yLenGradient,
                       float hx, float hy, float hz, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 32, 1);
    dim3 numBlocks((xLenInput + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (yLenInput + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (zLenInput + threadsPerBlock.z - 1) / threadsPerBlock.z);
    printCudaDims(threadsPerBlock, numBlocks);
    gradient <<< numBlocks, threadsPerBlock, 0, aStream >>> (cudaInput, xLenInput, yLenInput, zLenInput, cudaGrad, xLenGradient, yLenGradient, hx, hy, hz);
}


#endif
