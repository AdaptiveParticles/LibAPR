//
// Created by Krzysztof Gonciarz on 8/1/18.
//

#ifndef LIBAPR_COMPUTEPULLINGSCHEMECUDA_H
#define LIBAPR_COMPUTEPULLINGSCHEMECUDA_H

template <typename T>
__global__ void computeLevels(const T *grad, float *lis, size_t len, float mult_const) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
        uint32_t d = (grad[idx] / lis[idx]) * mult_const;
        //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
        lis[idx] = (d == 0) ? 0 : 31 - __clz(d); // fast log2
    }
}

template <typename T>
void runComputeLevels(const T *grad, float *lis, size_t len, float mult_const, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((len + threadsPerBlock.x - 1) / threadsPerBlock.x);
    computeLevels <<< numBlocks, threadsPerBlock, 0, aStream >>> (grad, lis, len, mult_const);
}


#endif //LIBAPR_COMPUTEPULLINGSCHEMECUDA_HPP
