#include "ComputeBsplineRecursiveFilterCuda.h"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "bsplineXdir.cuh"
#include "bsplineYdir.cuh"
#include "bsplineZdir.cuh"


typedef struct {
    std::vector<float> bc1;
    std::vector<float> bc2;
    std::vector<float> bc3;
    std::vector<float> bc4;
    size_t k0;
    float b1;
    float b2;
    float norm_factor;
} BsplineParams;

float impulse_resp(float k, float rho, float omg){
    //  Impulse Response Function
    return (pow(rho,(std::abs(k)))*sin((std::abs(k) + 1)*omg)) / sin(omg);
}

float impulse_resp_back(float k,float rho,float omg,float gamma,float c0){
    //  Impulse Response Function (nominator eq. 4.8, denominator from eq. 4.7)
    return c0*pow(rho,std::abs(k))*(cos(omg*std::abs(k)) + gamma*sin(omg*std::abs(k)))*(1.0/(pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2)));
}

template <typename T>
BsplineParams prepareBsplineStuff(MeshData<T> & image, float lambda, float tol) {
    // Recursive Filter Implimentation for Smoothing BSplines
    // B-Spline Signal Processing: Part II - Efficient Design and Applications, Unser 1993

    float xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda); // eq 4.6
    float rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda))); // eq 4.5
    float omg = atan(sqrt((1/xi)*(144*lambda - 1))); // eq 4.6

    float c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2)); // eq 4.8
    float gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg)); // eq 4.8

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const size_t idealK0Len = ceil(std::abs(log(tol)/log(rho)));
    const size_t minDimension = std::min(image.z_num, std::min(image.x_num, image.y_num));
    const size_t k0 = std::min(idealK0Len, minDimension);

    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);
    std::cout << "GPU: " << xi << " " << rho << " " << omg << " " << gamma << " " << b1 << " " << b2 << " " << k0 << " " << norm_factor << std::endl;
    std::cout << "k0=" << k0 << std::endl;

    // ------- Calculating boundary conditions

    // forward boundaries
    std::vector<float> impulse_resp_vec_f(k0 + 1);
    for (size_t k = 0; k < impulse_resp_vec_f.size(); ++k) impulse_resp_vec_f[k] = impulse_resp(k, rho, omg);

    //y(0) init
    std::vector<float> bc1(k0, 0);
    for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
    //y(1) init
    std::vector<float> bc2(k0, 0);
    bc2[1] = impulse_resp_vec_f[0];
    for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k+1];

    // backward boundaries
    std::vector<float> impulse_resp_vec_b(k0 + 1);
    for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k) impulse_resp_vec_b[k] = impulse_resp_back(k, rho, omg, gamma, c0);

    //y(N-1) init
    std::vector<float> bc3(k0, 0);
    bc3[0] = impulse_resp_vec_b[1];
    for (size_t k = 0; k < (k0-1); ++k) bc3[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    //y(N) init
    std::vector<float> bc4(k0, 0);
    bc4[0] = impulse_resp_vec_b[0];
    for (size_t k = 1; k < k0; ++k) bc4[k] += 2*impulse_resp_vec_b[k];


    return BsplineParams {
            bc1,
            bc2,
            bc3,
            bc4,
            k0,
            b1,
            b2,
            norm_factor
    };
}

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
    std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
}

template <typename ImgType>
void cudaFilterBsplineXdirection(MeshData<ImgType> &input, float lambda, float tolerance) {
    APRTimer timer;
    timer.verbose_flag=true;
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    thrust::device_vector<float> d_bc1(p.bc1);
    thrust::device_vector<float> d_bc2(p.bc2);
    thrust::device_vector<float> d_bc3(p.bc3);
    thrust::device_vector<float> d_bc4(p.bc4);
    float *bc1= raw_pointer_cast(d_bc1.data());
    float *bc2= raw_pointer_cast(d_bc2.data());
    float *bc3= raw_pointer_cast(d_bc3.data());
    float *bc4= raw_pointer_cast(d_bc4.data());
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    constexpr int numOfWorkersYdir = 64;
    dim3 threadsPerBlock(1, numOfWorkersYdir, 1);
    dim3 numBlocks(1,
                   (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    printCudaDims(threadsPerBlock, numBlocks);
    timer.start_timer("cuda: calculations on device ============================================================================ ");
    bsplineXdir<ImgType> <<<numBlocks, threadsPerBlock>>> (cudaInput, input.x_num, input.y_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    getDataFromKernel(input, inputSize, cudaInput);
    timer.stop_timer();
}

template <typename ImgType>
void cudaFilterBsplineZdirection(MeshData<ImgType> &input, float lambda, float tolerance) {
    APRTimer timer;
    timer.verbose_flag=true;
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    thrust::device_vector<float> d_bc1(p.bc1);
    thrust::device_vector<float> d_bc2(p.bc2);
    thrust::device_vector<float> d_bc3(p.bc3);
    thrust::device_vector<float> d_bc4(p.bc4);
    float *bc1= raw_pointer_cast(d_bc1.data());
    float *bc2= raw_pointer_cast(d_bc2.data());
    float *bc3= raw_pointer_cast(d_bc3.data());
    float *bc4= raw_pointer_cast(d_bc4.data());
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    constexpr int numOfWorkersYdir = 64;
    dim3 threadsPerBlock(1, numOfWorkersYdir, 1);
    dim3 numBlocks((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   1);
    printCudaDims(threadsPerBlock, numBlocks);
    timer.start_timer("cuda: calculations on device ============================================================================ ");
    bsplineZdir<ImgType> <<<numBlocks, threadsPerBlock>>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    getDataFromKernel(input, inputSize, cudaInput);
    timer.stop_timer();
}

template <typename ImgType>
void cudaFilterBsplineYdirection(MeshData<ImgType> &input, float lambda, float tolerance) {
    APRTimer timer;
    timer.verbose_flag=true;
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    thrust::device_vector<float> d_bc1(p.bc1);
    thrust::device_vector<float> d_bc2(p.bc2);
    thrust::device_vector<float> d_bc3(p.bc3);
    thrust::device_vector<float> d_bc4(p.bc4);
    float *bc1= raw_pointer_cast(d_bc1.data());
    float *bc2= raw_pointer_cast(d_bc2.data());
    float *bc3= raw_pointer_cast(d_bc3.data());
    float *bc4= raw_pointer_cast(d_bc4.data());
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    float *boundary;
    int boundaryLen = sizeof(float) * (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
    cudaMalloc(&boundary, boundaryLen);
    timer.stop_timer();

//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    timer.start_timer("cuda: calculations on device ============================================================================ ");
    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((input.x_num * input.z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
    printCudaDims(threadsPerBlock, numBlocks);
    size_t sharedMemSize = (2 /*bc vectors*/) * (p.k0) * sizeof(float) + numOfThreads * (p.k0) * sizeof(ImgType);
    bsplineYdirBoundary<ImgType> <<<numBlocks, threadsPerBlock, sharedMemSize>>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, boundary);
    sharedMemSize = numOfThreads * blockWidth * sizeof(ImgType);
    bsplineYdirProcess<ImgType> <<<numBlocks, threadsPerBlock, sharedMemSize>>> (cudaInput, input.x_num, input.y_num, input.z_num, p.k0, p.b1, p.b2, p.norm_factor, boundary);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    getDataFromKernel(input, inputSize, cudaInput);
    timer.stop_timer();
}

template <typename ImgType>
void cudaFilterBsplineFull(MeshData<ImgType> &input, float lambda, float tolerance) {
    APRTimer timer;
    timer.verbose_flag=true;
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    thrust::device_vector<float> d_bc1(p.bc1);
    thrust::device_vector<float> d_bc2(p.bc2);
    thrust::device_vector<float> d_bc3(p.bc3);
    thrust::device_vector<float> d_bc4(p.bc4);
    float *bc1= raw_pointer_cast(d_bc1.data());
    float *bc2= raw_pointer_cast(d_bc2.data());
    float *bc3= raw_pointer_cast(d_bc3.data());
    float *bc4= raw_pointer_cast(d_bc4.data());
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    float *boundary;
    int boundaryLen = sizeof(float) * (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
    cudaMalloc(&boundary, boundaryLen);
    timer.stop_timer();

//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    timer.start_timer("cuda: calculations on device FULL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ");
//    timer.start_timer("cuda: calculations on device Y ============================================================================ ");
    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((input.x_num * input.z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
    printCudaDims(threadsPerBlock, numBlocks);
    size_t sharedMemSize = (2 /*bc vectors*/) * (p.k0) * sizeof(float) + numOfThreads * (p.k0) * sizeof(ImgType);
    bsplineYdirBoundary<ImgType> <<<numBlocks, threadsPerBlock, sharedMemSize>>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, boundary);
    sharedMemSize = numOfThreads * blockWidth * sizeof(ImgType);
    bsplineYdirProcess<ImgType> <<<numBlocks, threadsPerBlock, sharedMemSize>>> (cudaInput, input.x_num, input.y_num, input.z_num, p.k0, p.b1, p.b2, p.norm_factor, boundary);
    waitForCuda();
//    timer.stop_timer();
    constexpr int numOfWorkersYdir = 64;
    dim3 threadsPerBlockX(1, numOfWorkersYdir, 1);
    dim3 numBlocksX(1,
                    (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                    (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    printCudaDims(threadsPerBlockX, numBlocksX);
//    timer.start_timer("cuda: calculations on device X ============================================================================ ");
    bsplineXdir<ImgType> <<<numBlocksX, threadsPerBlockX>>> (cudaInput, input.x_num, input.y_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
    waitForCuda();
//    timer.stop_timer();
    dim3 threadsPerBlockZ(1, numOfWorkersYdir, 1);
    dim3 numBlocksZ((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                    (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                    1);
    printCudaDims(threadsPerBlockZ, numBlocksZ);
//    timer.start_timer("cuda: calculations on device Z ============================================================================ ");
    bsplineZdir<ImgType> <<<numBlocksZ, threadsPerBlockZ>>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
    waitForCuda();
//    timer.stop_timer();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    getDataFromKernel(input, inputSize, cudaInput);
    timer.stop_timer();
}

void emptyCallForTemplateInstantiation() {
    MeshData<float> f = MeshData<float>(0,0,0);
    MeshData<uint16_t> u16 = MeshData<uint16_t>(0,0,0);

    cudaFilterBsplineYdirection(f, 3, 0.1);
    cudaFilterBsplineYdirection(u16, 3, 0.1);

    cudaFilterBsplineXdirection(f, 3, 0.1);
    cudaFilterBsplineXdirection(u16, 3, 0.1);

    cudaFilterBsplineZdirection(f, 3, 0.1);
    cudaFilterBsplineZdirection(u16, 3, 0.1);

    cudaFilterBsplineFull(f, 3, 0.1);
    cudaFilterBsplineFull(u16, 3, 0.1);
}
