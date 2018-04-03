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
    std::vector<float> bc1_vec;
    std::vector<float> bc2_vec;
    std::vector<float> bc3_vec;
    std::vector<float> bc4_vec;
    size_t k0;
    float b1;
    float b2;
    float norm_factor;
} BsplineParams;

float impulse_resp(float k,float rho,float omg){
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
    // B-Spline Signal Processing: Part 11-Efficient Design and Applications, Unser 1993

    float xi = 1 - 96*lambda + 24*lambda*sqrt(3 + 144*lambda); // eq 4.6
    float rho = (24*lambda - 1 - sqrt(xi))/(24*lambda)*sqrt((1/xi)*(48*lambda + 24*lambda*sqrt(3 + 144*lambda))); // eq 4.5
    float omg = atan(sqrt((1/xi)*(144*lambda - 1))); // eq 4.6

    float c0 = (1+ pow(rho,2))/(1-pow(rho,2)) * (1 - 2*rho*cos(omg) + pow(rho,2))/(1 + 2*rho*cos(omg) + pow(rho,2)); // eq 4.8
    float gamma = (1-pow(rho,2))/(1+pow(rho,2)) * (1/tan(omg)); // eq 4.8

    const float b1 = 2*rho*cos(omg);
    const float b2 = -pow(rho,2.0);

    const size_t xxx = ceil(std::abs(log(tol)/log(rho)));

    const size_t minDimension = std::min(image.z_num, std::min(image.x_num, image.y_num));
    const size_t k0 = std::min(xxx, minDimension);

    const float norm_factor = pow((1 - 2.0*rho*cos(omg) + pow(rho,2)),2);
    std::cout << "GPU: " << xi << " " << rho << " " << omg << " " << gamma << " " << b1 << " " << b2 << " " << k0 << " " << norm_factor << std::endl;
    //////////////////////////////////////////////////////////////
    //
    //  Setting up boundary conditions
    //
    //////////////////////////////////////////////////////////////

    // for boundaries
    std::cout << "k0=" << k0 << std::endl;
    std::vector<float> impulse_resp_vec_f(k0+3);  //forward
    for (size_t k = 0; k < (k0+3); ++k) {
        impulse_resp_vec_f[k] = impulse_resp(k,rho,omg);
    }

    std::vector<float> impulse_resp_vec_b(k0+3);  //backward
    for (size_t k = 0; k < (k0+3); ++k) {
        impulse_resp_vec_b[k] = impulse_resp_back(k,rho,omg,gamma,c0);
    }

    std::vector<float> bc1_vec(k0, 0);  //forward
    //y(1) init
    bc1_vec[1] = impulse_resp_vec_f[0];
    for (size_t k = 0; k < k0; ++k) {
        bc1_vec[k] += impulse_resp_vec_f[k+1];
    }

    std::vector<float> bc2_vec(k0, 0);  //backward
    //y(0) init
    for (size_t k = 0; k < k0; ++k) {
        bc2_vec[k] = impulse_resp_vec_f[k];
    }

    std::vector<float> bc3_vec(k0, 0);  //forward
    //y(N-1) init
    bc3_vec[0] = impulse_resp_vec_b[1];
    for (size_t k = 0; k < (k0-1); ++k) {
        bc3_vec[k+1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k+2];
    }

    std::vector<float> bc4_vec(k0, 0);  //backward
    //y(N) init
    bc4_vec[0] = impulse_resp_vec_b[0];
    for (size_t k = 1; k < k0; ++k) {
        bc4_vec[k] += 2*impulse_resp_vec_b[k];
    }

    return BsplineParams {
            bc1_vec,
            bc2_vec,
            bc3_vec,
            bc4_vec,
            k0,
            b1,
            b2,
            norm_factor
    };
}

//========== First naive version following CPU code ===============
template <typename T>
__global__ void bsplineY(T *image, size_t x_num, size_t y_num, size_t z_num, float *bc1_vec, float *bc2_vec, float *bc3_vec, float *bc4_vec, size_t k0, float b1, float b2, float norm_factor) {
    int xi = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int zi = ((blockIdx.z * blockDim.z) + threadIdx.z);
    if (xi >= x_num || zi >= z_num) return;

    //forwards direction
    const size_t zPlaneOffset = zi * x_num * y_num;
    const size_t yColOffset = xi * y_num;
    size_t yCol = zPlaneOffset + yColOffset;

    float temp1 = 0;
    float temp2 = 0;
    float temp3 = 0;
    float temp4 = 0;

    for (size_t k = 0; k < k0; ++k) {
        temp1 += bc1_vec[k]*image[yCol + k];
        temp2 += bc2_vec[k]*image[yCol + k];
        temp3 += bc3_vec[k]*image[yCol + y_num - 1 - k];
        temp4 += bc4_vec[k]*image[yCol + y_num - 1 - k];
    }

    //initialize the sequence
    image[yCol + 0] = temp2;
    image[yCol + 1] = temp1;

    // middle values
    for (auto it = (image + yCol + 2); it !=  (image+yCol + y_num); ++it) {
        float  temp = temp1*b1 + temp2*b2 + *it;
        *it = temp;
        temp2 = temp1;
        temp1 = temp;
    }

    // finish sequence
    image[yCol + y_num - 2] = temp3;
    image[yCol + y_num - 1] = temp4;

    // -------------- part 2
    temp2 = image[yCol + y_num - 1];
    temp1 = image[yCol + y_num - 2];
    image[yCol + y_num - 1]*=norm_factor;
    image[yCol + y_num - 2]*=norm_factor;

    for (auto it = (image + yCol + y_num-3); it !=  (image + yCol - 1); --it) {
        float temp = temp1*b1 + temp2*b2 + *it;
        *it = temp*norm_factor;
        temp2 = temp1;
        temp1 = temp;
    }
}

template <typename ImgType>
void cudaFilterBsplineYdirection(MeshData<ImgType> &input, float lambda, float tolerance) {
    APRTimer timer;
    timer.verbose_flag=true;

    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    ImgType *cudaInput;
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);

    float *boundary;
    int boundaryLen = sizeof(float) * 4 * input.x_num * input.z_num;
    cudaMalloc(&boundary, boundaryLen);

    thrust::device_vector<float> d_bc1_vec(p.bc1_vec);
    thrust::device_vector<float> d_bc2_vec(p.bc2_vec);
    thrust::device_vector<float> d_bc3_vec(p.bc3_vec);
    thrust::device_vector<float> d_bc4_vec(p.bc4_vec);
    float *bc1 = thrust::raw_pointer_cast(d_bc1_vec.data());
    float *bc2 = thrust::raw_pointer_cast(d_bc2_vec.data());
    float *bc3 = thrust::raw_pointer_cast(d_bc3_vec.data());
    float *bc4 = thrust::raw_pointer_cast(d_bc4_vec.data());
    timer.stop_timer();

//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    timer.start_timer("cuda: calculations on device ============================================================================ ");
    if (true) {
        dim3 threadsPerBlock(numOfThreads);
        dim3 numBlocks((input.x_num * input.z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
        std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
        std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;

        size_t sharedMemSize = (2 /*bc vectors*/) * (p.k0) * sizeof(float) + numOfThreads * (p.k0) * sizeof(ImgType);
        bsplineYdirBoundary<ImgType> <<<numBlocks, threadsPerBlock, sharedMemSize>>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, boundary);
        sharedMemSize = numOfThreads * blockWidth * sizeof(ImgType);
        bsplineYdirProcess<ImgType> <<<numBlocks, threadsPerBlock, sharedMemSize>>> (cudaInput, input.x_num, input.y_num, input.z_num, p.k0, p.b1, p.b2, p.norm_factor, boundary);
    } else {
        // old naive approach
        dim3 threadsPerBlock(8, 1, 8);
        dim3 numBlocks((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                       1,
                       (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
        bsplineY<ImgType> <<<numBlocks, threadsPerBlock>>>(cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)printf("Error: %s\n", cudaGetErrorString(err));
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaInput);
    timer.stop_timer();
}

template <typename ImgType>
void cudaFilterBsplineXdirection(MeshData<ImgType> &input, float lambda, float tolerance) {
    APRTimer timer;
    timer.verbose_flag=true;

    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    float *boundary;
    int boundaryLen = sizeof(float) * 4 * input.x_num * input.z_num;
    cudaMalloc(&boundary, boundaryLen);

    thrust::device_vector<float> d_bc1_vec(p.bc1_vec);
    thrust::device_vector<float> d_bc2_vec(p.bc2_vec);
    thrust::device_vector<float> d_bc3_vec(p.bc3_vec);
    thrust::device_vector<float> d_bc4_vec(p.bc4_vec);
    timer.stop_timer();

//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    float *bc1 = thrust::raw_pointer_cast(d_bc1_vec.data());
    float *bc2 = thrust::raw_pointer_cast(d_bc2_vec.data());
    float *bc3 = thrust::raw_pointer_cast(d_bc3_vec.data());
    float *bc4 = thrust::raw_pointer_cast(d_bc4_vec.data());

    constexpr int numOfWorkersYdir = 64;
    dim3 threadsPerBlock(1, numOfWorkersYdir, 1);
    dim3 numBlocks(1,
                   (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
    timer.start_timer("cuda: calculations on device ============================================================================ ");
    bsplineXdir<ImgType> <<<numBlocks, threadsPerBlock>>> (cudaInput, input.x_num, input.y_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
    cudaDeviceSynchronize();
    timer.stop_timer();


    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)printf("Error: %s\n", cudaGetErrorString(err));

    cudaMemcpy((void*)input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaInput);
    timer.stop_timer();
}

template <typename ImgType>
void cudaFilterBsplineZdirection(MeshData<ImgType> &input, float lambda, float tolerance) {
    APRTimer timer;
    timer.verbose_flag=true;

    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    float *boundary;
    int boundaryLen = sizeof(float) * 4 * input.x_num * input.z_num;
    cudaMalloc(&boundary, boundaryLen);

    thrust::device_vector<float> d_bc1_vec(p.bc1_vec);
    thrust::device_vector<float> d_bc2_vec(p.bc2_vec);
    thrust::device_vector<float> d_bc3_vec(p.bc3_vec);
    thrust::device_vector<float> d_bc4_vec(p.bc4_vec);
    timer.stop_timer();
    float *bc1 = thrust::raw_pointer_cast(d_bc1_vec.data());
    float *bc2 = thrust::raw_pointer_cast(d_bc2_vec.data());
    float *bc3 = thrust::raw_pointer_cast(d_bc3_vec.data());
    float *bc4 = thrust::raw_pointer_cast(d_bc4_vec.data());

    constexpr int numOfWorkersYdir = 64;
    dim3 threadsPerBlock(1, numOfWorkersYdir, 1);
    dim3 numBlocks((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   1);
    std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
    timer.start_timer("cuda: calculations on device ============================================================================ ");
    bsplineZdir<ImgType> <<<numBlocks, threadsPerBlock>>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
    cudaDeviceSynchronize();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)printf("Error: %s\n", cudaGetErrorString(err));

    cudaMemcpy((void*)input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaInput);
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
}
