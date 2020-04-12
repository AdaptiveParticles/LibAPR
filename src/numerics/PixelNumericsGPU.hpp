//
// Created by joel on 07.04.20.
//

#ifndef LIBAPR_PIXELNUMERICSGPU_HPP
#define LIBAPR_PIXELNUMERICSGPU_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "miscCuda.hpp"

#ifdef APR_USE_CUDA
#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"
#endif

/// high-level functions including data transfers

template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_333(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil, bool reflective_bc = false);

template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_555(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil, bool reflective_bc = false);

/// helper functions launching kernels for data already on the device

template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_333(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num);

template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_333_reflective(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num);

template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_555(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num);

template<typename inputType, typename outputType, typename stencilType>
void convolve_pixel_555_reflective(inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, int y_num, int x_num, int z_num);

/// richardson lucy deconvolution

template<typename inputType, typename stencilType>
void richardson_lucy_pixel(inputType* input, stencilType* output, stencilType* psf, stencilType* psf_flipped, int kernel_size, int npixels, int niter, std::vector<int>& dims);

template<typename inputType, typename stencilType>
void richardson_lucy_pixel(PixelData<inputType>& input, PixelData<stencilType>& output, PixelData<stencilType>& psf, int niter);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333_chunked(const inputType* input_image,
                                       outputType* output_image,
                                       const stencilType* stencil,
                                       const int z_num,
                                       const int x_num,
                                       const int y_num);

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
conv_pixel_555_chunked(const inputType* input_image,
                       outputType* output_image,
                       const stencilType* stencil,
                       const int z_num,
                       const int x_num,
                       const int y_num);

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333_reflective(const inputType* input_image,
                                          outputType* output_image,
                                          const stencilType* stencil,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num);

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
conv_pixel_555_reflective(const inputType* input_image,
                          outputType* output_image,
                          const stencilType* stencil,
                          const int z_num,
                          const int x_num,
                          const int y_num);


#endif //LIBAPR_PIXELNUMERICSGPU_HPP
