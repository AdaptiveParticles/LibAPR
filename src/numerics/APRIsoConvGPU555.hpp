//
// Created by joel on 08.04.20.
//

#ifndef LIBAPR_APRISOCONVGPU555_HPP
#define LIBAPR_APRISOCONVGPU555_HPP

#include "APRDownsampleGPU.hpp"
#include "APRStencilFunctions.hpp"
#include "miscCuda.hpp"
#include "APRIsoConvGPU333.hpp"


/// high-level functions including data transfer

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                            VectorData<stencilType>& stencil, VectorData<treeType>& tree_data, bool reflective_bc = false,
                            bool use_stencil_downsample = false, bool normalize_stencil = false);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                                VectorData<stencilType>& stencil, VectorData<treeType>& tree_data, bool use_stencil_downsample = false, bool normalize_stencil = false);

/// helper functions launching kernels for data already on the device

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_reflective(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,outputType* output_gpu,
                                       stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                                outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                               outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu,
                               int* ne_rows_555, VectorData<int>& ne_counter_555, int* ne_rows_333, VectorData<int>& ne_counter_333);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds_reflective(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                                          outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_555,
                                          VectorData<int>& ne_counter_555, int* ne_rows_333, VectorData<int>& ne_counter_333);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                                   outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu);

/// kernels

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_555_chunked(const uint64_t* level_xz_vec,
                                     const uint64_t* xz_end_vec,
                                     const uint16_t* y_vec,
                                     const inputType* input_particles,
                                     outputType* output_particles,
                                     const stencilType* stencil,
                                     const int z_num,
                                     const int x_num,
                                     const int y_num,
                                     const int z_num_parent,
                                     const int x_num_parent,
                                     const int level,
                                     const int* offset_ind);

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_555_chunked(const uint64_t* level_xz_vec,
                                          const uint64_t* xz_end_vec,
                                          const uint16_t* y_vec,
                                          const inputType* input_particles,
                                          outputType* output_particles,
                                          const stencilType* stencil,
                                          const uint64_t* level_xz_vec_tree,
                                          const uint64_t* xz_end_vec_tree,
                                          const uint16_t* y_vec_tree,
                                          const treeType* tree_data,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num,
                                          const int z_num_parent,
                                          const int x_num_parent,
                                          const int level,
                                          const int* offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_555_reflective(const uint64_t* level_xz_vec,
                                        const uint64_t* xz_end_vec,
                                        const uint16_t* y_vec,
                                        const inputType* input_particles,
                                        outputType* output_particles,
                                        const stencilType* stencil,
                                        const int z_num,
                                        const int x_num,
                                        const int y_num,
                                        const int z_num_parent,
                                        const int x_num_parent,
                                        const int level,
                                        const int* offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_555_reflective(const uint64_t* level_xz_vec,
                                             const uint64_t* xz_end_vec,
                                             const uint16_t* y_vec,
                                             const inputType* input_particles,
                                             outputType* output_particles,
                                             const stencilType* stencil,
                                             const uint64_t* level_xz_vec_tree,
                                             const uint64_t* xz_end_vec_tree,
                                             const uint16_t* y_vec_tree,
                                             const treeType* tree_data,
                                             const int z_num,
                                             const int x_num,
                                             const int y_num,
                                             const int z_num_parent,
                                             const int x_num_parent,
                                             const int level,
                                             const int* offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_555_chunked(const uint64_t* level_xz_vec,
                                     const uint64_t* xz_end_vec,
                                     const uint16_t* y_vec,
                                     const inputType* input_particles,
                                     outputType* output_particles,
                                     const stencilType* stencil,
                                     const int z_num,
                                     const int x_num,
                                     const int y_num,
                                     const int z_num_parent,
                                     const int x_num_parent,
                                     const int level);

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_555_chunked(const uint64_t* level_xz_vec,
                                          const uint64_t* xz_end_vec,
                                          const uint16_t* y_vec,
                                          const inputType* input_particles,
                                          outputType* output_particles,
                                          const stencilType* stencil,
                                          const uint64_t* level_xz_vec_tree,
                                          const uint64_t* xz_end_vec_tree,
                                          const uint16_t* y_vec_tree,
                                          const treeType* tree_data,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num,
                                          const int z_num_parent,
                                          const int x_num_parent,
                                          const int level);

#endif //LIBAPR_APRISOCONVGPU555_HPP
