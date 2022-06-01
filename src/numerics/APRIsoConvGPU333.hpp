//
// Created by joel on 08.04.20.
//

#ifndef LIBAPR_APRISOCONVGPU333_HPP
#define LIBAPR_APRISOCONVGPU333_HPP

#include "APRDownsampleGPU.hpp"
#include "APRStencil.hpp"
#include "miscCuda.hpp"



/// high-level functions including data transfer

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_direct(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                                   VectorData<outputType>& output, VectorData<stencilType>& stencil,
                                   VectorData<treeType>& tree_data, bool reflective_bc);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                            VectorData<outputType>& output, VectorData<stencilType>& stencil, VectorData<treeType>& tree_data,
                            bool reflective_bc=false, bool use_stencil_downsample=false, bool normalize_stencil=false) {
    tree_access.init_gpu();
    access.init_gpu(tree_access);

    assert(stencil.size() == 27);
    VectorData<stencilType> stencil_vec;
    const int nlevels = use_stencil_downsample ? access.level_max() - access.level_min() : 1;
    APRStencil::get_downsampled_stencils(stencil, stencil_vec, nlevels, normalize_stencil);
    isotropic_convolve_333_direct(access, tree_access, input, output, stencil_vec, tree_data, reflective_bc);
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                            VectorData<outputType>& output, PixelData<stencilType>& stencil, VectorData<treeType>& tree_data,
                            bool reflective_bc=false, bool use_stencil_downsample=false, bool normalize_stencil=false) {
    tree_access.init_gpu();
    access.init_gpu(tree_access);

    assert(stencil.z_num == 3);
    assert(stencil.x_num == 3);
    assert(stencil.y_num == 3);
    VectorData<stencilType> stencil_vec;
    const int nlevels = use_stencil_downsample ? access.level_max() - access.level_min() : 1;
    APRStencil::get_downsampled_stencils(stencil, stencil_vec, nlevels, normalize_stencil);
    isotropic_convolve_333_direct(access, tree_access, input, output, stencil_vec, tree_data, reflective_bc);
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                                VectorData<stencilType>& stencil, VectorData<treeType>& tree_data, bool use_stencil_downsample = false, bool normalize_stencil = false);


/// helper functions launching kernels for data already on the device

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter, bool use_stencil_downsample);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_reflective(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                                       stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter, bool use_stencil_downsample);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                                stencilType* stencil_gpu, treeType* tree_data_gpu, bool use_stencil_downsample);


/// kernels

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                     const uint64_t* __restrict__ xz_end_vec,
                                     const uint16_t* __restrict__ y_vec,
                                     const inputType* __restrict__ input_particles,
                                     outputType* __restrict__ output_particles,
                                     const stencilType* __restrict__ stencil,
                                     const int z_num,
                                     const int x_num,
                                     const int y_num,
                                     const int z_num_parent,
                                     const int x_num_parent,
                                     const int level,
                                     const int* __restrict__ offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                          const uint64_t* __restrict__ xz_end_vec,
                                          const uint16_t* __restrict__ y_vec,
                                          const inputType* __restrict__ input_particles,
                                          outputType* __restrict__ output_particles,
                                          const stencilType* __restrict__ stencil,
                                          const uint64_t* __restrict__ level_xz_vec_tree,
                                          const uint64_t* __restrict__ xz_end_vec_tree,
                                          const uint16_t* __restrict__ y_vec_tree,
                                          const treeType* __restrict__ tree_data,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num,
                                          const int z_num_parent,
                                          const int x_num_parent,
                                          const int level,
                                          const int* __restrict__ offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_reflective(const uint64_t* __restrict__ level_xz_vec,
                                        const uint64_t* __restrict__ xz_end_vec,
                                        const uint16_t* __restrict__ y_vec,
                                        const inputType* __restrict__ input_particles,
                                        outputType* __restrict__ output_particles,
                                        const stencilType* __restrict__ stencil,
                                        const int z_num,
                                        const int x_num,
                                        const int y_num,
                                        const int z_num_parent,
                                        const int x_num_parent,
                                        const int level,
                                        const int* __restrict__ offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_reflective(const uint64_t* __restrict__ level_xz_vec,
                                             const uint64_t* __restrict__ xz_end_vec,
                                             const uint16_t* __restrict__ y_vec,
                                             const inputType* __restrict__ input_particles,
                                             outputType* __restrict__ output_particles,
                                             const stencilType* __restrict__ stencil,
                                             const uint64_t* __restrict__ level_xz_vec_tree,
                                             const uint64_t* __restrict__ xz_end_vec_tree,
                                             const uint16_t* __restrict__ y_vec_tree,
                                             const treeType* __restrict__ tree_data,
                                             const int z_num,
                                             const int x_num,
                                             const int y_num,
                                             const int z_num_parent,
                                             const int x_num_parent,
                                             const int level,
                                             const int* __restrict__ offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                     const uint64_t* __restrict__ xz_end_vec,
                                     const uint16_t* __restrict__ y_vec,
                                     const inputType* __restrict__ input_particles,
                                     outputType* __restrict__ output_particles,
                                     const stencilType* __restrict__ stencil,
                                     const int z_num,
                                     const int x_num,
                                     const int y_num,
                                     const int z_num_parent,
                                     const int x_num_parent,
                                     const int level);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                          const uint64_t* __restrict__ xz_end_vec,
                                          const uint16_t* __restrict__ y_vec,
                                          const inputType* __restrict__ input_particles,
                                          outputType* __restrict__ output_particles,
                                          const stencilType* __restrict__ stencil,
                                          const uint64_t* __restrict__ level_xz_vec_tree,
                                          const uint64_t* __restrict__ xz_end_vec_tree,
                                          const uint16_t* __restrict__ y_vec_tree,
                                          const treeType* __restrict__ tree_data,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num,
                                          const int z_num_parent,
                                          const int x_num_parent,
                                          const int level);

#endif //LIBAPR_APRISOCONVGPU333_HPP
