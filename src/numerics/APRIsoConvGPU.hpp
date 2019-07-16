//
// Created by Joel Jonsson on 2019-07-14.
//

#ifndef LIBAPR_APRISOCONVGPU_HPP
#define LIBAPR_APRISOCONVGPU_HPP

#include "APRDownsampleGPU.hpp"


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_wrapper(GPUAccessHelper&, GPUAccessHelper&, std::vector<inputType>&,
                                    std::vector<outputType>&, std::vector<stencilType>&, std::vector<treeType>&);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                            std::vector<outputType>& output, std::vector<stencilType>& stencil, std::vector<treeType>& tree_data) {

    isotropic_convolve_333_wrapper(access, tree_access, input, output, stencil, tree_data);
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_wrapper(GPUAccessHelper&, GPUAccessHelper&, std::vector<inputType>&,
                                    std::vector<outputType>&, std::vector<stencilType>&, std::vector<treeType>&);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                            std::vector<outputType>& output, std::vector<stencilType>& stencil, std::vector<treeType>& tree_data) {

    isotropic_convolve_555_wrapper(access, tree_access, input, output, stencil, tree_data);
}


/// force template instantiation for some different type combinations
//333
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<float>&, std::vector<float>&);
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<float>&);
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<double>&);
//555
template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<float>&, std::vector<float>&);
template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<float>&);
template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<double>&);




template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             const uint16_t* y_vec,
                             const inputType* input_particles,
                             outputType* particle_data_output,
                             const stencilType* stencil,
                             const int z_num,
                             const int x_num,
                             const int y_num,
                             const int z_num_parent,
                             const int x_num_parent,
                             const int y_num_parent,
                             const int level);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333(const uint64_t* level_xz_vec,
                                  const uint64_t* xz_end_vec,
                                  const uint16_t* y_vec,
                                  const inputType* input_particles,
                                  outputType* particle_data_output,
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
                                  const int y_num_parent,
                                  const int level);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_min_333(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             const uint16_t* y_vec,
                             const inputType* input_particles,
                             outputType* particle_data_output,
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
                             const int y_num_parent,
                             const int level);


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_555(const uint64_t* level_xz_vec,
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
                             const int y_num_parent,
                             const int level);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_555(const uint64_t* level_xz_vec,
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
                                  const int y_num_parent,
                                  const int level);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_min_555(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             const uint16_t* y_vec,
                             const inputType* input_particles,
                             outputType* particle_data_output,
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
                             const int y_num_parent,
                             const int level);

#endif //LIBAPR_APRISOCONVGPU_HPP
