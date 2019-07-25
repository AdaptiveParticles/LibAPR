//
// Created by Joel Jonsson on 2019-07-14.
//

#ifndef LIBAPR_APRISOCONVGPU_HPP
#define LIBAPR_APRISOCONVGPU_HPP

#include "APRDownsampleGPU.hpp"

struct timings {
    float transfer_H2D = 0;
    float fill_tree = 0;
    float run_kernels = 0;
    float transfer_D2H = 0;
};

void run_check_blocks(GPUAccessHelper& access, bool* blocks_empty);


/// new 333 max method
template<typename inputType, typename outputType, typename stencilType>
void run_max_333_new_wrapper(GPUAccessHelper& access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu);

template<typename inputType, typename outputType, typename stencilType>
void run_max_333_new(GPUAccessHelper& access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu) {
    run_max_333_new_wrapper(access, input_gpu, output_gpu, stencil_gpu);
}

/// old 333 max method
template<typename inputType, typename outputType, typename stencilType>
void run_max_333_old_wrapper(GPUAccessHelper& access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, bool* blocks_empty);

template<typename inputType, typename outputType, typename stencilType>
void run_max_333_old(GPUAccessHelper& access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, bool* blocks_empty) {
    run_max_333_old_wrapper(access, input_gpu, output_gpu, stencil_gpu, blocks_empty);
}
/// end of new stuff


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_333_wrapper(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil);


template<typename inputType, typename outputType, typename stencilType>
timings pixel_convolve_333(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil) {

    return convolve_pixel_333_wrapper(input, output, stencil);
}


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_555_wrapper(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil);


template<typename inputType, typename outputType, typename stencilType>
timings pixel_convolve_555(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil) {

    return convolve_pixel_555_wrapper(input, output, stencil);
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333_wrapper(GPUAccessHelper&, GPUAccessHelper&, std::vector<inputType>&,
                                    std::vector<outputType>&, std::vector<stencilType>&, std::vector<treeType>&);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                            std::vector<outputType>& output, std::vector<stencilType>& stencil, std::vector<treeType>& tree_data) {

    return isotropic_convolve_333_wrapper(access, tree_access, input, output, stencil, tree_data);
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555_wrapper(GPUAccessHelper&, GPUAccessHelper&, std::vector<inputType>&,
                                    std::vector<outputType>&, std::vector<stencilType>&, std::vector<treeType>&);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                            std::vector<outputType>& output, std::vector<stencilType>& stencil, std::vector<treeType>& tree_data) {

    return isotropic_convolve_555_wrapper(access, tree_access, input, output, stencil, tree_data);
}




template<unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
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
                             const int level,
                             const bool* blocks_empty);


template<unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
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
                                  const int level,
                                  const bool* blocks_empty);


template<unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
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
                             const int level,
                             const bool* blocks_empty);


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
                             const int level,
                             const bool* blocks_empty);


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
                                  const int level,
                                  const bool* blocks_empty);


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
                             const int level,
                             const bool* blocks_empty);


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333(const inputType* input_image,
                               outputType* output_image,
                               const stencilType* stencil,
                               const int z_num,
                               const int x_num,
                               const int y_num);


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_555(const inputType* input_image,
                               outputType* output_image,
                               const stencilType* stencil,
                               const int z_num,
                               const int x_num,
                               const int y_num);


template<unsigned int blockSize>
__global__ void check_blocks(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             bool* blocks_nonempty,
                             const int level,
                             const int x_num);


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_alt(const uint64_t* level_xz_vec,
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
                                 const int level);

#endif //LIBAPR_APRISOCONVGPU_HPP
