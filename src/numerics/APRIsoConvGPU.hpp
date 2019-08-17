//
// Created by Joel Jonsson on 2019-07-14.
//

#ifndef LIBAPR_APRISOCONVGPU_HPP
#define LIBAPR_APRISOCONVGPU_HPP

#include "APRDownsampleGPU.hpp"

#ifdef __CUDACC__
    #define L(x,y) __launch_bounds__(x,y)
#else
    #define L(x,y)
#endif

struct timings {
    float transfer_H2D = 0;
    float fill_tree = 0;
    float run_kernels = 0;
    float transfer_D2H = 0;
    float allocation = 0;
    float compute_ne_rows = 0;
    float compute_ne_rows_interior = 0;
    float init_access = 0;
    std::vector<float> lvl_timings;

    uint64_t counter_ne_rows = 0;
    uint64_t counter_ne_rows_int = 0;
};

void run_check_blocks(GPUAccessHelper& access, bool* blocks_empty);


/// new 333 max method
template<typename inputType, typename outputType, typename stencilType>
void run_max_333_new(GPUAccessHelper& access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu);

/// old 333 max method
template<typename inputType, typename outputType, typename stencilType>
void run_max_333_old(GPUAccessHelper& access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, bool* blocks_empty);
/// end of new stuff


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_333(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil);


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_333_basic(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil);


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_555(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<inputType>&,
                                VectorData<outputType>&, VectorData<stencilType>&, VectorData<treeType>&);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<inputType>&,
                                VectorData<outputType>&, VectorData<stencilType>&, VectorData<treeType>&);


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
__global__ void //L(100, 16)
conv_interior_333(const uint64_t* level_xz_vec,
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


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_chunked(const uint64_t* level_xz_vec,
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
                                     const int level,const int* offset_ind);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_chunked(const uint64_t* level_xz_vec,
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
__global__ void conv_pixel_333_chunked(const inputType* input_image,
                                       outputType* output_image,
                                       const stencilType* stencil,
                                       const int z_num,
                                       const int x_num,
                                       const int y_num);


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333_basic_kernel(const inputType* input_image,
                                             outputType* output_image,
                                             const stencilType* stencil,
                                             const int z_num,
                                             const int x_num,
                                             const int y_num);

#endif //LIBAPR_APRISOCONVGPU_HPP
