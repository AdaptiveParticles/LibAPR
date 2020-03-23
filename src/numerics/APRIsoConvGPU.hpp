//
// Created by Joel Jonsson on 2019-07-14.
//

#ifndef LIBAPR_APRISOCONVGPU_HPP
#define LIBAPR_APRISOCONVGPU_HPP

#include "APRDownsampleGPU.hpp"
#include "io/TiffUtils.hpp"
//#include "APRFilter.hpp"
#include "APRStencilFunctions.hpp"

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
    float compute_ne_rows_ds = 0;
    float init_access = 0;
    float downsample_stencil = 0;
    std::vector<float> lvl_timings;

    uint64_t counter_ne_rows = 0;
    uint64_t counter_ne_rows_int = 0;
};


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_333(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil);


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_555(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil);


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<inputType>&,
                                VectorData<outputType>&, VectorData<stencilType>&, VectorData<treeType>&);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<inputType>&,
                                   VectorData<outputType>&, VectorData<stencilType>&, VectorData<treeType>&);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data, int* ne_rows_gpu, VectorData<int>& ne_counter, bool downsample_stencil = false, stencilType pad_value = 0);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                            outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data, bool downsample_stencil = false, stencilType pad_value = 0);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<inputType>&,
                                VectorData<outputType>&, VectorData<stencilType>&, VectorData<treeType>&);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<inputType>&,
                               VectorData<outputType>&, VectorData<stencilType>&, VectorData<treeType>&);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter, stencilType pad_value = 0);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                            outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu, stencilType pad_value = 0);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                                  PixelData<stencilType>& stencil, VectorData<treeType>& tree_data, bool use_ne_rows = false, bool normalize_stencil = false);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                                  VectorData<outputType>& output, PixelData<stencilType>& stencil, VectorData<treeType>& tree_data,
                                  bool use_ne_rows = false, bool normalize_stencil = false);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                               outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu, stencilType pad_value = 0);

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                               outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu,
                               int* ne_rows_555, VectorData<int>& ne_counter_555, int* ne_rows_333, VectorData<int>& ne_counter_333, stencilType pad_value = 0);

template<typename inputType, typename stencilType>
void richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                     VectorData<stencilType>& output, PixelData<stencilType>& psf, int niter, bool downsample_stencil, bool normalize_stencil);

template<typename inputType, typename stencilType>
void richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input, stencilType* output, PixelData<stencilType>& psf,
                     int niter, bool downsample_stencil, bool normalize_stencil);

template<typename inputType, typename stencilType>
void richardson_lucy_pixel(inputType* input, stencilType* output, stencilType* psf, stencilType* psf_flipped, int kernel_size, int npixels, int niter, std::vector<int>& dims);


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
                                     const int level,
                                     const int* offset_ind,
                                     const stencilType pad_value = 0);


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
                                          const int* offset_ind,
                                          const stencilType pad_value = 0);


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
                                     const int level,
                                     const stencilType pad_value = 0);


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
                                          const stencilType pad_value = 0);


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
                                     const int* offset_ind,
                                     const stencilType pad_value = 0);

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
                                          const int* offset_ind,
                                          const stencilType pad_value = 0);

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
                                     const stencilType pad_value = 0);

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
                                          const stencilType pad_value = 0);

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333_chunked(const inputType* input_image,
                                       outputType* output_image,
                                       const stencilType* stencil,
                                       const int z_num,
                                       const int x_num,
                                       const int y_num);


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_555_chunked(const inputType* input_image,
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

template<typename T>
__global__ void elementWiseMult(T* in1,
                                const T* in2,
                                const size_t size);

template<typename T, typename S>
__global__ void elementWiseDiv(const T* numerator,
                               const S* denominator,
                               S* out,
                               const size_t size);

template<typename T>
__global__ void copyKernel(const T* in,
                           T* out,
                           const size_t size);

template<typename T>
__global__ void fillWithValue(T* in, T value, const size_t size);

__global__ void print_value(const float* data, const size_t index);

template<unsigned int blockSize, typename T>
__global__ void compute_average(T* data, T* result, const size_t size);

#endif //LIBAPR_APRISOCONVGPU_HPP
