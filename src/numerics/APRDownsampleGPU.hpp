//
// Created by Joel Jonsson on 2019-07-11.
//

#ifndef LIBAPR_APRDOWNSAMPLEGPU_HPP
#define LIBAPR_APRDOWNSAMPLEGPU_HPP

#include "data_structures/APR/access/GPUAccess.hpp"
#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"


template<typename inputType, typename treeType>
void downsample_avg_init_wrapper(GPUAccessHelper&, GPUAccessHelper&, std::vector<inputType>&, std::vector<treeType>&);


template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                    std::vector<treeType>& tree_data) {

    downsample_avg_init_wrapper(access, tree_access, input, tree_data);
}


template<typename inputType, typename treeType>
void downsample_avg_wrapper(GPUAccessHelper&, GPUAccessHelper&, inputType*, treeType*);


template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu) {

    downsample_avg_wrapper(access, tree_access, input_gpu, tree_data_gpu);
}


/// force instantiation for some different type combinations
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<uint16_t>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, std::vector<float>&, std::vector<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, std::vector<float>&, std::vector<double>&);

template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, uint16_t*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, float*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, double*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, float*, float*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, float*, double*);


template<typename inputType, typename outputType>
__global__ void down_sample_avg(const uint64_t* level_xz_vec,
                                const uint64_t* xz_end_vec,
                                const uint16_t* y_vec,
                                const inputType* input_particles,
                                const uint64_t* level_xz_vec_tree,
                                const uint64_t* xz_end_vec_tree,
                                const uint16_t* y_vec_tree,
                                outputType* particle_data_output,
                                const int z_num,
                                const int x_num,
                                const int y_num,
                                const int z_num_parent,
                                const int x_num_parent,
                                const int y_num_parent,
                                const int level);


template<typename inputType, typename outputType>
__global__ void down_sample_avg_interior(const uint64_t* level_xz_vec,
                                         const uint64_t* xz_end_vec,
                                         const uint16_t* y_vec,
                                         const inputType* input_particles,
                                         const uint64_t* level_xz_vec_tree,
                                         const uint64_t* xz_end_vec_tree,
                                         const uint16_t* y_vec_tree,
                                         outputType* particle_data_output,
                                         const int z_num,
                                         const int x_num,
                                         const int y_num,
                                         const int z_num_parent,
                                         const int x_num_parent,
                                         const int y_num_parent,
                                         const int level);


#endif //LIBAPR_APRDOWNSAMPLEGPU_HPP
