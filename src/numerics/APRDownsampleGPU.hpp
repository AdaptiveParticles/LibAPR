//
// Created by Joel Jonsson on 2019-07-11.
//

#ifndef LIBAPR_APRDOWNSAMPLEGPU_HPP
#define LIBAPR_APRDOWNSAMPLEGPU_HPP

#include "miscCuda.hpp"


template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu);

template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<treeType>& tree_data);

template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu, int* ne_rows, VectorData<int>& ne_offset);

template<typename inputType, typename treeType>
void downsample_avg_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu);

template<typename inputType, typename treeType>
void downsample_avg_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<treeType>& tree_data);

void compute_ne_rows_tree(GPUAccessHelper& tree_access, VectorData<int>& ne_counter, VectorData<int>& ne_rows);

template<int blockSize_z, int blockSize_x>
void compute_ne_rows_tree_cuda(GPUAccessHelper& tree_access, VectorData<int>& ne_count, ScopedCudaMemHandler<int*, JUST_ALLOC>& ne_rows_gpu);


#endif //LIBAPR_APRDOWNSAMPLEGPU_HPP
