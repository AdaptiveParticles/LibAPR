//
// Created by Joel Jonsson on 2019-07-11.
//

#ifndef LIBAPR_APRDOWNSAMPLEGPU_HPP
#define LIBAPR_APRDOWNSAMPLEGPU_HPP

#include "data_structures/APR/access/GPUAccess.hpp"
#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"
#include "data_structures/Mesh/PixelData.hpp"


template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu);

template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<treeType>& tree_data);

template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu, int* ne_rows, VectorData<int>& ne_offset);

void compute_ne_rows(GPUAccessHelper& tree_access,VectorData<int>& ne_counter,VectorData<int>& ne_rows);


#endif //LIBAPR_APRDOWNSAMPLEGPU_HPP
