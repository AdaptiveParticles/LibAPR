//
// Created by Joel Jonsson on 2019-07-11.
//

#ifndef LIBAPR_APRDOWNSAMPLEGPU_HPP
#define LIBAPR_APRDOWNSAMPLEGPU_HPP

#include "data_structures/APR/access/GPUAccess.hpp"

//template<typename T, typename S>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<uint16_t>& input, std::vector<uint16_t>& tree_data);

#endif //LIBAPR_APRDOWNSAMPLEGPU_HPP
