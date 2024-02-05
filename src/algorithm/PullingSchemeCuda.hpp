//
// Created by gonciarz on 10/18/18.
//

#ifndef LIBAPR_PULLINGSCHEMECUDA_HPP
#define LIBAPR_PULLINGSCHEMECUDA_HPP


#include "data_structures/Mesh/PixelData.hpp"

using TreeElementType = uint8_t;

template <typename T, typename S>
void computeOVPC(const PixelData<T> &input, PixelData<S> &output, int levelMin, int levelMax);
template <typename T, typename S>
void computeOvpcCuda(const PixelData<T> &input, std::vector<PixelData<S>> &pct, int levelMin, int levelMax);

#endif //LIBAPR_PULLINGSCHEMECUDA_HPP
