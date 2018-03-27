//
// Created by Krzysztof Gonciarz on 3/8/18.
//

#ifndef LIBAPR_COMPUTEGRADIENTCUDA_HPP
#define LIBAPR_COMPUTEGRADIENTCUDA_HPP

#include "data_structures/Mesh/MeshData.hpp"


void cudaDownsampledGradient(const MeshData<float> &input, MeshData<float> &grad, const float hx, const float hy, const float hz);


#endif //LIBAPR_COMPUTEGRADIENTCUDA_HPP
