//
// Created by Krzysztof Gonciarz on 3/8/18.
//

#ifndef LIBAPR_COMPUTEGRADIENTCUDA_HPP
#define LIBAPR_COMPUTEGRADIENTCUDA_HPP

#include "data_structures/Mesh/MeshData.hpp"


void cudaDownsampledGradient(const MeshData<float> &input, MeshData<float> &grad, const float hx, const float hy, const float hz);
template <typename T>
void cudaFilterBsplineYdirection(MeshData<T> &input, float lambda, float tolerance);
template <typename T>
void cudaFilterBsplineXdirection(MeshData<T> &input, float lambda, float tolerance);
template <typename T>
void cudaFilterBsplineZdirection(MeshData<T> &input, float lambda, float tolerance);

#endif //LIBAPR_COMPUTEGRADIENTCUDA_HPP
