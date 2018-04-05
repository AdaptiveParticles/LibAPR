//
// Created by Krzysztof Gonciarz on 3/27/18.
//

#ifndef LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
#define LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H

#include "data_structures/Mesh/MeshData.hpp"

using TypeOfFlags = uint16_t;
constexpr TypeOfFlags BSPLINE_Y_DIR = 0x01;
constexpr TypeOfFlags BSPLINE_X_DIR = 0x02;
constexpr TypeOfFlags BSPLINE_Z_DIR = 0x04;
constexpr TypeOfFlags BSPLINE_ALL_DIR = BSPLINE_Y_DIR | BSPLINE_X_DIR | BSPLINE_Z_DIR;

template <typename T>
void cudaFilterBsplineFull(MeshData<T> &input, float lambda, float tolerance, TypeOfFlags flags = BSPLINE_ALL_DIR);


#endif //LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
