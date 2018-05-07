//
// Created by Krzysztof Gonciarz on 3/27/18.
//

#ifndef LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
#define LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H

#include "data_structures/Mesh/MeshData.hpp"

using TypeOfRecBsplineFlags = uint16_t;
constexpr TypeOfRecBsplineFlags BSPLINE_Y_DIR = 0x01;
constexpr TypeOfRecBsplineFlags BSPLINE_X_DIR = 0x02;
constexpr TypeOfRecBsplineFlags BSPLINE_Z_DIR = 0x04;
constexpr TypeOfRecBsplineFlags BSPLINE_ALL_DIR = BSPLINE_Y_DIR | BSPLINE_X_DIR | BSPLINE_Z_DIR;

template <typename T>
void cudaFilterBsplineFull(MeshData<T> &input, float lambda, float tolerance, TypeOfRecBsplineFlags flags = BSPLINE_ALL_DIR, int k0 = -1);


#endif //LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
