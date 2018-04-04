//
// Created by Krzysztof Gonciarz on 3/27/18.
//

#ifndef LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
#define LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H

#include "data_structures/Mesh/MeshData.hpp"


constexpr uint16_t BSPLINE_Y_DIR = 0x01;
constexpr uint16_t BSPLINE_X_DIR = 0x02;
constexpr uint16_t BSPLINE_Z_DIR = 0x04;
constexpr uint16_t BSPLINE_ALL_DIR = BSPLINE_Y_DIR | BSPLINE_X_DIR | BSPLINE_Z_DIR;
template <typename T>
void cudaFilterBsplineFull(MeshData<T> &input, float lambda, float tolerance, uint16_t flags = BSPLINE_ALL_DIR);


#endif //LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
