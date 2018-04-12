//
// Created by Krzysztof Gonciarz on 4/5/18.
//

#ifndef LIBAPR_COMPUTEINVERSECUBICBSPLINECUDA_H
#define LIBAPR_COMPUTEINVERSECUBICBSPLINECUDA_H

#include "data_structures/Mesh/MeshData.hpp"

using TypeOfInvBsplineFlags = uint16_t;
constexpr TypeOfInvBsplineFlags INV_BSPLINE_Y_DIR = 0x01;
constexpr TypeOfInvBsplineFlags INV_BSPLINE_X_DIR = 0x02;
constexpr TypeOfInvBsplineFlags INV_BSPLINE_Z_DIR = 0x04;
constexpr TypeOfInvBsplineFlags INV_BSPLINE_ALL_DIR = INV_BSPLINE_Y_DIR | INV_BSPLINE_X_DIR | INV_BSPLINE_Z_DIR;


template <typename ImgType>
void cudaInverseBspline(MeshData<ImgType> &input, TypeOfInvBsplineFlags flags = INV_BSPLINE_ALL_DIR);

#endif //LIBAPR_COMPUTEINVERSECUBICBSPLINECUDA_H
