//
// Created by Krzysztof Gonciarz on 4/11/18.
//

#ifndef LIBAPR_LOCALINTENSITYSCALECUDA_H
#define LIBAPR_LOCALINTENSITYSCALECUDA_H


#include "data_structures/Mesh/MeshData.hpp"


using TypeOfMeanFlags = uint16_t;
constexpr TypeOfMeanFlags MEAN_Y_DIR = 0x01;
constexpr TypeOfMeanFlags MEAN_X_DIR = 0x02;
constexpr TypeOfMeanFlags MEAN_Z_DIR = 0x04;
constexpr TypeOfMeanFlags MEAN_ALL_DIR = MEAN_Y_DIR | MEAN_X_DIR | MEAN_Z_DIR;

template <typename T>
void calcMean(MeshData<T> &image, int offset, TypeOfMeanFlags flags = MEAN_ALL_DIR);


#endif //LIBAPR_LOCALINTENSITYSCALECUDA_H
