//
// Created by Krzysztof Gonciarz on 4/11/18.
//

#ifndef LIBAPR_LOCALINTENSITYSCALECUDA_H
#define LIBAPR_LOCALINTENSITYSCALECUDA_H


#include "data_structures/Mesh/MeshData.hpp"


template <typename T>
void calcMeanYdir(MeshData<T> &image, int offset);
template <typename T>
void calcMeanXdir(MeshData<T> &image, int offset);
template <typename T>
void calcMeanZdir(MeshData<T> &image, int offset);

#endif //LIBAPR_LOCALINTENSITYSCALECUDA_H
