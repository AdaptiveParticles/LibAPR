//
// Created by Krzysztof Gonciarz on 3/27/18.
//

#ifndef LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
#define LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H

#include "data_structures/Mesh/MeshData.hpp"


template <typename T>
void cudaFilterBsplineYdirection(MeshData<T> &input, float lambda, float tolerance);
template <typename T>
void cudaFilterBsplineXdirection(MeshData<T> &input, float lambda, float tolerance);
template <typename T>
void cudaFilterBsplineZdirection(MeshData<T> &input, float lambda, float tolerance);


#endif //LIBAPR_COMPUTEBSPLINERECURSIVEFILTERCUDA_H
