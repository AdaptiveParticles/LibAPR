//
// Created by Krzysztof Gonciarz on 4/9/18.
//

#ifndef LIBAPR_MESHDATACUDA_H
#define LIBAPR_MESHDATACUDA_H


#include "MeshData.hpp"

template<typename T, typename S>
void downsampleMeanCuda(const MeshData<T> &aInput, MeshData<S> &aOutput);

template <typename T, typename S>
void downsampleMaxCuda(const MeshData<T> &input, MeshData<S> &output);

#endif //LIBAPR_MESHDATACUDA_H
