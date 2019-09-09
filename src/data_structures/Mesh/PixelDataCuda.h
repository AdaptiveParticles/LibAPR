//
// Created by Krzysztof Gonciarz on 4/9/18.
//

#ifndef LIBAPR_PIXELDATACUDA_H
#define LIBAPR_PIXELDATACUDA_H


#include "PixelData.hpp"

template<typename T, typename S>
void downsampleMeanCuda(const PixelData<T> &aInput, PixelData<S> &aOutput);

template <typename T, typename S>
void downsampleMaxCuda(const PixelData<T> &input, PixelData<S> &output);

template <typename T>
void padCuda(const PixelData<T> &input, PixelData<T> &output, const PixelDataDim &padSize);


#endif //LIBAPR_PIXELDATACUDA_H
