#ifndef LIBAPR_PIXELDATACUDA_H
#define LIBAPR_PIXELDATACUDA_H


#include "PixelData.hpp"


template<typename T, typename S>
void downsampleMeanCuda(const PixelData<T> &aInput, PixelData<S> &aOutput);

template <typename T, typename S>
void downsampleMaxCuda(const PixelData<T> &input, PixelData<S> &output);

/**
 * Copies data from input to output (which is bigger by pad size) reflecting around the edge pixels.
 * @tparam T
 * @param input
 * @param output
 * @param padSize
 */
template <typename T>
void paddPixelsCuda(const PixelData<T> &input, PixelData<T> &output, const PixelDataDim &padSize);

/**
 * Copies data from input to output (which is smaller by pad size).
 * @tparam T
 * @param input
 * @param output
 * @param padSize
 */
template <typename T>
void unpaddPixelsCuda(const PixelData<T> &input, PixelData<T> &output, const PixelDataDim &padSize);

#endif

