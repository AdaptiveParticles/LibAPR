//
// Created by Krzysztof Gonciarz on 3/8/18.
//

#ifndef LIBAPR_COMPUTEGRADIENTCUDA_HPP
#define LIBAPR_COMPUTEGRADIENTCUDA_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRParameters.hpp"

void cudaDownsampledGradient(const PixelData<float> &input, PixelData<float> &grad, const float hx, const float hy, const float hz);

template <typename T>
void thresholdGradient(PixelData<float> &output, const PixelData<T> &input, const float Ip_th);

template <typename T>
void thresholdImg(PixelData<T> &image, const float threshold);

template <typename ImgType>
void getGradient(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par);

template <typename ImgType>
void getFullPipeline(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par, int maxLevel);


#endif //LIBAPR_COMPUTEGRADIENTCUDA_HPP
