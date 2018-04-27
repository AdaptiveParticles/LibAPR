//
// Created by Krzysztof Gonciarz on 3/8/18.
//

#ifndef LIBAPR_COMPUTEGRADIENTCUDA_HPP
#define LIBAPR_COMPUTEGRADIENTCUDA_HPP

#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/APRParameters.hpp"

void cudaDownsampledGradient(const MeshData<float> &input, MeshData<float> &grad, const float hx, const float hy, const float hz);

template <typename T>
void thresholdGradient(MeshData<float> &output, const MeshData<T> &input, const float Ip_th);

template <typename T>
void thresholdImg(MeshData<T> &image, const float threshold);

template <typename ImgType>
void getGradient(MeshData<ImgType> &image, MeshData<ImgType> &grad_temp, MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par);

template <typename ImgType>
void getFullPipeline(MeshData<ImgType> &image, MeshData<ImgType> &grad_temp, MeshData<float> &local_scale_temp, MeshData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par);


#endif //LIBAPR_COMPUTEGRADIENTCUDA_HPP
