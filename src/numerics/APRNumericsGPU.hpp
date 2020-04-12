//
// Created by joel on 08.04.20.
//

#ifndef LIBAPR_APRNUMERICSGPU_HPP
#define LIBAPR_APRNUMERICSGPU_HPP

#include "APRIsoConvGPU555.hpp"

/// including data transfers
template<typename inputType, typename stencilType>
void richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                     VectorData<stencilType>& output, PixelData<stencilType>& psf, int niter, bool use_stencil_downsample = true, bool normalize_stencil = false);

/// for data already on the gpu
template<typename inputType, typename stencilType>
void richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input, stencilType* output,
                     stencilType* psf, stencilType* psf_flipped, int kernel_size, int niter, bool use_stencil_downsample);

#endif //LIBAPR_APRNUMERICSGPU_HPP
