//
// Created by joel on 08.04.20.
//

#ifndef LIBAPR_APRNUMERICSGPU_HPP
#define LIBAPR_APRNUMERICSGPU_HPP

#include "APRStencil.hpp"
#include "miscCuda.hpp"
#include "APRIsoConvGPU555.hpp"

namespace APRNumericsGPU {

    /**
     * Compute the gradient in a given dimension using level-adaptive central finite differences.
     * Note: uses 3x3x3 convolution instead of 1x1x3, which is quite inefficient
     * @tparam InputType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param dimension         dimension along which the gradient is computed (0: y, 1: x, 2: z)
     * @param delta             pixel size used to scale the gradient (default 1.0f)
     */
    template<typename InputType>
    void gradient_cfd(GPUAccessHelper &access,
                      GPUAccessHelper &tree_access,
                      VectorData<InputType>& inputParticles,
                      VectorData<float>& outputParticles,
                      int dimension,
                      float delta = 1.0f);


    template<typename inputType, typename stencilType>
    void richardson_lucy(GPUAccessHelper &access, GPUAccessHelper &tree_access, VectorData<inputType> &input,
                         VectorData<stencilType> &output, PixelData<stencilType> &psf, int niter,
                         bool use_stencil_downsample = true, bool normalize_stencil = false, bool resume = false);

    /// for data already on the gpu
    template<typename inputType, typename stencilType>
    void richardson_lucy(GPUAccessHelper &access, GPUAccessHelper &tree_access, inputType *input, stencilType *output,
                         stencilType *psf, stencilType *psf_flipped, int kernel_size, int niter,
                         bool use_stencil_downsample, bool resume = false);
}



template<typename InputType>
void APRNumericsGPU::gradient_cfd(GPUAccessHelper &access, GPUAccessHelper &tree_access, VectorData<InputType> &inputParticles,
                                  VectorData<float> &outputParticles, const int dimension, const float delta) {

    PixelData<float> stencil(3, 3, 3, 0);
    stencil.at(dimension == 0 ? 0 : 1, dimension == 1 ? 0 : 1, dimension == 2 ? 0 : 1) = -1.f / (2.f * delta);
    stencil.at(dimension == 0 ? 2 : 1, dimension == 1 ? 2 : 1, dimension == 2 ? 2 : 1) = 1.f / (2.f * delta);

    VectorData<float> stencil_vec;
    APRStencil::get_rescaled_stencils(stencil, stencil_vec, access.level_max()-access.level_min());

    VectorData<float> tree_data;
    isotropic_convolve_333(access, tree_access, inputParticles, outputParticles, stencil_vec, tree_data, true);
}


#endif //LIBAPR_APRNUMERICSGPU_HPP
