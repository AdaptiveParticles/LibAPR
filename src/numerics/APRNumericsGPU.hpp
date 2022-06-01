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


    /**
     * Compute the gradient in a given dimension using level-adaptive Sobel filters (smoothing perpendicular to the
     * gradient dimension, followed by central finite differences). Combines the operations into a dense 3x3x3 convolution.
     * @tparam InputType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param dimension             dimension along which the gradient is computed (0: y, 1: x, 2: z)
     * @param delta                 pixel size used to scale the gradient (default: 1)
     */
    template<typename InputType>
    void gradient_sobel(GPUAccessHelper &access,
                        GPUAccessHelper &tree_access,
                        VectorData<InputType>& inputParticles,
                        VectorData<float>& outputParticles,
                        int dimension,
                        float delta = 1.0f);


    /**
     * Apply 3x3x3 convolution using three input stencils and compute the magnitude sqrt(dz*dz + dy*dy + dx*dx)
     * of the results.
     * @tparam InputType
     * @param access
     * @param tree_access
     * @param inputParticles
     * @param outputParticles
     * @param stencil_vec_y     Stencil (vectors) to be applied. Should be of size 27 or `27 * (access.level_max() - access.level_min())`
     * @param stencil_vec_x
     * @param stencil_vec_z
     */
    template<typename InputType>
    void gradient_magnitude(GPUAccessHelper &access, GPUAccessHelper &tree_access, VectorData<InputType> &inputParticles,
                            VectorData<float> &outputParticles, VectorData<float> &stencil_vec_y,
                            VectorData<float> &stencil_vec_x, VectorData<float> &stencil_vec_z);


    /**
     * Compute the gradient magnitude using level-adaptive central finite differences.
     * Note: uses 3x3x3 convolutions instead of e.g. 1x1x3, which is quite inefficient.
     * @tparam InputType
     * @tparam GradientType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param deltas                pixel size in each dimension, used to scale the gradients (default: {1, 1, 1})
     */
    template<typename InputType>
    void gradient_magnitude_cfd(GPUAccessHelper &access,
                                GPUAccessHelper &tree_access,
                                VectorData<InputType> &inputParticles,
                                VectorData<float> &outputParticles,
                                const std::vector<float> &deltas = {1.0f, 1.0f, 1.0f});


    /**
     * Compute the gradient magnitude using level-adaptive Sobel filters.
     * @tparam InputType
     * @tparam GradientType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param deltas                pixel size in each dimension, used to scale the gradients (default: {1, 1, 1})
     */
    template<typename InputType>
    void gradient_magnitude_sobel(GPUAccessHelper &access,
                                  GPUAccessHelper &tree_access,
                                  VectorData<InputType>& inputParticles,
                                  VectorData<float>& outputParticles,
                                  const std::vector<float>& deltas = {1.0f, 1.0f, 1.0f});


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


template<typename InputType>
void APRNumericsGPU::gradient_sobel(GPUAccessHelper &access, GPUAccessHelper &tree_access,
                                    VectorData<InputType> &inputParticles, VectorData<float> &outputParticles,
                                    int dimension, float delta) {

    auto stencil = APRStencil::create_sobel_filter<float>(dimension, delta);

    VectorData<float> stencil_vec;
    APRStencil::get_rescaled_stencils(stencil, stencil_vec, access.level_max()-access.level_min());

    VectorData<float> tree_data;
    isotropic_convolve_333(access, tree_access, inputParticles, outputParticles, stencil_vec, tree_data, true);
}


template<typename InputType>
void APRNumericsGPU::gradient_magnitude_cfd(GPUAccessHelper &access, GPUAccessHelper &tree_access,
                                            VectorData<InputType> &inputParticles, VectorData<float> &outputParticles,
                                            const std::vector<float> &deltas) {

    // generate cfd stencils
    PixelData<float> stencil_y(3, 3, 3, 0);
    PixelData<float> stencil_x(3, 3, 3, 0);
    PixelData<float> stencil_z(3, 3, 3, 0);

    stencil_y.at(0, 1, 1) = -1.f/(2*deltas[0]); stencil_y.at(2, 1, 1) = 1.f/(2*deltas[0]);
    stencil_x.at(1, 0, 1) = -1.f/(2*deltas[1]); stencil_x.at(1, 2, 1) = 1.f/(2*deltas[1]);
    stencil_z.at(1, 1, 0) = -1.f/(2*deltas[2]); stencil_z.at(1, 1, 2) = 1.f/(2*deltas[2]);

    // rescale stencils for each level
    VectorData<float> stencil_vec_y, stencil_vec_x, stencil_vec_z;
    APRStencil::get_rescaled_stencils(stencil_y, stencil_vec_y, access.level_max()-access.level_min());
    APRStencil::get_rescaled_stencils(stencil_x, stencil_vec_x, access.level_max()-access.level_min());
    APRStencil::get_rescaled_stencils(stencil_z, stencil_vec_z, access.level_max()-access.level_min());

    // compute gradient magnitude
    gradient_magnitude(access, tree_access, inputParticles, outputParticles, stencil_vec_y, stencil_vec_x, stencil_vec_z);
}


template<typename InputType>
void APRNumericsGPU::gradient_magnitude_sobel(GPUAccessHelper &access, GPUAccessHelper &tree_access,
                                              VectorData<InputType> &inputParticles, VectorData<float> &outputParticles,
                                              const std::vector<float> &deltas) {
    // generate Sobel stencils
    auto stencil_y = APRStencil::create_sobel_filter<float>(0, deltas[0]);
    auto stencil_x = APRStencil::create_sobel_filter<float>(1, deltas[1]);
    auto stencil_z = APRStencil::create_sobel_filter<float>(2, deltas[2]);

    // rescale stencils for each level
    VectorData<float> stencil_vec_y, stencil_vec_x, stencil_vec_z;
    APRStencil::get_rescaled_stencils(stencil_y, stencil_vec_y, access.level_max()-access.level_min());
    APRStencil::get_rescaled_stencils(stencil_x, stencil_vec_x, access.level_max()-access.level_min());
    APRStencil::get_rescaled_stencils(stencil_z, stencil_vec_z, access.level_max()-access.level_min());

    // compute gradient magnitude
    gradient_magnitude(access, tree_access, inputParticles, outputParticles, stencil_vec_y, stencil_vec_x, stencil_vec_z);
}




#endif //LIBAPR_APRNUMERICSGPU_HPP
