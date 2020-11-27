//
// Created by joeljonsson on 23.11.2020
//

#ifndef LIBAPR_APRNUMERICS_HPP
#define LIBAPR_APRNUMERICS_HPP

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "numerics/APRFilter.hpp"
#include "numerics/APRStencil.hpp"
#include "numerics/APRTreeNumerics.hpp"


namespace APRNumerics {

    /**
     * Compute the gradient in a given dimension using level-adaptive central finite differences
     * @tparam InputType
     * @tparam GradientType     must be floating point
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param dimension         dimension along which the gradient is computed (0: y, 1: x, 2: z)
     * @param delta             pixel size used to scale the gradient (default 1.0f)
     */
    template<typename InputType, typename GradientType,
            std::enable_if_t<std::is_floating_point<GradientType>::value, bool> = true>
    void gradient_cfd(APR& apr,
                      const ParticleData<InputType>& inputParticles,
                      ParticleData<GradientType>& outputParticles,
                      int dimension,
                      float delta = 1.0f);


    /**
     * Compute the gradient in a given dimension using level-adaptive Sobel filters (smoothing perpendicular to the
     * gradient dimension, followed by central finite differences). Combines the operations into a dense 3x3x3 convolution.
     * @tparam InputType
     * @tparam GradientType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param dimension             dimension along which the gradient is computed (0: y, 1: x, 2: z)
     * @param delta                 pixel size used to scale the gradient (default: 1)
     */
    template<typename InputType, typename GradientType,
            std::enable_if_t<std::is_floating_point<GradientType>::value, bool> = true>
    void gradient_sobel(APR& apr, const ParticleData<InputType>& inputParticles, ParticleData<GradientType>& outputParticles,
                        int dimension, float delta = 1.0f);


    /**
     * Compute the gradient magnitude using APRNumerics::gradient_cfd in each dimension.
     * @tparam InputType
     * @tparam GradientType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param deltas                pixel size in each dimension, used to scale the gradients (default: {1, 1, 1})
     */
    template<typename InputType, typename GradientType,
            std::enable_if_t<std::is_floating_point<GradientType>::value, bool> = true>
    void gradient_magnitude_cfd(APR& apr, const ParticleData<InputType>& inputParticles, ParticleData<GradientType>& outputParticles,
                                const std::vector<float>& deltas = {1.0f, 1.0f, 1.0f});


    /**
     * Compute the gradient magnitude using APRNumerics::gradient_sobel in each dimension.
     * @tparam InputType
     * @tparam GradientType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param deltas                pixel size in each dimension, used to scale the gradients (default: {1, 1, 1})
     */
    template<typename InputType, typename GradientType,
            std::enable_if_t<std::is_floating_point<GradientType>::value, bool> = true>
    void gradient_magnitude_sobel(APR& apr, const ParticleData<InputType>& inputParticles, ParticleData<GradientType>& outputParticles,
                                  const std::vector<float>& deltas = {1.0f, 1.0f, 1.0f});


    /**
     * Computes the local standard deviation in a given window around each particle. At coarser resolution particles,
     * the window is rescaled and weighted
     * @tparam InputType
     * @tparam OutputType
     * @param apr
     * @param inputParticles
     * @param outputParticles
     * @param size                  size of the window in each dimension
     */
    template<typename InputType, typename OutputType,
            std::enable_if_t<std::is_floating_point<OutputType>::value, bool> = true>
    void local_std(APR& apr, const ParticleData<InputType>& inputParticles, ParticleData<OutputType>& outputParticles,
                   const std::vector<int>& size = {3, 3, 3});


    /**
     * Apply a filter to each particle and its face-side neighbours in a given dimension.
     */
    template<typename S,typename U>
    void face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                               const std::vector<float>& filter, int dimension);

    /**
     * Successively apply a filter to each particle and its face-side neighbours in each dimension (y -> x -> z)
     */
    template<typename S,typename U>
    void seperable_face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                                         const std::vector<float>& filter, int repeats = 1);


    template<typename InputType, typename OutputType>
    void adaptive_min(APR& apr, const ParticleData<InputType>& input_data, ParticleData<OutputType>& loc_min,
                      int num_tree_smooth=3, int level_delta=1, int num_part_smooth=2);


    template<typename InputType, typename OutputType>
    void adaptive_max(APR& apr, const ParticleData<InputType>& input_data, ParticleData<OutputType>& loc_max,
                      int num_tree_smooth=3, int level_delta=1, int num_part_smooth=2);


    template<typename InputType, typename StencilType, typename OutputType,
            std::enable_if_t<std::is_floating_point<StencilType>::value, bool> = true>
    void richardson_lucy(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                         std::vector<PixelData<StencilType>>& psf_vec, std::vector<PixelData<StencilType>>& psf_flipped_vec,
                         int number_iterations, bool resume=false);

    template<typename InputType, typename StencilType, typename OutputType,
            std::enable_if_t<std::is_floating_point<StencilType>::value, bool> = true>
    void richardson_lucy(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                         PixelData<StencilType> &psf, int number_iterations, bool use_stencil_downsample=true,
                         bool normalize=false, bool resume=false);

    template<typename InputType, typename StencilType, typename OutputType,
            std::enable_if_t<std::is_floating_point<StencilType>::value, bool> = true>
    void richardson_lucy_tv(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                            std::vector<PixelData<StencilType>>& psf_vec, std::vector<PixelData<StencilType>>& psf_flipped_vec,
                            int number_iterations, float reg_factor, bool resume);

    template<typename InputType, typename StencilType, typename OutputType,
            std::enable_if_t<std::is_floating_point<StencilType>::value, bool> = true>
    void richardson_lucy_tv(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                            PixelData<StencilType> &psf, int number_iterations, float reg_factor, bool use_stencil_downsample,
                            bool normalize, bool resume);

    /**
     * Computes the divergence of the normalized gradient using level-adaptive central finite differences.
     */
    template<typename InputType, typename GradientType,
            std::enable_if_t<std::is_floating_point<GradientType>::value, bool> = true>
    void div_norm_grad(APR &apr,
                       const ParticleData<InputType> &input,
                       ParticleData<GradientType> &grad_x,
                       ParticleData<GradientType> &grad_y,
                       ParticleData<GradientType> &grad_z,
                       ParticleData<GradientType> &result,
                       const std::vector<float>& deltas = {1.0f, 1.0f, 1.0f});
}


template<typename InputType, typename GradientType,
        std::enable_if_t<std::is_floating_point<GradientType>::value, bool>>
void APRNumerics::gradient_cfd(APR& apr,
                               const ParticleData<InputType>& inputParticles,
                               ParticleData<GradientType>& outputParticles,
                               const int dimension,
                               const float delta) {

    if (dimension < 0 || dimension > 2) {
        throw std::invalid_argument("APRNumerics::gradient_cfd argument 'dimension' must be 0 (y), 1 (x) or 2 (z)");
    }

    PixelData<GradientType> stencil((dimension == 0) ? 3 : 1, (dimension == 1) ? 3 : 1, (dimension == 2) ? 3 : 1);
    stencil.mesh[0] = -1.0f/(2*delta);
    stencil.mesh[1] = 0;
    stencil.mesh[2] = 1.0f/(2*delta);

    std::vector<PixelData<GradientType>> stencil_vec;
    APRStencil::get_rescaled_stencils(stencil, stencil_vec, apr.level_max() - apr.level_min());

    APRFilter::convolve_pencil(apr, stencil_vec, inputParticles, outputParticles, true);
}


template<typename InputType, typename GradientType,
        std::enable_if_t<std::is_floating_point<GradientType>::value, bool>>
void APRNumerics::gradient_sobel(APR& apr,
                                 const ParticleData<InputType>& inputParticles,
                                 ParticleData<GradientType>& outputParticles,
                                 const int dimension,
                                 const float delta) {

    if (dimension < 0 || dimension > 2) {
        throw std::invalid_argument("APRNumerics::gradient_sobel argument 'dimension' must be 0 (y), 1 (x) or 2 (z)");
    }

    auto stencil = APRStencil::create_sobel_filter<GradientType>(dimension, delta);
    std::vector<PixelData<GradientType>> stencil_vec;
    APRStencil::get_rescaled_stencils(stencil, stencil_vec, apr.level_max() - apr.level_min());

    APRFilter::convolve_pencil(apr, stencil_vec, inputParticles, outputParticles, true);
}


template<typename InputType, typename GradientType,
        std::enable_if_t<std::is_floating_point<GradientType>::value, bool>>
void APRNumerics::gradient_magnitude_cfd(APR& apr,
                                         const ParticleData<InputType>& inputParticles,
                                         ParticleData<GradientType>& outputParticles,
                                         const std::vector<float>& deltas) {

    outputParticles.init(apr.total_number_particles());
    ParticleData<GradientType> tmp;

    // compute y gradient
    gradient_cfd(apr, inputParticles, outputParticles, 0, deltas[0]);

    // square the result
    auto square_h = [](const GradientType& a) -> GradientType { return a*a; };
    outputParticles.unary_map(outputParticles, square_h);

    auto add_square_h = [](const GradientType &a, const GradientType &b) -> GradientType { return a + b*b; };

    if (apr.org_dims(1) > 1) {
        gradient_cfd(apr, inputParticles, tmp, 1, deltas[1]);             // compute x gradient
        outputParticles.binary_map(tmp, outputParticles, add_square_h);   // add squared x-gradient to outputParticles
    }

    if (apr.org_dims(2) > 1) {
        gradient_cfd(apr, inputParticles, tmp, 2, deltas[2]);             // compute z gradient
        outputParticles.binary_map(tmp, outputParticles, add_square_h);   // add squared x-gradient to outputParticles
    }

    // square root
    auto sqrtf_h = [](const GradientType& a) -> GradientType { return sqrtf(a); };
    outputParticles.unary_map(outputParticles, sqrtf_h);
}


template<typename InputType, typename GradientType,
        std::enable_if_t<std::is_floating_point<GradientType>::value, bool>>
void APRNumerics::gradient_magnitude_sobel(APR& apr,
                                           const ParticleData<InputType>& inputParticles,
                                           ParticleData<GradientType>& outputParticles,
                                           const std::vector<float>& deltas) {

    outputParticles.init(apr.total_number_particles());
    ParticleData<GradientType> tmp;

    // compute y gradient
    gradient_sobel(apr, inputParticles, outputParticles, 0, deltas[0]);

    // square the result
    auto square_h = [](const GradientType& a) -> GradientType { return a*a; };
    outputParticles.unary_map(outputParticles, square_h);

    auto add_square_h = [](const GradientType &a, const GradientType &b) -> GradientType { return a + b*b; };

    if (apr.org_dims(1) > 1) {
        gradient_sobel(apr, inputParticles, tmp, 1, deltas[1]);           // compute x gradient
        outputParticles.binary_map(tmp, outputParticles, add_square_h);   // add squared x-gradient to outputParticles
    }

    if (apr.org_dims(2) > 1) {
        gradient_sobel(apr, inputParticles, tmp, 2, deltas[2]);           // compute z gradient
        outputParticles.binary_map(tmp, outputParticles, add_square_h);   // add squared gradient to outputParticles
    }

    // square root
    auto sqrtf_h = [](const GradientType& a) -> GradientType { return sqrtf(a); };
    outputParticles.unary_map(outputParticles, sqrtf_h);
}


template<typename S,typename U>
void APRNumerics::seperable_face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                                                  const std::vector<float>& filter, const int repeats) {

    output_data.init(apr.total_number_particles());

    ParticleData<U> tmp;
    tmp.copy(input_data);

    for(int i = 0; i < repeats; ++i) {
        face_neighbour_filter(apr, tmp, output_data, filter, 0);
        face_neighbour_filter(apr, output_data, tmp, filter, 1);
        face_neighbour_filter(apr, tmp, output_data, filter, 2);
        output_data.swap(tmp);
    }
    output_data.swap(tmp);
}


template<typename S,typename U>
void APRNumerics::face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                                        const std::vector<float>& filter, const int dimension) {

    output_data.init(apr.total_number_particles());

    int faces[2] = {2*dimension, 2*dimension+1};

    auto apr_iterator = apr.random_iterator();
    auto neighbour_iterator = apr.random_iterator();

    const std::vector<float> filter_t = {filter[2], filter[0]};

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_iterator, neighbour_iterator)
#endif
        for (int z = 0; z < apr_iterator.z_num(level); z++) {
            for (int x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); apr_iterator++) {

                    float current_intensity = input_data[apr_iterator];
                    output_data[apr_iterator] = current_intensity * filter[1];

                    for (int i = 0; i < 2; ++i) {
                        float intensity_sum = 0;
                        float count_neighbours = 0;
                        const int direction = faces[i];

                        apr_iterator.find_neighbours_in_direction(direction);

                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                intensity_sum += input_data[neighbour_iterator];
                                count_neighbours++;
                            }
                        }
                        if (count_neighbours > 0) {
                            output_data[apr_iterator] += filter_t[i] * intensity_sum / count_neighbours;
                        } else {
                            output_data[apr_iterator] += filter_t[i] * current_intensity;
                        }
                    }
                }
            }
        }
    }
}


template<typename InputType, typename OutputType,
        std::enable_if_t<std::is_floating_point<OutputType>::value, bool>>
void APRNumerics::local_std(APR& apr,
                            const ParticleData<InputType>& inputParticles,
                            ParticleData<OutputType>& outputParticles,
                            const std::vector<int>& size) {

    // box filter
    auto box_dense = APRStencil::create_mean_filter<OutputType>(size);

    ParticleData<OutputType> loc_mean;
    ParticleData<OutputType> input_temp;
    ParticleData<OutputType> tree_data;

    // copy input particles and fill tree by averaging
    input_temp.copy(inputParticles);
    APRTreeNumerics::fill_tree_mean(apr, input_temp, tree_data);

    // compute local means using
    APRFilter::convolve_pencil(apr, box_dense, input_temp, tree_data, loc_mean, true, true, true);

    // square input copy and tree data
    auto square_h = [](const OutputType &a) -> OutputType { return a * a; };
    input_temp.unary_map(input_temp, square_h);
    tree_data.unary_map(tree_data, square_h);

    // compute local means of squared data
    APRFilter::convolve_pencil(apr, box_dense, input_temp, tree_data, outputParticles, true, true, true);

    // compute standard deviation
    auto fun_h = [](const OutputType &a, const OutputType &b) -> OutputType {return sqrtf(std::max(a-b*b, 0.0f));};
    outputParticles.binary_map(loc_mean, outputParticles, fun_h);
}


template<typename InputType, typename OutputType>
void APRNumerics::adaptive_min(APR& apr, const ParticleData<InputType>& input_data, ParticleData<OutputType>& loc_min,
                               const int num_tree_smooth, const int level_delta, const int num_part_smooth) {

    ParticleData<float> tree_data;
    APRTreeNumerics::fill_tree_min(apr, input_data, tree_data);

    ParticleData<float> tree_data_smooth;
    APRTreeNumerics::seperable_face_neighbour_filter(apr, tree_data, tree_data_smooth,
                                                     {0.25f, 0.5f, 0.25f}, num_tree_smooth, level_delta);

    APRTreeNumerics::push_down_tree(apr, tree_data_smooth, level_delta);

    APRTreeNumerics::push_to_leaves(apr, tree_data_smooth, tree_data);

    APRNumerics::seperable_face_neighbour_filter(apr, tree_data, loc_min, {0.25f, 0.5f, 0.25f}, num_part_smooth);
}


template<typename InputType, typename OutputType>
void APRNumerics::adaptive_max(APR& apr, const ParticleData<InputType>& input_data, ParticleData<OutputType>& loc_max,
                               const int num_tree_smooth, const int level_delta, const int num_part_smooth) {

    ParticleData<float> tree_data;
    APRTreeNumerics::fill_tree_max(apr, input_data, tree_data);

    ParticleData<float> tree_data_smooth;
    APRTreeNumerics::seperable_face_neighbour_filter(apr, tree_data, tree_data_smooth,
                                                     {0.25f, 0.5f, 0.25f}, num_tree_smooth, level_delta);

    APRTreeNumerics::push_down_tree(apr, tree_data_smooth, level_delta);

    tree_data.init(apr.total_number_particles());
    APRTreeNumerics::push_to_leaves(apr, tree_data_smooth, tree_data);

    APRNumerics::seperable_face_neighbour_filter(apr, tree_data, loc_max, {0.25f, 0.5f, 0.25f}, num_part_smooth);
}


template<typename InputType, typename GradientType,
        std::enable_if_t<std::is_floating_point<GradientType>::value, bool>>
void APRNumerics::div_norm_grad(APR &apr,
                                const ParticleData<InputType> &input,
                                ParticleData<GradientType> &grad_x,
                                ParticleData<GradientType> &grad_y,
                                ParticleData<GradientType> &grad_z,
                                ParticleData<GradientType> &result,
                                const std::vector<float>& deltas) {

    auto add_h = [](const GradientType& a, const GradientType& b) -> GradientType { return a + b; };

    /// compute gradient in y, x and z directions using level-adaptive central finite differences
    gradient_cfd(apr, input, grad_y, 0, deltas[0]);

    if(apr.org_dims(1) > 1) {
        gradient_cfd(apr, input, grad_x, 1, deltas[1]);
    } else {
        grad_x.init(input.size());  // check if the size is correct. if it is, this should do nothing
        grad_x.set_to_zero();
    }

    if(apr.org_dims(2) > 1) {
        gradient_cfd(apr, input, grad_z, 2, deltas[2]);
    } else {
        grad_z.init(input.size());  // check if the size is correct. if it is, this should do nothing
        grad_z.set_to_zero();
    }

    /// normalize the gradients
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) default(none) shared(grad_x, grad_y, grad_z)
#endif
    for(uint64_t i = 0; i < grad_y.size(); ++i) {
        float gradmag = std::sqrt(grad_z[i] * grad_z[i] + grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]);

        if(gradmag > 1e-6) {
            grad_z[i] /= gradmag;
            grad_x[i] /= gradmag;
            grad_y[i] /= gradmag;
        }
    }

    /// compute divergence
    gradient_cfd(apr, grad_y, result, 0, deltas[0]);        // y-gradient -> result

    if(apr.x_num(apr.level_max()) > 1) {
        gradient_cfd(apr, grad_x, grad_y, 1, deltas[1]);    // x-gradient -> grad_y
        result.binary_map(grad_y, result, add_h);           // add grad_y to result
    }

    if(apr.z_num(apr.level_max()) > 1) {
        gradient_cfd(apr, grad_z, grad_y, 2, deltas[2]);    // z-gradient -> grad_y
        result.binary_map(grad_y, result, add_h);           // add grad_y to result
    }
}


template<typename InputType, typename StencilType, typename OutputType,
        std::enable_if_t<std::is_floating_point<StencilType>::value, bool>>
void APRNumerics::richardson_lucy_tv(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                                   std::vector<PixelData<StencilType>>& psf_vec, std::vector<PixelData<StencilType>>& psf_flipped_vec,
                                   int number_iterations, float reg_factor, bool resume) {

    auto divide_h = [](const StencilType& a, const InputType& b) -> StencilType {return b / a;};

    // if not continuing from previous iterations, initialize output with 1s
    if(!resume) {
        particle_output.init(apr.total_number_particles());
        particle_output.fill(1.0f);
    }

    ParticleData<StencilType> relative_blur(apr.total_number_particles());
    ParticleData<StencilType> error_est(apr.total_number_particles());
    ParticleData<StencilType> tmp1(apr.total_number_particles());
    ParticleData<StencilType> tmp2(apr.total_number_particles());
    ParticleData<StencilType> tmp3(apr.total_number_particles());

    for(int iter = 0; iter < number_iterations; ++iter) {

        APRFilter::convolve(apr, psf_flipped_vec, particle_output, relative_blur);  // re-blur estimate
        relative_blur.binary_map(particle_input, relative_blur, divide_h);          // particle_input / relative_blur
        APRFilter::convolve(apr, psf_vec, relative_blur, error_est);                // correlate ratio
        div_norm_grad(apr, particle_output, tmp1, tmp2, tmp3, relative_blur);       // divergence of normalized gradient

        // update estimate
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) default(none) shared(particle_output, error_est, relative_blur, reg_factor)
#endif
        for(uint64_t i = 0; i < particle_output.data.size(); ++i) {
            particle_output[i] = particle_output[i] * error_est[i] / (1.0f - reg_factor * relative_blur[i]);
        }
    }
}


template<typename InputType, typename StencilType, typename OutputType,
        std::enable_if_t<std::is_floating_point<StencilType>::value, bool>>
void APRNumerics::richardson_lucy_tv(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                                   PixelData<StencilType> &psf, int number_iterations, float reg_factor, bool use_stencil_downsample,
                                   bool normalize, bool resume) {

    PixelData<StencilType> psf_flipped(psf, false);
    for(size_t i = 0; i < psf.mesh.size(); ++i) {
        psf_flipped.mesh[i] = psf.mesh[psf.mesh.size()-1-i];
    }

    std::vector<PixelData<StencilType>> psf_vec;
    std::vector<PixelData<StencilType>> psf_flipped_vec;

    int nstencils = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    APRStencil::get_downsampled_stencils(psf, psf_vec, nstencils, normalize);
    APRStencil::get_downsampled_stencils(psf_flipped, psf_flipped_vec, nstencils, normalize);

    richardson_lucy_tv(apr, particle_input, particle_output, psf_vec, psf_flipped_vec, number_iterations, reg_factor, resume);
}


template<typename InputType, typename StencilType, typename OutputType,
        std::enable_if_t<std::is_floating_point<StencilType>::value, bool>>
void APRNumerics::richardson_lucy(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                                std::vector<PixelData<StencilType>>& psf_vec, std::vector<PixelData<StencilType>>& psf_flipped_vec,
                                int number_iterations, bool resume) {

    auto divide_h = [](const StencilType& a, const InputType& b) -> StencilType {return b/a;};
    auto multiply_h = [](const StencilType& a, const StencilType&b) -> StencilType { return a*b; };

    // if not continuing from previous iterations, initialize output with 1s
    if(!resume) {
        particle_output.init(apr.total_number_particles());
        particle_output.fill(1.0f);
    }
    ParticleData<StencilType> relative_blur(apr.total_number_particles());
    ParticleData<StencilType> error_est(apr.total_number_particles());

    for(int iter = 0; iter < number_iterations; ++iter) {
        APRFilter::convolve_pencil(apr, psf_flipped_vec, particle_output, relative_blur);   // re-blur current estimate
        relative_blur.binary_map(particle_input, relative_blur, divide_h);                  // input / blurred estimate
        APRFilter::convolve_pencil(apr, psf_vec, relative_blur, error_est);                 // correlate ratio
        particle_output.binary_map(error_est, particle_output, multiply_h);                 // update estimate
    }
}

template<typename InputType, typename StencilType,typename OutputType,
        std::enable_if_t<std::is_floating_point<StencilType>::value, bool>>
void APRNumerics::richardson_lucy(APR &apr, ParticleData<InputType> &particle_input, ParticleData<OutputType> &particle_output,
                                  PixelData<StencilType>& psf, int number_iterations, bool use_stencil_downsample, bool normalize,
                                  bool resume) {

    PixelData<StencilType> psf_flipped(psf, false);
    for(size_t i = 0; i < psf.mesh.size(); ++i) {
        psf_flipped.mesh[i] = psf.mesh[psf.mesh.size()-1-i];
    }

    std::vector<PixelData<StencilType>> psf_vec;
    std::vector<PixelData<StencilType>> psf_flipped_vec;

    int nstencils = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    APRStencil::get_downsampled_stencils(psf, psf_vec, nstencils, normalize);
    APRStencil::get_downsampled_stencils(psf_flipped, psf_flipped_vec, nstencils, normalize);

    richardson_lucy(apr, particle_input, particle_output, psf_vec, psf_flipped_vec, number_iterations, resume);
}


#endif //LIBAPR_APRNUMERICS_HPP
