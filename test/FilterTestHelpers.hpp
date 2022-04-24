//
// Created by joel on 24.04.22.
//

#ifndef APR_FILTERTESTHELPERS_HPP
#define APR_FILTERTESTHELPERS_HPP

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "numerics/APRReconstruction.hpp"
#include <vector>


namespace FilterTestHelpers {

    template<typename InputType, typename StencilType, typename OutputType>
    void compute_convolution_gt(APR& apr,
                                const std::vector<PixelData<StencilType>>& stencil_vec,
                                const ParticleData<InputType>& input_particles,
                                ParticleData<OutputType>& output_particles,
                                bool reflect_boundary = true);

    template<typename InputType, typename OutputType>
    void compute_generic_filter_gt(APR& apr,
                                   const ParticleData<InputType>& input_particles,
                                   ParticleData<OutputType>& output_particles,
                                   int size_y,
                                   int size_x,
                                   int size_z,
                                   bool reflect_boundary,
                                   OutputType filter(std::vector<OutputType>&));


    template<typename T>
    T median(std::vector<T>& input) {
        std::sort(input.begin(), input.end());
        const auto n = input.size();
        return (n % 2 == 0) ? (input[n/2 - 1] + input[n/2]) / 2 : input[n/2];
    }


    template<typename InputType, typename OutputType>
    void compute_median_filter_gt(APR& apr,
                                  const ParticleData<InputType>& input_particles,
                                  ParticleData<OutputType>& output_particles,
                                  int size_y,
                                  int size_x,
                                  int size_z) {

        compute_generic_filter_gt(apr, input_particles, output_particles, size_y, size_x, size_z, true, FilterTestHelpers::median);
    }

}


template<typename InputType, typename StencilType, typename OutputType>
void FilterTestHelpers::compute_convolution_gt(APR &apr,
                                               const std::vector<PixelData<StencilType>> &stencil_vec,
                                               const ParticleData<InputType> &input_particles,
                                               ParticleData<OutputType> &output_particles,
                                               const bool reflect_boundary) {

    output_particles.init(apr);
    auto apr_it = apr.iterator();
    int stencil_counter = 0;

    for(int level = apr.level_max(); level >= apr.level_min(); --level) {
        // reconstruct full image at current level
        ReconPatch patch_spec;
        patch_spec.level_delta = level - apr.level_max();
        PixelData<StencilType> by_level_recon;
        APRReconstruction::reconstruct_constant(apr, by_level_recon, input_particles, patch_spec);

        // current stencil
        PixelData<StencilType> stencil(stencil_vec[stencil_counter], true);
        std::vector<int> stencil_halves = {((int)stencil.y_num-1)/2, ((int)stencil.x_num-1)/2, ((int)stencil.z_num-1)/2};

        // parallel iteration over particles - at each location compute convolution output using the reconstructed image
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it)
#endif
        for (int z = 0; z < apr_it.z_num(level); ++z) {
            for (int x = 0; x < apr_it.x_num(level); ++x) {
                for (apr_it.begin(level, z, x); apr_it < apr_it.end(); apr_it++) {
                    StencilType neigh_sum = 0;
                    size_t counter = 0;
                    const int y = apr_it.y();

                    if(reflect_boundary) {
                        for (int l = -stencil_halves[2]; l <= stencil_halves[2]; ++l) {
                            for (int q = -stencil_halves[1]; q <= stencil_halves[1]; ++q) {
                                for (int w = -stencil_halves[0]; w <= stencil_halves[0]; ++w) {

                                    int iy = y + w;
                                    int ix = x + q;
                                    int iz = z + l;

                                    if(iy < 0) {    // reflect y at boundaries
                                        iy = -iy;
                                    } else if(iy >= apr_it.y_num(level)) {
                                        iy = (apr_it.y_num(level) - 1) - (iy - (apr_it.y_num(level) - 1));
                                    }

                                    if(ix < 0) {    // reflect x at boundaries
                                        ix = -ix;
                                    } else if(ix >= apr_it.x_num(level)) {
                                        ix = (apr_it.x_num(level) - 1) - (ix - (apr_it.x_num(level) - 1));
                                    }

                                    if(iz < 0) {    // reflect z at boundaries
                                        iz = -iz;
                                    } else if(iz >= apr_it.z_num(level)) {
                                        iz = (apr_it.z_num(level) - 1) - (iz - (apr_it.z_num(level) - 1));
                                    }

                                    // accumulate output value and increment counter
                                    neigh_sum += stencil.mesh[counter++] * by_level_recon.at(iy, ix, iz);
                                }
                            }
                        }
                    } else {
                        for (int l = -stencil_halves[2]; l < stencil_halves[2]+1; ++l) {
                            for (int q = -stencil_halves[1]; q < stencil_halves[1]+1; ++q) {
                                for (int w = -stencil_halves[0]; w < stencil_halves[0]+1; ++w) {

                                    int iy = y + w;
                                    int ix = x + q;
                                    int iz = z + l;

                                    // if within bounds, accumulate output value
                                    if((iy>=0) && (iy < apr.y_num(level))){
                                        if((ix>=0) && (ix < apr.x_num(level))){
                                            if((iz>=0) && (iz < apr.z_num(level))) {
                                                neigh_sum += stencil.mesh[counter] * by_level_recon.at(iy, ix, iz);
                                            }
                                        }
                                    }
                                    counter++;
                                }
                            }
                        }
                    }
                    output_particles[apr_it] = neigh_sum;
                }
            }
        }
        // increment stencil_counter to use the next stencil for the next level (if available)
        stencil_counter = std::min(stencil_counter+1, (int)stencil_vec.size()-1);
    }
}


template<typename InputType, typename OutputType>
void FilterTestHelpers::compute_generic_filter_gt(APR &apr, const ParticleData<InputType> &input_particles,
                                                  ParticleData<OutputType> &output_particles, const int size_y,
                                                  const int size_x, const int size_z, const bool reflect_boundary,
                                                  OutputType filter(std::vector<OutputType>&)) {
    output_particles.init(apr);
    auto apr_it = apr.iterator();

    std::vector<OutputType> tmp_vec(size_z * size_x * size_y);

    for(int level = apr.level_max(); level >= apr.level_min(); --level) {
        // reconstruct full image at current level
        ReconPatch patch_spec;
        patch_spec.level_delta = level - apr.level_max();
        PixelData<OutputType> by_level_recon;
        APRReconstruction::reconstruct_constant(apr, by_level_recon, input_particles, patch_spec);

        // iteration over particles - at each location compute filter output using the reconstructed image
        for (int z = 0; z < apr_it.z_num(level); ++z) {
            for (int x = 0; x < apr_it.x_num(level); ++x) {
                for (apr_it.begin(level, z, x); apr_it < apr_it.end(); apr_it++) {

                    const int y = apr_it.y();
                    size_t counter = 0;

                    // copy input patch to `tmp_vec`, taking care of boundary conditions
                    for(int i = -(size_z-1)/2; i <= (size_z-1)/2; ++i) {
                        for(int j = -(size_x-1)/2; j <= (size_x-1)/2; ++j) {
                            for(int k = -(size_y-1)/2; k <= (size_y-1)/2; ++k) {
                                int iz = z + i;
                                int ix = x + j;
                                int iy = y + k;
                                OutputType val = 0;

                                if(reflect_boundary) {

                                    if(iy < 0) {    // reflect y at boundaries
                                        iy = -iy;
                                    } else if(iy >= apr_it.y_num(level)) {
                                        iy = (apr_it.y_num(level) - 1) - (iy - (apr_it.y_num(level) - 1));
                                    }

                                    if(ix < 0) {    // reflect x at boundaries
                                        ix = -ix;
                                    } else if(ix >= apr_it.x_num(level)) {
                                        ix = (apr_it.x_num(level) - 1) - (ix - (apr_it.x_num(level) - 1));
                                    }

                                    if(iz < 0) {    // reflect z at boundaries
                                        iz = -iz;
                                    } else if(iz >= apr_it.z_num(level)) {
                                        iz = (apr_it.z_num(level) - 1) - (iz - (apr_it.z_num(level) - 1));
                                    }

                                    val = by_level_recon.at(iy, ix, iz);
                                } else {
                                    if(iz >= 0 && iz < apr_it.z_num(level) &&
                                       ix >= 0 && ix < apr_it.x_num(level) &&
                                       iy >= 0 && iy < apr_it.y_num(level)) {
                                        val = by_level_recon.at(iy, ix, iz);
                                    }
                                }
                                tmp_vec[counter++] = val;
                            }
                        }
                    }
                    output_particles[apr_it] = filter(tmp_vec);
                }
            }
        }
    }
}


#endif //APR_FILTERTESTHELPERS_HPP
