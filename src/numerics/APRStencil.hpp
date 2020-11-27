//
// Created by joel on 30.01.20.
//
#include "data_structures/Mesh/PixelData.hpp"

#ifndef LIBAPR_APRSTENCIL_HPP
#define LIBAPR_APRSTENCIL_HPP


namespace APRStencil {

    template<typename T, typename S>
    void downsample_stencil(const PixelData<T> &aInput, PixelData<S> &aOutput, const int level_delta,
                            const bool normalize = false) {

        const int z_num = aInput.z_num;
        const int x_num = aInput.x_num;
        const int y_num = aInput.y_num;

        const int ndim = (z_num > 1) + (x_num > 1) + (y_num > 1);
        const int step_size = (int) std::pow(2.0f, (float) level_delta);
        const int factor = (int) std::pow((float) step_size, (float) ndim);

        const int z_num_ds = (z_num > 1) ? std::max(2 * (((z_num - 1) / 2 + step_size - 1) / step_size) + 1, 3) : 1;
        const int x_num_ds = (x_num > 1) ? std::max(2 * (((x_num - 1) / 2 + step_size - 1) / step_size) + 1, 3) : 1;
        const int y_num_ds = (y_num > 1) ? std::max(2 * (((y_num - 1) / 2 + step_size - 1) / step_size) + 1, 3) : 1;

        aOutput.initWithValue(y_num_ds, x_num_ds, z_num_ds, 0);

        int z_offset = ((z_num_ds / 2) * step_size - z_num / 2);
        int x_offset = ((x_num_ds / 2) * step_size - x_num / 2);
        int y_offset = ((y_num_ds / 2) * step_size - y_num / 2);

        for (int iz = 0; iz < z_num; ++iz) {
            for (int ix = 0; ix < x_num; ++ix) {
                for (int iy = 0; iy < y_num; ++iy) {

                    for (int dz = 0; dz < z_num_ds; ++dz) {

                        int z_start = std::max(dz * step_size, z_offset + iz);
                        int z_end = std::min(dz * step_size + step_size, z_offset + iz + step_size);
                        float overlap_z = std::max(z_end - z_start, 0);

                        for (int dx = 0; dx < x_num_ds; ++dx) {

                            int x_start = std::max(dx * step_size, x_offset + ix);
                            int x_end = std::min(dx * step_size + step_size, x_offset + ix + step_size);
                            float overlap_x = std::max(x_end - x_start, 0);

                            for (int dy = 0; dy < y_num_ds; ++dy) {

                                int y_start = std::max(dy * step_size, y_offset + iy);
                                int y_end = std::min(dy * step_size + step_size, y_offset + iy + step_size);
                                float overlap_y = std::max(y_end - y_start, 0);

                                float overlap = overlap_x * overlap_y * overlap_z;
                                aOutput.at(dy, dx, dz) += overlap * aInput.at(iy, ix, iz);
                            }
                        }
                    }
                }
            }
        }

        float sum = 0;
        for (size_t i = 0; i < aOutput.mesh.size(); ++i) {
            aOutput.mesh[i] /= factor;
            sum += aOutput.mesh[i];
        }

        if (normalize) {
            float nfactor = 1.0f / sum;
            for (size_t i = 0; i < aOutput.mesh.size(); ++i) {
                aOutput.mesh[i] *= nfactor;
            }
        }
    }


    inline int compute_stencil_vec_size(const int kernel_size, const int nlevels) {
        int output_size = kernel_size * kernel_size * kernel_size;
        int step_size = 1;

        for (int level_delta = 1; level_delta < nlevels; ++level_delta) {
            step_size *= 2;
            int ds_size = std::max((kernel_size + step_size - 1) / step_size, 3);
            output_size += ds_size * ds_size * ds_size;
        }

        return output_size;
    }


    template<typename T, typename S>
    void get_downsampled_stencils(const PixelData<T> &aInput, VectorData<S> &aOutput, const int nlevels,
                                  const bool normalize = false) {

        const int kernel_size = aInput.y_num;

        aOutput.resize(compute_stencil_vec_size(kernel_size, nlevels));

        std::copy(aInput.mesh.begin(), aInput.mesh.end(), aOutput.begin());

        int c = aInput.mesh.size();
        PixelData<S> stencil_ds;
        for (int level_delta = 1; level_delta < nlevels; ++level_delta) {

            downsample_stencil(aInput, stencil_ds, level_delta, normalize);
            std::copy(stencil_ds.mesh.begin(), stencil_ds.mesh.end(), aOutput.begin() + c);
            c += stencil_ds.mesh.size();
        }
    }


    template<typename T, typename S>
    void get_downsampled_stencils(const PixelData<T> &aInput, std::vector<PixelData<S>> &aOutput, const int nlevels,
                                  const bool normalize = false) {

        aOutput.resize(nlevels);
        aOutput[0].init(aInput);
        std::copy(aInput.mesh.begin(), aInput.mesh.end(), aOutput[0].mesh.begin());

        if(normalize) {
            aOutput[0].normalize();
        }

        for (int level_delta = 1; level_delta < nlevels; ++level_delta) {
            downsample_stencil(aInput, aOutput[level_delta], level_delta, normalize);
        }
    }


    template<typename T, typename S>
    void rescale_stencil(const PixelData<T> &aInput, PixelData<S> &aOutput, const int level_delta) {
        aOutput.init(aInput);
        const float step_size = std::pow(2, level_delta);

        for (size_t i = 0; i < aInput.mesh.size(); ++i) {
            aOutput.mesh[i] = aInput.mesh[i] / step_size;
        }
    }


    template<typename T, typename S>
    void get_rescaled_stencils(const PixelData<T> &aInput, std::vector<PixelData<S>> &aOutput, const int nlevels) {
        aOutput.resize(nlevels);

        for (int level_delta = 0; level_delta < nlevels; ++level_delta) {
            rescale_stencil(aInput, aOutput[level_delta], level_delta);
        }
    }


    template<typename T, typename S>
    void downsample_stencil(const VectorData<T> &aInput, VectorData<S> &aOutput, const int level_delta,
                            const bool normalize = false) {

        /// assumes input kernel has same size in x, y, and z dimensions
        const int kernel_size = (int) std::round(std::cbrt((float) aInput.size()));

        const int step_size = (int) std::pow(2.0f, (float) level_delta);
        const int factor = (int) std::pow((float) step_size, 3.0f);

        const int ds_kernel_size = std::max((kernel_size + step_size - 1) / step_size, 3);

        aOutput.resize(ds_kernel_size * ds_kernel_size * ds_kernel_size, 0);

        int offset = ((ds_kernel_size / 2) * step_size - kernel_size / 2);

        for (int iz = 0; iz < kernel_size; ++iz) {
            for (int ix = 0; ix < kernel_size; ++ix) {
                for (int iy = 0; iy < kernel_size; ++iy) {

                    for (int dz = 0; dz < ds_kernel_size; ++dz) {

                        int z_start = std::max(dz * step_size, offset + iz);
                        int z_end = std::min(dz * step_size + step_size, offset + iz + step_size);
                        float overlap_z = (z_end > z_start) ? z_end - z_start : 0;

                        for (int dx = 0; dx < ds_kernel_size; ++dx) {

                            int x_start = std::max(dx * step_size, offset + ix);
                            int x_end = std::min(dx * step_size + step_size, offset + ix + step_size);
                            float overlap_x = (x_end > x_start) ? x_end - x_start : 0;

                            for (int dy = 0; dy < ds_kernel_size; ++dy) {

                                int y_start = std::max(dy * step_size, offset + iy);
                                int y_end = std::min(dy * step_size + step_size, offset + iy + step_size);
                                float overlap_y = (y_end > y_start) ? y_end - y_start : 0;

                                float overlap = overlap_x * overlap_y * overlap_z;

                                aOutput[dz * ds_kernel_size * ds_kernel_size + dx * ds_kernel_size + dy] +=
                                        overlap * aInput[iz * kernel_size * kernel_size + ix * kernel_size + iy];
                            }
                        }
                    }

                }
            }
        }

        float sum = 0;
        for (size_t i = 0; i < aOutput.size(); ++i) {
            aOutput[i] /= factor;
            sum += aOutput[i];
        }

        if (normalize) {
            float nfactor = 1.0f / sum;
            for (size_t i = 0; i < aOutput.size(); ++i) {
                aOutput[i] *= nfactor;
            }
        }
    }


    template<typename T, typename S>
    void get_downsampled_stencils(const VectorData<T> &aInput, VectorData<S> &aOutput, const int nlevels,
                                  const bool normalize = false) {

        const int kernel_size = (int) std::round(std::cbrt((float) aInput.size()));

        aOutput.resize(compute_stencil_vec_size(kernel_size, nlevels));

        std::copy(aInput.begin(), aInput.end(), aOutput.begin());

        int c = aInput.size();
        VectorData<S> stencil_ds;
        for (int level_delta = 1; level_delta < nlevels; ++level_delta) {

            downsample_stencil(aInput, stencil_ds, level_delta, normalize);
            std::copy(stencil_ds.begin(), stencil_ds.end(), aOutput.begin() + c);
            c += stencil_ds.size();
        }
    }


    template<typename T>
    PixelData<T> create_gaussian_filter(const std::vector<float>& sigma = {1, 1, 1},
                                        const std::vector<int>& size = {5, 5, 5},
                                        const bool normalize=true) {

        PixelData<T> stencil(size[0], size[1], size[2]);

        float gauss_y[size[0]];
        float gauss_x[size[1]];
        float gauss_z[size[2]];

        int c = 0;
        if(size[0] > 1) {
            for (int i = -size[0] / 2; i <= size[0] / 2; ++i) {
                gauss_y[c++] = expf(-i * i / (2 * sigma[0] * sigma[0])) / (sigma[0] * sqrtf(2 * M_PI));
            }
        } else {
            gauss_y[0] = 1;
        }

        if(size[1] > 1) {
            c = 0;
            for (int i = -size[1] / 2; i <= size[1] / 2; ++i) {
                gauss_x[c++] = expf(-i * i / (2 * sigma[1] * sigma[1])) / (sigma[1] * sqrtf(2 * M_PI));
            }
        } else {
            gauss_x[0] = 1;
        }

        if(size[2] > 1) {
            c = 0;
            for (int i = -size[2] / 2; i <= size[2] / 2; ++i) {
                gauss_z[c++] = expf(-i * i / (2 * sigma[2] * sigma[2])) / (sigma[2] * sqrtf(2 * M_PI));
            }
        } else {
            gauss_z[0] = 1;
        }

        float sum = 0;
        for(int i = 0; i < size[0]; ++i) {
            for(int j = 0; j < size[1]; ++j) {
                for(int k = 0; k < size[2]; ++k) {
                    stencil.at(i, j, k) = gauss_y[i] * gauss_x[j] * gauss_z[k];
                    sum += stencil.at(i, j, k);
                }
            }
        }

        if(normalize) {
            for(int i = 0; i < stencil.mesh.size(); ++i) {
                stencil.mesh[i] /= sum;
            }
        }

        return stencil;
    }

    template<typename T>
    PixelData<T> create_gaussian_filter(const float sigma, const int size, const bool normalize=true) {
        return create_gaussian_filter<T>({sigma, sigma, sigma}, {size, size, size}, normalize);
    }


    template<typename T>
    PixelData<T> create_mean_filter(const std::vector<int>& size = {5, 5, 5}) {
        PixelData<T> stencil(size[0], size[1], size[2]);

        T sum = stencil.mesh.size();
        stencil.fill(1.0/sum);

        return stencil;
    }


    template<typename T>
    PixelData<T> create_mean_filter(const int size) {
        return create_mean_filter<T>({size, size, size});
    }


    template<typename T>
    PixelData<T> create_sobel_filter(const int dim, const float delta = 1.0f) {

        PixelData<T> stencil(3, 3, 3);
        const std::vector<float> smooth1d = {0.25f, 0.5f, 0.25f};
        const std::vector<float> diff1d = {-1.0f/(2*delta), 0.0f, 1.0f/(2*delta)};

        // central finite difference in dimension 'dim', smoothing in the remaining dimensions
        const std::vector<std::vector<float>> filter_bank = {
                dim == 0 ? diff1d : smooth1d,
                dim == 1 ? diff1d : smooth1d,
                dim == 2 ? diff1d : smooth1d
        };

        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                for(int k = 0; k < 3; ++k) {
                    stencil.at(i, j, k) = filter_bank[0][i] * filter_bank[1][j] * filter_bank[2][k];
                }
            }
        }
        return stencil;
    }

    template<typename T>
    PixelData<T> create_sobel_filter2d(const int dim) {

        PixelData<T> stencil(3, 3, 1);
        const std::vector<float> smooth1d = {0.25f, 0.5f, 0.25f};
        const std::vector<float> diff1d = {-1.0f, 0.0f, 1.0f};

        // central finite difference in dimension 'dim', smoothing in the other dimension
        const std::vector<std::vector<float>> filter_bank = {
                dim == 0 ? diff1d : smooth1d,
                dim == 1 ? diff1d : smooth1d,
        };

        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                stencil.at(i, j, 1) = filter_bank[0][i] * filter_bank[1][j];
            }
        }
        return stencil;
    }
}


///**
// * For testing purposes only! Sums contribution in an explicit loop using Kahan summation. Gets very slow for level_delta > ~6
// * Aggressive optimization may remove the Kahan summation steps
// */
//template<typename T, typename S>
//void downsample_stencil_bruteforce(const PixelData<T>& aInput, PixelData<S>& aOutput, const int level_delta, bool normalize = false) {
//
//    const int z_num = aInput.z_num;
//    const int x_num = aInput.x_num;
//    const int y_num = aInput.y_num;
//
//    const int ndim = (z_num > 1) + (x_num > 1) + (y_num > 1);
//    const int step_size = (int)std::pow(2.0f, (float)level_delta);
//    const int factor = (int)std::pow((float)step_size, (float)ndim);
//
//    const int z_num_ds = std::max((z_num + step_size - 1) / step_size, 3);
//    const int x_num_ds = std::max((x_num + step_size - 1) / step_size, 3);
//    const int y_num_ds = std::max((y_num + step_size - 1) / step_size, 3);
//
//    aOutput.initWithValue(y_num_ds, x_num_ds, z_num_ds, 0);
//
//    const int z_offset = ((z_num_ds / 2) * step_size - z_num / 2);
//    const int x_offset = ((x_num_ds / 2) * step_size - x_num / 2);
//    const int y_offset = ((y_num_ds / 2) * step_size - y_num / 2);
//
//    /// loop over output stencil elements
//    for (int dz = 0; dz < z_num_ds; ++dz) {
//        for (int dx = 0; dx < x_num_ds; ++dx) {
//            for (int dy = 0; dy < y_num_ds; ++dy) {
//
//                float sum = 0.0f;
//                float c = 0.0f;
//
//                /// loop over input stencil elements
//                for (int iz = 0; iz < z_num; ++iz) {
//                    for (int ix = 0; ix < x_num; ++ix) {
//                        for (int iy = 0; iy < y_num; ++iy) {
//
//                            /// loop over high-res stencil positions in low-res center pixel
//                            for(int z_pos = 0; z_pos < step_size; ++z_pos) {
//                                int z_coord_ds = (z_offset + z_pos + iz) / step_size;
//                                if(z_coord_ds != dz) { continue; }
//
//                                for(int x_pos = 0; x_pos < step_size; ++x_pos) {
//                                    int x_coord_ds = (x_offset + x_pos + ix) / step_size;
//                                    if(x_coord_ds != dx) { continue; }
//
//                                    for(int y_pos = 0; y_pos < step_size; ++y_pos) {
//                                        int y_coord_ds = (y_offset + y_pos + iy) / step_size;
//                                        if(y_coord_ds != dy) { continue; }
//
//                                        /// if the high-res stencil element is within the current low-res element, add it to the sum
//
//                                        // Kahan summation to reduce numerical errors
//                                        float y = aInput.at(iy, ix, iz) - c;
//                                        float t = sum + y;
//                                        c = (t - sum) - y;
//                                        sum = t;
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }
//                aOutput.at(dy, dx, dz) = sum;
//            }
//        }
//    }
//
//    float sum = 0;
//    for(size_t i = 0; i < aOutput.mesh.size(); ++i) {
//        aOutput.mesh[i] /= factor;
//        sum += aOutput.mesh[i];
//    }
//
//    if (normalize) {
//        float nfactor = 1.0f / sum;
//        for (size_t i = 0; i < aOutput.mesh.size(); ++i) {
//            aOutput.mesh[i] *= nfactor;
//        }
//    }
//}
//
//template<typename T, typename S>
//void get_downsampled_stencils_bruteforce(const PixelData<T>& aInput, VectorData<S>& aOutput, const int nlevels, const bool normalize = false) {
//
//    const int kernel_size = aInput.y_num;
//
//    aOutput.resize( compute_stencil_vec_size(kernel_size, nlevels) );
//
//    std::copy(aInput.mesh.begin(), aInput.mesh.end(), aOutput.begin());
//
//    int c = aInput.mesh.size();
//    PixelData<S> stencil_ds;
//    for (int level_delta = 1; level_delta < nlevels; ++level_delta) {
//
//        downsample_stencil_bruteforce(aInput, stencil_ds, level_delta, normalize);
//        std::copy(stencil_ds.mesh.begin(), stencil_ds.mesh.end(), aOutput.begin() + c);
//        c += stencil_ds.mesh.size();
//    }
//}


#endif //LIBAPR_APRSTENCIL_HPP
