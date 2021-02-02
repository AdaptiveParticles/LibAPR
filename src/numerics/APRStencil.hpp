//
// Created by joel on 30.01.20.
//
#include "data_structures/Mesh/PixelData.hpp"

#ifndef LIBAPR_APRSTENCIL_HPP
#define LIBAPR_APRSTENCIL_HPP


template<typename stencilType>
class Stencil {
public:
    std::vector<stencilType> data;
    std::vector<int> shape = {0, 0, 0};

    Stencil() = default;
    ~Stencil() = default;

    Stencil(const int y_num, const int x_num, const int z_num) {
        init(y_num, x_num, z_num);
    }

    Stencil(const int y_num, const int x_num, const int z_num, const stencilType aValue) {
        init(y_num, x_num, z_num, aValue);
    }

    void init(const int y_num, const int x_num, const int z_num) {
        shape[0] = y_num;
        shape[1] = x_num;
        shape[2] = z_num;
        data.resize((size_t)z_num*x_num*y_num);
    }

    void init(const int y_num, const int x_num, const int z_num, const stencilType aValue) {
        shape[0] = y_num;
        shape[1] = x_num;
        shape[2] = z_num;
        data.resize((size_t)z_num*x_num*y_num, aValue);
    }

    template<typename T>
    void init(const PixelData<T>& aMesh) {
        shape[0] = aMesh.y_num;
        shape[1] = aMesh.x_num;
        shape[2] = aMesh.z_num;
        data.insert(data.end(), aMesh.mesh.begin(), aMesh.mesh.end());
    }

    template<typename T>
    void init(const Stencil<T>& other) {
        shape[0] = other.shape[0];
        shape[1] = other.shape[1];
        shape[2] = other.shape[2];
        data.insert(data.end(), other.data.begin(), other.data.end());
    }

    void swap(Stencil& other){
        shape.swap(other.shape);
        data.swap(other.data);
    }

    template<typename U>
    void copy(const Stencil<U>& other) {
        shape[0] = other.shape[0];
        shape[1] = other.shape[1];
        shape[2] = other.shape[2];
        data.resize(other.data.size());
        std::copy(other.data.begin(), other.data.end(), data.begin());
    }

    void normalize() {
        stencilType sum = std::accumulate(data.begin(), data.end(), stencilType(0));
        const stencilType inv_sum = 1/sum;
        std::transform(data.begin(), data.end(), data.begin(), [inv_sum](stencilType &a) -> stencilType { return a*inv_sum; });
    }

    /**
     * access element at provided indices without boundary checking
     * @param y
     * @param x
     * @param z
     * @return element @(y, x, z)
     */
    stencilType& at(size_t y, size_t x, size_t z) {
        return data[z * shape[0] * shape[1] + x * shape[0] + y];
    }

    const stencilType& at(size_t y, size_t x, size_t z) const {
        return data[z * shape[0] * shape[1] + x * shape[0] + y];
    }
};


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


    template<typename T, typename S>
    void downsample_stencil(const Stencil<T> &aInput, Stencil<S> &aOutput, const int level_delta) {

        const int y_num = aInput.shape[0];
        const int x_num = aInput.shape[1];
        const int z_num = aInput.shape[2];

        const int ndim = (z_num > 1) + (x_num > 1) + (y_num > 1);
        const int step_size = (int) std::pow(2.0f, (float) level_delta);
        const int factor = (int) std::pow((float) step_size, (float) ndim);

        const int z_num_ds = (z_num > 1) ? std::max(2 * (((z_num - 1) / 2 + step_size - 1) / step_size) + 1, 3) : 1;
        const int x_num_ds = (x_num > 1) ? std::max(2 * (((x_num - 1) / 2 + step_size - 1) / step_size) + 1, 3) : 1;
        const int y_num_ds = (y_num > 1) ? std::max(2 * (((y_num - 1) / 2 + step_size - 1) / step_size) + 1, 3) : 1;

        aOutput.init(y_num_ds, x_num_ds, z_num_ds, 0);

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
        // rescale the result
        for (size_t i = 0; i < aOutput.data.size(); ++i) {
            aOutput.data[i] /= factor;
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

        std::vector<float> gauss_y(size[0]);
        std::vector<float> gauss_x(size[1]);
        std::vector<float> gauss_z(size[2]);

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
            for(size_t i = 0; i < stencil.mesh.size(); ++i) {
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




template<typename stencilType>
class MultiStencil {
public:
    std::vector<Stencil<stencilType>> stencils;

    MultiStencil() = default;
    ~MultiStencil() = default;

    MultiStencil(const int numLevels) { stencils.resize(numLevels); }

    template<typename U>
    MultiStencil(const Stencil<U>& aStencil, const int numLevels=1) {
        init(aStencil, numLevels);
    }

    template<typename U>
    void init(const Stencil<U>& aStencil, const int numLevels=1) {
        stencils.resize(numLevels);
        stencils[0].copy(aStencil);
    }

    inline size_t size() const { return stencils.size(); }
    inline Stencil<stencilType>& operator[](int index) { return stencils[index]; }
    inline const Stencil<stencilType>& operator[](int index) const { return stencils[index]; }

    /**
     * Concatenate all stencils into a single vector (e.g. for transfer to GPU)
     * @tparam T
     * @param vec
     */
    template<typename T>
    void concatenate(VectorData<T>& vec) {
        size_t total_size = 0;
        for(auto s : stencils) {
            total_size += s.size();
        }
        vec.resize(total_size);
        size_t offset = 0;
        for(auto s : stencils) {
            std::copy(s.begin(), s.end(), vec.begin()+offset);
            offset += s.size();
        }
    }

    /**
     * Normalize all stencils to sum to unity
     */
    void normalize() {
        for(auto s : stencils) {
            s.normalize();
        }
    }

    /**
     * Fill 'stencils' vector with numLevels restricted stencils. The restricted stencils are computed such that
     * applying stencils[l] to a patch at level level_max-l is equivalent to the following:
     *      1. interpolate the patch to the fine resolution level_max, via piecewise constant interpolation
     *      2. apply the fine resolution stencil (stencils[0])
     *      3. average downsample the result back to the center element of the coarse resolution patch
     *
     * @param numLevels     number of levels to compute stencil for (typically 'apr.level_max()-apr.level_min()')
     * @param normalize     if true, normalize each stencil to sum to unity
     */
    void restrict_stencils(const int numLevels, const bool normalize=false) {
        stencils.resize(numLevels);
        if(normalize) { stencils[0].normalize(); }

        for (int level_delta = 1; level_delta < numLevels; ++level_delta) {
            APRStencil::downsample_stencil(stencils[0], stencils[level_delta], level_delta);
            if(normalize) { stencils[level_delta].normalize(); }
        }
    }

    /**
     * Fill 'stencils' vector with numLevels stencils, rescaled according to the particle distance at coarser levels.
     * That is, the stencil for level 0 < l < numLevels is computed according to
     *      stencils[l] = stencils[0] / 2^l
     *
     * @param numLevels
     */
    void rescale_stencils(const int numLevels) {
        stencils.resize(numLevels);

        float level_size = 2.0f;
        for (int level_delta = 1; level_delta < numLevels; ++level_delta) {
            stencils[level_delta].init(stencils[0]);

            const float factor = 1.0f / level_size;
            std::transform(stencils[level_delta].data.begin(), stencils[level_delta].data.end(), stencils[level_delta].data.begin(),
                           [factor](float &a){ return a*factor; });
            level_size *= 2;
        }
    }
};



#endif //LIBAPR_APRSTENCIL_HPP
