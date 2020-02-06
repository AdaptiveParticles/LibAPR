//
// Created by joel on 30.01.20.
//
#include "data_structures/Mesh/PixelData.hpp"

#ifndef LIBAPR_APRSTENCILFUNCTIONS_HPP
#define LIBAPR_APRSTENCILFUNCTIONS_HPP

template<typename T, typename S>
void downsample_stencil(const PixelData<T>& aInput, PixelData<S>& aOutput, const int level_delta, bool normalize = false) {

    const size_t z_num = aInput.z_num;
    const size_t x_num = aInput.x_num;
    const size_t y_num = aInput.y_num;

    const int ndim = (z_num > 1) + (x_num > 1) + (y_num > 1);
    const int step_size = (int)std::pow(2.0f, (float)level_delta);
    const int factor = (int)std::pow((float)step_size, (float)ndim);

    size_t z_num_ds = std::max((z_num + step_size - 1) / step_size, (size_t)3);
    size_t x_num_ds = std::max((x_num + step_size - 1) / step_size, (size_t)3);
    size_t y_num_ds = std::max((y_num + step_size - 1) / step_size, (size_t)3);

    aOutput.initWithValue(y_num_ds, x_num_ds, z_num_ds, 0);

    for (int dz = 0; dz < step_size; ++dz) {
        for (int dx = 0; dx < step_size; ++dx) {
            for (int dy = 0; dy < step_size; ++dy) {

                for (int iz = 0; iz < z_num; ++iz) {
                    // center of DS stencil in original coords + delta + original stencil coord (origin at center)
                    int z_ds = ((z_num_ds / 2) * step_size + dz + iz - z_num / 2) / step_size;

                    for (int ix = 0; ix < x_num; ++ix) {

                        int x_ds = ((x_num_ds / 2) * step_size + dx + ix - x_num / 2) / step_size;

                        for (int iy = 0; iy < y_num; ++iy) {

                            int y_ds = ((y_num_ds / 2) * step_size + dy + iy - y_num / 2) / step_size;

                            aOutput.at(y_ds, x_ds, z_ds) += aInput.at(iy, ix, iz);

                        }
                    }
                }
            }
        }
    }

    float sum = 0;
    for(int i = 0; i < aOutput.mesh.size(); ++i) {
        aOutput.mesh[i] /= factor;
        sum += aOutput.mesh[i];
    }

    if (normalize) {
        float nfactor = 1.0f / sum;
        for (int i = 0; i < aOutput.mesh.size(); ++i) {
            aOutput.mesh[i] *= nfactor;
        }
    }
}

inline int compute_stencil_vec_size(const int kernel_size, const int nlevels) {
    int output_size = kernel_size * kernel_size * kernel_size;
    int step_size = 1;

    for(int level_delta = 1; level_delta < nlevels; ++level_delta) {
        step_size *= 2;
        int ds_size = std::max((kernel_size + step_size - 1) / step_size, 3);
        output_size += ds_size * ds_size * ds_size;
    }

    return output_size;
}

template<typename T, typename S>
void get_downsampled_stencils(const PixelData<T>& aInput, VectorData<S>& aOutput, const int nlevels, const bool normalize = false) {

    const int kernel_size = aInput.y_num;

    aOutput.resize( compute_stencil_vec_size(kernel_size, nlevels) );

    std::copy(aInput.mesh.begin(), aInput.mesh.end(), aOutput.begin());

    int c = aInput.mesh.size();
    PixelData<S> stencil_ds;
    for (int level_delta = 1; level_delta < nlevels; ++level_delta) {

        downsample_stencil(aInput, stencil_ds, level_delta, normalize);
        std::copy(stencil_ds.mesh.begin(), stencil_ds.mesh.end(), aOutput.begin() + c);
        c += stencil_ds.mesh.size();
    }
}


/// todo: doesnt seem to work properly -- fix!
template<typename T, typename S>
void downsample_stencil(VectorData<T>& aInput, VectorData<S>& aOutput, int level_delta, bool normalize = false) {

    const int kernel_size = (int)std::round(std::cbrt((float)aInput.size())); /// assumes input kernel has same size in x, y, and z dimensions

    const int step_size = (int)std::pow(2.0f, (float)level_delta);
    const int factor = (int)std::pow((float)step_size, 3.0f);

    const int ds_kernel_size = std::max((kernel_size + step_size - 1) / step_size, 3);

    aOutput.resize(ds_kernel_size*ds_kernel_size*ds_kernel_size, 0);

    for (int dz = 0; dz < step_size; ++dz) {
        for (int dx = 0; dx < step_size; ++dx) {
            for (int dy = 0; dy < step_size; ++dy) {

                for (int iz = 0; iz < kernel_size; ++iz) {
                    // "center of DS stencil in original coords" + "delta" + "original stencil coord (origin at center)"
                    int z_ds = ((ds_kernel_size / 2) * step_size + dz + iz - kernel_size / 2) / step_size;

                    for (int ix = 0; ix < kernel_size; ++ix) {

                        int x_ds = ((ds_kernel_size / 2) * step_size + dx + ix - kernel_size / 2) / step_size;

                        for (int iy = 0; iy < kernel_size; ++iy) {

                            int y_ds = ((ds_kernel_size / 2) * step_size + dy + iy - kernel_size / 2) / step_size;

                            aOutput[y_ds + x_ds * ds_kernel_size + z_ds*ds_kernel_size*ds_kernel_size] =
                                    aInput[iy + ix*kernel_size + iz*kernel_size*kernel_size];
                        }
                    }
                }
            }
        }
    }

    float sum = 0;
    for(int i = 0; i < aOutput.size(); ++i) {
        aOutput[i] /= factor;
        sum += aOutput[i];
    }

    if (normalize) {
        float nfactor = 1.0f / sum;
        for (int i = 0; i < aOutput.size(); ++i) {
            aOutput[i] *= nfactor;
        }
    }
}

/// todo: doesnt seem to work properly -- fix!
template<typename T, typename S>
void get_downsampled_stencils(VectorData<T>& aInput, VectorData<S>& aOutput, const int nlevels, const bool normalize = false) {

    const int kernel_size = (int)std::round( std::cbrt((float)aInput.size()) );

    aOutput.resize( compute_stencil_vec_size(kernel_size, nlevels) );

    std::copy(aInput.begin(), aInput.end(), aOutput.begin());

    int c = aInput.size();
    VectorData<S> stencil_ds;
    for (int level_delta = 1; level_delta < nlevels; ++level_delta) {

        downsample_stencil(aInput, stencil_ds, level_delta, normalize);
        std::copy(stencil_ds.begin(), stencil_ds.end(), aOutput.begin() + c);
        c += stencil_ds.size();
    }
}

#endif //LIBAPR_APRSTENCILFUNCTIONS_HPP
