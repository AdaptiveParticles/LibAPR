//
// Created by joel on 30.01.20.
//
#include "data_structures/Mesh/PixelData.hpp"

#ifndef LIBAPR_APRSTENCILFUNCTIONS_HPP
#define LIBAPR_APRSTENCILFUNCTIONS_HPP


template<typename T, typename S>
void downsample_stencil(const PixelData<T>& aInput, PixelData<S>& aOutput, const int level_delta, bool normalize = false) {

    const int z_num = aInput.z_num;
    const int x_num = aInput.x_num;
    const int y_num = aInput.y_num;

    const int ndim = (z_num > 1) + (x_num > 1) + (y_num > 1);
    const int step_size = (int)std::pow(2.0f, (float)level_delta);
    const int factor = (int)std::pow((float)step_size, (float)ndim);

    const int z_num_ds = std::max(2*(((z_num-1)/2 + step_size - 1) / step_size)+1, 3);
    const int x_num_ds = std::max(2*(((x_num-1)/2 + step_size - 1) / step_size)+1, 3);
    const int y_num_ds = std::max(2*(((y_num-1)/2 + step_size - 1) / step_size)+1, 3);

    aOutput.initWithValue(y_num_ds, x_num_ds, z_num_ds, 0);

    int z_offset = ((z_num_ds / 2) * step_size - z_num / 2);
    int x_offset = ((x_num_ds / 2) * step_size - x_num / 2);
    int y_offset = ((y_num_ds / 2) * step_size - y_num / 2);

    for (int iz = 0; iz < z_num; ++iz) {
        for (int ix = 0; ix < x_num; ++ix) {
            for (int iy = 0; iy < y_num; ++iy) {

                for(int dz = 0; dz < z_num_ds; ++dz) {

                    int z_start = std::max(dz*step_size, z_offset + iz);
                    int z_end = std::min(dz*step_size + step_size, z_offset + iz + step_size);
                    float overlap_z = (z_end > z_start) ? z_end - z_start : 0;

                    for(int dx = 0; dx < x_num_ds; ++dx){

                        int x_start = std::max(dx*step_size, x_offset + ix);
                        int x_end = std::min(dx*step_size + step_size, x_offset + ix + step_size);
                        float overlap_x = (x_end > x_start) ? x_end - x_start: 0;

                        for(int dy = 0; dy < y_num_ds; ++dy) {

                            int y_start = std::max(dy*step_size, y_offset + iy);
                            int y_end = std::min(dy*step_size + step_size, y_offset + iy + step_size);
                            float overlap_y = (y_end > y_start) ? y_end-y_start : 0;

                            float overlap = overlap_x * overlap_y * overlap_z;
                            aOutput.at(dy, dx, dz) += overlap * aInput.at(iy, ix, iz);
                        }
                    }
                }

            }
        }
    }

    float sum = 0;
    for(size_t i = 0; i < aOutput.mesh.size(); ++i) {
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


template<typename T, typename S>
void get_downsampled_stencils(const PixelData<T>& aInput, std::vector<PixelData<S>>& aOutput, const int nlevels, const bool normalize = false) {

    aOutput.resize(nlevels);
    aOutput[0].init(aInput);
    std::copy(aInput.mesh.begin(), aInput.mesh.end(), aOutput[0].mesh.begin());

    PixelData<S> stencil_ds;
    for (int level_delta = 1; level_delta < nlevels; ++level_delta) {

        downsample_stencil(aInput, stencil_ds, level_delta, normalize);

        aOutput[level_delta].init(stencil_ds);
        std::copy(stencil_ds.mesh.begin(), stencil_ds.mesh.end(), aOutput[level_delta].mesh.begin());
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

    int offset = ((ds_kernel_size / 2) * step_size - kernel_size / 2);

    for (int iz = 0; iz < kernel_size; ++iz) {
        for (int ix = 0; ix < kernel_size; ++ix) {
            for (int iy = 0; iy < kernel_size; ++iy) {

                for(int dz = 0; dz < ds_kernel_size; ++dz) {

                    int z_start = std::max(dz*step_size, offset + iz);
                    int z_end = std::min(dz*step_size + step_size, offset + iz + step_size);
                    float overlap_z = (z_end > z_start) ? z_end - z_start : 0;

                    for(int dx = 0; dx < ds_kernel_size; ++dx){

                        int x_start = std::max(dx*step_size, offset + ix);
                        int x_end = std::min(dx*step_size + step_size, offset + ix + step_size);
                        float overlap_x = (x_end > x_start) ? x_end - x_start: 0;

                        for(int dy = 0; dy < ds_kernel_size; ++dy) {

                            int y_start = std::max(dy*step_size, offset + iy);
                            int y_end = std::min(dy*step_size + step_size, offset + iy + step_size);
                            float overlap_y = (y_end > y_start) ? y_end-y_start : 0;

                            float overlap = overlap_x * overlap_y * overlap_z;

                            aOutput[dz*ds_kernel_size*ds_kernel_size + dx*ds_kernel_size + dy] +=
                                    overlap * aInput[iz*kernel_size*kernel_size + ix*kernel_size + iy];
                        }
                    }
                }

            }
        }
    }

    float sum = 0;
    for(size_t i = 0; i < aOutput.size(); ++i) {
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


#endif //LIBAPR_APRSTENCILFUNCTIONS_HPP
