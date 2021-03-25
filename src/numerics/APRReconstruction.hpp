//
// Created by cheesema on 16.01.18.
// Modified by joeljonsson on 03.03.21
//

#ifndef LIBAPR_APRRECONSTRUCTION_HPP
#define LIBAPR_APRRECONSTRUCTION_HPP

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/iterators/APRIterator.hpp"
#include "data_structures/APR/iterators/APRTreeIterator.hpp"
#include "numerics/MeshNumerics.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/APR/particles/PartCellData.hpp"
#include "numerics/APRTreeNumerics.hpp"

struct ReconPatch{
    int x_begin=0;
    int x_end=-1;
    int y_begin=0;
    int y_end=-1;
    int z_begin=0;
    int z_end=-1;
    int level_delta=0;

    size_t size() const {
        return (size_t)(z_end-z_begin)*(x_end-x_begin)*(y_end-y_begin);
    }

    /**
     * Check the patch limits against the domain size of an APR. Lower limits < 0 are set to 0, upper limits < 0
     * are set to the dimension size at the specified level_delta.
     * @param apr
     */
    bool check_limits(APR& apr) {
        int max_img_y = std::ceil(apr.org_dims(0)*pow(2.f, level_delta));
        int max_img_x = std::ceil(apr.org_dims(1)*pow(2.f, level_delta));
        int max_img_z = std::ceil(apr.org_dims(2)*pow(2.f, level_delta));

        y_begin = std::max(0, y_begin);
        x_begin = std::max(0, x_begin);
        z_begin = std::max(0, z_begin);

        y_end = y_end < 0 ? max_img_y : std::min(max_img_y, y_end);
        x_end = x_end < 0 ? max_img_x : std::min(max_img_x, x_end);
        z_end = z_end < 0 ? max_img_z : std::min(max_img_z, z_end);

        if(y_begin >= y_end || x_begin >= x_end || z_begin >= z_end) {
            std::wcerr << "ReconPatch expects begin < end in all dimensions" << std::endl;
            return false;
        }
        return true;
    }
};


namespace APRReconstruction {

    /**
     * Reconstruct pixel image by piecewise constant interpolation
     * @tparam U
     * @tparam ParticleDataType
     * @param apr
     * @param img
     * @param parts
     */
    template<typename U,typename ParticleDataType>
    void reconstruct_constant(APR& apr, PixelData<U>& img, ParticleDataType& parts) {

        const int y_num = apr.y_num(apr.level_max());
        const int x_num = apr.x_num(apr.level_max());
        const int z_num = apr.z_num(apr.level_max());

        auto it = apr.iterator();

        img.initWithResize(y_num, x_num, z_num);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it) collapse(2)
#endif
        for(int z = 0; z < z_num; ++z) {
            for(int x = 0; x < x_num; ++x) {

                /// insert finest resolution particles
                for(it.begin(it.level_max(), z, x); it < it.end(); ++it) {
                    img.at(it.y(), x, z) = parts[it];
                }

                /// insert coarser particles (multiple pixel values for each particle)
                for(int level = it.level_max()-1; level >= it.level_min(); --level) {

                    const int step_size = std::pow(2, it.level_max() - level);
                    const int x_l = x / step_size;
                    const int z_l = z / step_size;

                    for(it.begin(level, z_l, x_l); it < it.end(); ++it) {
                        const int y_begin = it.y() * step_size;
                        const int y_end = std::min(y_begin+step_size, y_num);

                        const U pval = parts[it];
                        for(int y = y_begin; y < y_end; ++y) {
                            img.at(y, x, z) = pval;
                        }
                    }
                }
            }
        }
    }


    /**
     * Construct pixel image containing the particle levels at each pixel location
     * @tparam U
     * @tparam ParticleDataType
     * @param apr
     * @param img
     * @param parts
     */
    template<typename U>
    void reconstruct_level(APR& apr, PixelData<U>& img) {

        const int y_num = apr.y_num(apr.level_max());
        const int x_num = apr.x_num(apr.level_max());
        const int z_num = apr.z_num(apr.level_max());

        auto it = apr.iterator();

        img.initWithResize(y_num, x_num, z_num);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it) collapse(2)
#endif
        for(int z = 0; z < z_num; ++z) {
            for(int x = 0; x < x_num; ++x) {

                /// insert maximum level
                const int max_level = it.level_max();
                for(it.begin(it.level_max(), z, x); it < it.end(); ++it) {
                    img.at(it.y(), x, z) = max_level;
                }

                /// insert coarser levels (multiple pixel values for each particle)
                for(int level = max_level-1; level >= it.level_min(); --level) {

                    const int step_size = std::pow(2, it.level_max() - level);
                    const int x_l = x / step_size;
                    const int z_l = z / step_size;

                    for(it.begin(level, z_l, x_l); it < it.end(); ++it) {
                        const int y_begin = it.y() * step_size;
                        const int y_end = std::min(y_begin+step_size, y_num);
                        for(int y = y_begin; y < y_end; ++y) {
                            img.at(y, x, z) = level;
                        }
                    }
                }
            }
        }
    }


    /**
     * Piecewise constant reconstruction in an image patch, with user-supplied tree particle values.
     * Can be used to reconstruct at a different resolution by specifying patch.level_delta
     */
    template<typename U, typename partsType, typename treeType>
    void reconstruct_constant(APR& apr, PixelData<U>& img, partsType& parts, treeType& tree_parts, ReconPatch& patch);


    /**
     * Piecewise constant patch reconstruction using default method to compute tree particle values.
     * If `patch.level_delta < 0`, values are capped at `apr.level_max() + patch.level_delta`.
     */
    template<typename U, typename partsType>
    void reconstruct_constant(APR& apr, PixelData<U>& img, partsType& parts, ReconPatch& patch) {
        ParticleData<float> tree_data;
        reconstruct_constant(apr, img, parts, tree_data, patch);
    }

    /**
     * Piecewise constant reconstruction in an image (sub-) region specified by 'patch'.
     */
    template<typename U>
    void reconstruct_level(APR& apr, PixelData<U>& img, ReconPatch& patch);


    template<typename T>
    void calc_sat_adaptive_y(PixelData<T>& input, PixelData<uint8_t>& offset_img, float scale_in, int offset_max_in, int d_max);

    template<typename T>
    void calc_sat_adaptive_x(PixelData<T>& input, PixelData<uint8_t>& offset_img, float scale_in, int offset_max_in, int d_max);

    template<typename T>
    void calc_sat_adaptive_z(PixelData<T>& input, PixelData<uint8_t>& offset_img, float scale_in, int offset_max_in, int d_max);

    /**
     * Reconstruct pixel values via separable level-adaptive smoothing
     * @tparam U
     * @tparam partsType
     * @param apr
     * @param img
     * @param parts
     * @param scale_d
     */
    template<typename U,typename partsType>
    void reconstruct_smooth(APR& apr, PixelData<U>& img, partsType& parts, const std::vector<float>& scale_d = {2, 2, 2});

    /**
     * Smooth reconstruction in a patch, with user-supplied tree particle values
     */
    template<typename U, typename partsType, typename treeType>
    void reconstruct_smooth(APR& apr, PixelData<U>& img, partsType& parts, treeType& tree_data, ReconPatch& patch, const std::vector<float>& scale_d = {2, 2, 2});

    /**
     * Smooth reconstruction in a patch, using default method to compute tree particle values
     */
    template<typename U, typename partsType>
    void reconstruct_smooth(APR& apr, PixelData<U>& img, partsType& parts, ReconPatch& patch, const std::vector<float>& scale_d = {2, 2, 2}) {
        ParticleData<float> tree_data;
        reconstruct_smooth(apr, img, parts, tree_data, patch, scale_d);
    }


    /**
     * Reconstruction via iterative upsampling, optionally with smoothing in each step
     */
    template<typename U,typename V>
    void interp_img_us_smooth(APR& apr, PixelData<U>& img, ParticleData<V>& parts, bool smooth, int delta = 0);
}


template<typename U,typename partsType>
void APRReconstruction::reconstruct_smooth(APR& apr, PixelData<U>& img, partsType& parts, const std::vector<float>& scale_d){
    //
    //  Performs a smooth interpolation, based on the depth (level l) in each direction.
    //

    PixelData<uint8_t> k_img;

    int offset_max = 20;

    reconstruct_constant(apr, img, parts);
    reconstruct_level(apr, k_img);

    if(img.y_num > 1) {
        calc_sat_adaptive_y(img, k_img, scale_d[0], offset_max, apr.level_max());
    }

    if(img.x_num > 1) {
        calc_sat_adaptive_x(img, k_img, scale_d[1], offset_max, apr.level_max());
    }

    if(img.z_num > 1) {
        calc_sat_adaptive_z(img, k_img, scale_d[2], offset_max, apr.level_max());
    }
}


template<typename U, typename partsType, typename treeType>
void APRReconstruction::reconstruct_constant(APR& apr, PixelData<U>& img, partsType& parts, treeType& tree_parts, ReconPatch& patch) {

    if (!patch.check_limits(apr)) {
        std::wcerr << "APRReconstruction::reconstruct_constant: invalid patch size - exiting" << std::endl;
        return;
    }

    if (patch.level_delta < 0) {
        if (tree_parts.size() != apr.total_number_tree_particles()) {
            APRTreeNumerics::fill_tree_mean(apr, parts, tree_parts);
        }
    }

    const int y_begin = patch.y_begin;
    const int y_end = patch.y_end;
    const int x_begin = patch.x_begin;
    const int x_end = patch.x_end;
    const int z_begin = patch.z_begin;
    const int z_end = patch.z_end;

    img.initWithResize(y_end - y_begin, x_end - x_begin, z_end - z_begin);

    auto it = apr.iterator();
    const int max_level = it.level_max() + patch.level_delta;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it) collapse(2)
#endif
    for(int x = x_begin; x < x_end; ++x) {
        for(int z = z_begin; z < z_end; ++z) {

            // x and z coordinates in the patch buffer
            const int x_off = x - x_begin;
            const int z_off = z - z_begin;

            for(int level = std::min(max_level, it.level_max()); level >= it.level_min(); --level) {

                if(level == max_level) {
                    // find start of y region
                    it.begin(level, z, x);
                    while(it.y() < y_begin && it < it.end()) { ++it; }

                    // iterate over y region and insert values
                    for(; (it < it.end()) && (it.y() < y_end); ++it) {
                        img.at(it.y() - y_begin, x_off, z_off) = parts[it];
                    }
                } else {
                    const float step_size = std::pow(2.f, max_level - level);
                    const int x_l = std::floor(x / step_size);
                    const int z_l = std::floor(z / step_size);
                    const int y_begin_l = std::floor(y_begin / step_size);
                    const int y_end_l = std::min((int) std::ceil(y_end / step_size), it.y_num(level));

                    // find start of y region
                    it.begin(level, z_l, x_l);
                    while (it.y() < y_begin_l && it < it.end()) { ++it; }

                    // iterate over y region and insert values
                    for (; (it < it.end()) && (it.y() < y_end_l); ++it) {
                        const int yp_begin = std::max((int) (it.y() * step_size), patch.y_begin) - patch.y_begin;
                        const int yp_end = std::min((int) ((it.y() + 1) * step_size), patch.y_end) - patch.y_begin;
                        const U pval = parts[it];

                        for (int y = yp_begin; y < yp_end; ++y) {
                            img.at(y, x_off, z_off) = pval;
                        }
                    }
                }
            }
        }
    }

    // if reconstructing at a coarse resolution, downsampled values are taken from the interior tree particles
    if(max_level < it.level_max()) {
        auto tree_it = apr.tree_iterator();
        const int level = max_level;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(tree_it) collapse(2)
#endif
        for(int x = x_begin; x < x_end; ++x) {
            for (int z = z_begin; z < z_end; ++z) {

                const int x_off = x - x_begin;
                const int z_off = z - z_begin;

                // find start of y region
                tree_it.begin(level, z, x);
                while(tree_it.y() < y_begin && tree_it < tree_it.end()) { ++tree_it; }

                // iterate over y region and project
                for(; (tree_it < tree_it.end()) && (tree_it.y() < y_end); ++tree_it) {
                    img.at(tree_it.y() - y_begin, x_off, z_off) = tree_parts[tree_it];
                }
            }
        }
    }
}


template<typename U>
void APRReconstruction::reconstruct_level(APR& apr, PixelData<U>& img, ReconPatch& patch) {

    if (!patch.check_limits(apr)) {
        std::wcerr << "APRReconstruction::reconstruct_level: invalid patch size - exiting" << std::endl;
        return;
    }

    const int y_begin = patch.y_begin;
    const int y_end = patch.y_end;
    const int x_begin = patch.x_begin;
    const int x_end = patch.x_end;
    const int z_begin = patch.z_begin;
    const int z_end = patch.z_end;

    img.initWithResize(y_end - y_begin, x_end - x_begin, z_end - z_begin);

    auto it = apr.iterator();
    const int max_level = it.level_max() + patch.level_delta;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it) collapse(2)
#endif
    for(int x = x_begin; x < x_end; ++x) {
        for(int z = z_begin; z < z_end; ++z) {

            // x and z coordinates in the patch buffer
            const int x_off = x - x_begin;
            const int z_off = z - z_begin;

            for(int level = std::min(max_level, it.level_max()); level >= it.level_min(); --level) {

                if(level == max_level) {
                    // find start of y region
                    it.begin(level, z, x);
                    while(it.y() < y_begin && it < it.end()) { ++it; }

                    // iterate over y region and insert values
                    for(; (it < it.end()) && (it.y() < y_end); ++it) {
                        img.at(it.y() - y_begin, x_off, z_off) = max_level;
                    }
                } else {
                    const float step_size = std::pow(2.f, max_level - level);
                    const int x_l = std::floor(x / step_size);
                    const int z_l = std::floor(z / step_size);
                    const int y_begin_l = std::floor(y_begin / step_size);
                    const int y_end_l = std::min((int) std::ceil(y_end / step_size), it.y_num(level));

                    // find start of y region
                    it.begin(level, z_l, x_l);
                    while (it.y() < y_begin_l && it < it.end()) { ++it; }

                    // iterate over y region and insert values
                    for (; (it < it.end()) && (it.y() < y_end_l); ++it) {
                        const int yp_begin = std::max((int) (it.y() * step_size), patch.y_begin) - patch.y_begin;
                        const int yp_end = std::min((int) ((it.y() + 1) * step_size), patch.y_end) - patch.y_begin;

                        for (int y = yp_begin; y < yp_end; ++y) {
                            img.at(y, x_off, z_off) = level;
                        }
                    }
                }
            }
        }
    }

    // if reconstructing at a coarse resolution
    if(max_level < it.level_max()) {

        auto tree_it = apr.tree_iterator();
        const int level = max_level;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(tree_it) collapse(2)
#endif
        for(int x = x_begin; x < x_end; ++x) {
            for (int z = z_begin; z < z_end; ++z) {

                const int x_off = x - x_begin;
                const int z_off = z - z_begin;

                // find start of y region
                tree_it.begin(level, z, x);
                while(tree_it.y() < y_begin && tree_it < tree_it.end()) { ++tree_it; }

                // iterate over y region and project
                for(; (tree_it < tree_it.end()) && (tree_it.y() < y_end); ++tree_it) {
                    img.at(tree_it.y() - y_begin, x_off, z_off) = level;
                }
            }
        }
    }
}


template<typename U, typename partsType, typename treeType>
void APRReconstruction::reconstruct_smooth(APR& apr, PixelData<U>& img, partsType& parts, treeType& tree_data,
                                           ReconPatch& patch, const std::vector<float>& scale_d) {
    //
    //  Performs a smooth interpolation, based on the depth (level l) in each direction.
    //

    if (!patch.check_limits(apr)) {
        std::wcerr << "APRReconstruction::reconstruct_smooth: invalid patch size - exiting" << std::endl;
        return;
    }

    PixelData<uint8_t> k_img;

    int offset_max = 10;

    reconstruct_constant(apr, img, parts, tree_data, patch);
    reconstruct_level(apr, k_img, patch);

    const int max_level = apr.level_max() + patch.level_delta;

    if(img.y_num > 1) {
        calc_sat_adaptive_y(img, k_img, scale_d[0], offset_max, max_level);
    }

    if(img.x_num > 1) {
        calc_sat_adaptive_x(img, k_img, scale_d[1], offset_max, max_level);
    }

    if(img.z_num > 1) {
        calc_sat_adaptive_z(img, k_img, scale_d[2], offset_max, max_level);
    }
}


template<typename T>
void APRReconstruction::calc_sat_adaptive_y(PixelData<T> &input, PixelData<uint8_t> &offset_img, float scale_in,
                                            int offset_max_in, const int d_max) {
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates a O(1) recursive mean using SAT.
    //

    offset_max_in = std::min(offset_max_in, (input.y_num/2 - 1));

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    std::vector<float> temp_vec;
    temp_vec.resize(y_num,0);

    std::vector<float> offset_vec;
    offset_vec.resize(y_num,0);


    int64_t i, k, index;
    float counter, temp, divisor,offset;

    //need to introduce an offset max to make the algorithm still work, and it also makes sense.
    const int offset_max = offset_max_in;

    float scale = scale_in;

    //const unsigned int d_max = this->level_max();

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(i,k,counter,temp,index,divisor,offset) firstprivate(temp_vec,offset_vec)
#endif
    for(int64_t j = 0;j < z_num;j++){
        for(i = 0;i < x_num;i++){

            index = j*x_num*y_num + i*y_num;

            //first update the fixed length scale
            for (k = 0; k < y_num;k++){
                offset_vec[k] = std::min((T)floor(pow(2,d_max- offset_img.mesh[index + k])/scale),(T)offset_max);
            }


            //first pass over and calculate cumsum
            temp = 0;
            for (k = 0; k < y_num;k++){
                temp += input.mesh[index + k];
                temp_vec[k] = temp;
            }

            input.mesh[index] = 0;
            //handling boundary conditions (LHS)
            for (k = 1; k <= (offset_max+1);k++){
                divisor = 2*offset_vec[k] + 1;
                offset = offset_vec[k];

                if(k <= (offset+1)){
                    //emulate the bc
                    input.mesh[index + k] = -temp_vec[0]/divisor;
                }

            }

            //second pass calculate mean
            for (k = 0; k < y_num;k++){
                divisor = 2*offset_vec[k] + 1;
                offset = offset_vec[k];
                if(k >= (offset+1)){
                    input.mesh[index + k] = -temp_vec[k - offset - 1]/divisor;
                }

            }


            //second pass calculate mean
            for (k = 0; k < (y_num);k++){
                divisor = 2*offset_vec[k] + 1;
                offset = offset_vec[k];
                if(k < (y_num - offset)) {
                    input.mesh[index + k] += temp_vec[k + offset] / divisor;
                }
            }


            counter = 0;
            //handling boundary conditions (RHS)
            for (k = ( y_num - offset_max); k < (y_num);k++){

                divisor = 2*offset_vec[k] + 1;
                offset = offset_vec[k];

                if(k >= (y_num - offset)){
                    counter = k - (y_num-offset)+1;

                    input.mesh[index + k]*= divisor;
                    input.mesh[index + k]+= temp_vec[y_num-1];
                    input.mesh[index + k]*= 1.0/(divisor - counter);

                }

            }

            //handling boundary conditions (LHS), need to rehandle the boundary
            for (k = 1; k < (offset_max + 1);k++){

                divisor = 2*offset_vec[k] + 1;
                offset = offset_vec[k];

                if(k < (offset + 1)){
                    input.mesh[index + k] *= divisor/(1.0*k + offset);
                }

            }

            //end point boundary condition
            divisor = 2*offset_vec[0] + 1;
            offset = offset_vec[0];
            input.mesh[index] *= divisor/(offset+1);
        }
    }
}


template<typename T>
void APRReconstruction::calc_sat_adaptive_x(PixelData<T> &input, PixelData<uint8_t> &offset_img, float scale_in,
                                            int offset_max_in, int d_max) {
    //
    //  Adaptive form of Matteusz' SAT code.
    //

    offset_max_in = std::min(offset_max_in, (input.x_num/2 - 1));

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    int offset_max = offset_max_in;

    std::vector<float> temp_vec;
    temp_vec.resize(y_num*(2*offset_max + 2),0);

    int64_t i,k;
    float temp;
    int64_t index_modulo, previous_modulo, jxnumynum, offset,forward_modulo,backward_modulo;

    const float scale = scale_in;
    //const unsigned int d_max = this->level_max();


#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(i,k,temp,index_modulo, previous_modulo, forward_modulo,backward_modulo, jxnumynum,offset) \
        firstprivate(temp_vec)
#endif
    for(int j = 0; j < z_num; j++) {

        jxnumynum = j * x_num * y_num;

        //prefetching

        for(k = 0; k < y_num ; k++){
            // std::copy ?
            temp_vec[k] = input.mesh[jxnumynum + k];
        }


        for(i = 1; i < 2 * offset_max + 1; i++) {
            for(k = 0; k < y_num; k++) {
                temp_vec[i*y_num + k] = input.mesh[jxnumynum + i*y_num + k] + temp_vec[(i-1)*y_num + k];
            }
        }


        // LHS boundary

        for(i = 0; i < offset_max + 1; i++){
            for(k = 0; k < y_num; k++) {
                offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[jxnumynum + i * y_num + k])/scale),(T)offset_max);
                if(i < (offset + 1)) {
                    input.mesh[jxnumynum + i * y_num + k] = (temp_vec[(i + offset) * y_num + k]) / (i + offset + 1);
                }
            }
        }

        // middle

        //for(i = offset + 1; i < x_num - offset; i++){

        for(i = 1; i < x_num ; i++){
            // the current cumsum

            for(k = 0; k < y_num; k++) {


                offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[jxnumynum + i * y_num + k])/scale),(T)offset_max);

                if((i >= offset_max + 1) & (i < (x_num - offset_max))) {
                    // update the buffers

                    index_modulo = (i + offset_max) % (2 * offset_max + 2);
                    previous_modulo = (i + offset_max - 1) % (2 * offset_max + 2);
                    temp = input.mesh[jxnumynum + (i + offset_max) * y_num + k] + temp_vec[previous_modulo * y_num + k];
                    temp_vec[index_modulo * y_num + k] = temp;

                }

                //perform the mean calculation
                if((i >= offset+ 1) & (i < (x_num - offset))) {
                    // calculate the positions in the buffers
                    forward_modulo = (i + offset) % (2 * offset_max + 2);
                    backward_modulo = (i - offset - 1) % (2 * offset_max + 2);
                    input.mesh[jxnumynum + i * y_num + k] = (temp_vec[forward_modulo * y_num + k] - temp_vec[backward_modulo * y_num + k]) /
                                                            (2 * offset + 1);

                }
            }

        }

        // RHS boundary //circular buffer

        for(i = x_num - offset_max; i < x_num; i++){

            for(k = 0; k < y_num; k++){

                offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[jxnumynum + i * y_num + k])/scale),(T)offset_max);

                if(i >= (x_num - offset)){
                    // calculate the positions in the buffers
                    backward_modulo  = (i - offset - 1) % (2 * offset_max + 2); //maybe the top and the bottom different
                    forward_modulo = (x_num - 1) % (2 * offset_max + 2); //reached the end so need to use that

                    input.mesh[jxnumynum + i * y_num + k] = (temp_vec[forward_modulo * y_num + k] -
                                                             temp_vec[backward_modulo * y_num + k]) /
                                                            (x_num - i + offset);
                }
            }
        }
    }


}


template<typename T>
void APRReconstruction::calc_sat_adaptive_z(PixelData<T>& input, PixelData<uint8_t>& offset_img, float scale_in, int offset_max_in, const int d_max ){

    // The same, but in place

    offset_max_in = std::min(offset_max_in, (input.z_num/2 - 1));

    const int64_t z_num = input.z_num;
    const int64_t x_num = input.x_num;
    const int64_t y_num = input.y_num;

    int64_t j,k;
    float temp;
    int64_t index_modulo, previous_modulo, iynum,forward_modulo,backward_modulo,offset;
    int64_t xnumynum = x_num * y_num;

    const int offset_max = offset_max_in;
    const float scale = scale_in;
    //const unsigned int d_max = this->level_max();

    std::vector<float> temp_vec;
    temp_vec.resize(y_num*(2*offset_max + 2),0);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,k,temp,index_modulo, previous_modulo,backward_modulo,forward_modulo, iynum,offset) \
        firstprivate(temp_vec)
#endif
    for(int i = 0; i < x_num; i++) {

        iynum = i * y_num;

        //prefetching

        for(k = 0; k < y_num ; k++){
            // std::copy ?
            temp_vec[k] = input.mesh[iynum + k];
        }

        //(updated z)
        for(j = 1; j < 2 * offset_max+ 1; j++) {
            for(k = 0; k < y_num; k++) {
                temp_vec[j*y_num + k] = input.mesh[j * xnumynum + iynum + k] + temp_vec[(j-1)*y_num + k];
            }
        }

        // LHS boundary (updated)
        for(j = 0; j < offset_max + 1; j++){
            for(k = 0; k < y_num; k++) {
                offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[j * xnumynum + iynum + k])/scale),(T)offset_max);
                if(i < (offset + 1)) {
                    input.mesh[j * xnumynum + iynum + k] = (temp_vec[(j + offset) * y_num + k]) / (j + offset + 1);
                }
            }
        }

        // middle
        for(j = 1; j < z_num ; j++){

            for(k = 0; k < y_num; k++) {

                offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[j * xnumynum + iynum + k])/scale),(T)offset_max);

                //update the buffer
                if((j >= offset_max + 1) & (j < (z_num - offset_max))) {

                    index_modulo = (j + offset_max) % (2 * offset_max + 2);
                    previous_modulo = (j + offset_max - 1) % (2 * offset_max + 2);

                    // the current cumsum
                    temp = input.mesh[(j + offset_max) * xnumynum + iynum + k] + temp_vec[previous_modulo*y_num + k];
                    temp_vec[index_modulo*y_num + k] = temp;
                }

                if((j >= offset+ 1) & (j < (z_num - offset))) {
                    // calculate the positions in the buffers
                    forward_modulo = (j + offset) % (2 * offset_max + 2);
                    backward_modulo = (j - offset - 1) % (2 * offset_max + 2);

                    input.mesh[j * xnumynum + iynum + k] =
                            (temp_vec[forward_modulo * y_num + k] - temp_vec[backward_modulo * y_num + k]) /
                            (2 * offset + 1);

                }
            }
        }

        // RHS boundary

        for(j = z_num - offset_max; j < z_num; j++){
            for(k = 0; k < y_num; k++){

                offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[j * xnumynum + iynum + k])/scale),(T)offset_max);

                if(j >= (z_num - offset)){
                    //calculate the buffer offsets
                    backward_modulo  = (j - offset - 1) % (2 * offset_max + 2); //maybe the top and the bottom different
                    forward_modulo = (z_num - 1) % (2 * offset_max + 2); //reached the end so need to use that

                    input.mesh[j * xnumynum + iynum + k] = (temp_vec[forward_modulo*y_num + k] -
                                                            temp_vec[backward_modulo*y_num + k]) / (z_num - j + offset);

                }
            }

        }
    }

}


template<typename U,typename V>
void APRReconstruction::interp_img_us_smooth(APR& apr, PixelData<U>& img, ParticleData<V>& parts, bool smooth, int delta){
    //
    //  Bevan Cheeseman 2016
    //  Reconstruction via iterative upsampling, optionally with smoothing in each step
    //


    ParticleData<float> tree_parts;
    APRTreeNumerics::fill_tree_mean(apr,parts,tree_parts);

    MeshNumerics meshNumerics;
    std::vector<PixelData<float>> stencils;
    if(smooth) {
        meshNumerics.generate_smooth_stencil(stencils);
    }

    auto apr_iterator = apr.iterator();
    auto apr_tree_iterator = apr.tree_iterator();

    img.initWithValue(apr_iterator.y_num(apr.level_max() + delta), apr_iterator.x_num(apr.level_max() + delta), apr_iterator.z_num(apr.level_max() + delta), 0);

    std::vector<PixelData<U>> temp_imgs;
    temp_imgs.resize(apr.level_max()+1);

    for (int i = apr_iterator.level_min(); i < apr_iterator.level_max(); ++i) {
        temp_imgs[i].init(apr_iterator.y_num(i),apr_iterator.x_num(i),apr_iterator.z_num(i));
    }

    temp_imgs[apr.level_max()].swap(img);

    for (int level = apr_iterator.level_min(); level <= (apr_iterator.level_max()+delta); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (int x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator++) {

                    temp_imgs[level].at(apr_iterator.y(), x, z)=parts[apr_iterator];
                }
            }
        }

        if(level < apr.level_max()) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_tree_iterator)
#endif
            for (z = 0; z < apr_tree_iterator.z_num(level); z++) {
                for (x = 0; x < apr_tree_iterator.x_num(level); ++x) {
                    for (apr_tree_iterator.begin(level, z, x);
                         apr_tree_iterator < apr_tree_iterator.end();
                         apr_tree_iterator++) {

                        temp_imgs[level].at(apr_tree_iterator.y(), x, z) = tree_parts[apr_tree_iterator];
                    }
                }
            }
        }

        int curr_stencil = std::min((int)stencils.size()-1,(int)(apr.level_max()-level));

        if(smooth) {
            if(level!=apr.level_max()) {
                meshNumerics.apply_stencil(temp_imgs[level], stencils[curr_stencil]);
            }
        }

        if(level<(apr.level_max()+delta)) {
            const_upsample_img(temp_imgs[level+1], temp_imgs[level]);
        }
    }

    temp_imgs[apr.level_max()+delta].swap(img);

}


#endif //LIBAPR_APRRECONSTRUCTION_HPP
