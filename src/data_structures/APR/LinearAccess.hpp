//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_LINACCESS_HPP
#define PARTPLAY_LINACCESS_HPP

#include <map>
#include <utility>
#include <numeric>
#include "data_structures/Mesh/PixelData.hpp"

#include "algorithm/APRParameters.hpp"

#include "APRAccessStructures.hpp"

#include "GenInfo.hpp"

class LinearAccess : public GenAccess {

public:

    //New Linear Access Structures

    std::vector<uint16_t> y_vec;
    std::vector<uint64_t> xz_end_vec;
    std::vector<uint64_t> level_end_vec;
    std::vector<uint64_t> level_xz_vec;

protected:

    void initialize_xz_linear(){

        uint64_t counter_total = 1; //the buffer val

        level_end_vec.resize(level_max() + 1);

        for (int i = 0; i <= level_max(); ++i) {

            counter_total += x_num(i)*z_num(i);
            level_end_vec[i] = counter_total;

        }

        xz_end_vec.resize(counter_total,0);

    }

public:

    void initialize_linear_structure(APRParameters& apr_parameters,std::vector<PixelData<uint8_t>> &p_map);


};


inline void LinearAccess::initialize_linear_structure(APRParameters& apr_parameters,std::vector<PixelData<uint8_t>> &p_map) {
    /*
     * This function direclty intiitalizes the linear access data structure with explicit y.
     *
     * The algorithm logic has been designed such that it is portable to the GPU (i.e. pre-allocation of memory)
     *
     */

    genInfo->x_num.resize(genInfo->l_max+1);
    genInfo->y_num.resize(genInfo->l_max+1);
    genInfo->z_num.resize(genInfo->l_max+1);

    for(size_t i = genInfo->l_min;i < genInfo->l_max; ++i) {
        genInfo->x_num[i] = p_map[i].x_num;
        genInfo->y_num[i] = p_map[i].y_num;
        genInfo->z_num[i] = p_map[i].z_num;
    }

    genInfo-> y_num[genInfo->l_max] = genInfo->org_dims[0];
    genInfo->x_num[genInfo->l_max] = genInfo->org_dims[1];
    genInfo->z_num[genInfo->l_max] = genInfo->org_dims[2];

    genInfo->level_size.resize(level_max() + 1);
    for (int k = 0; k <= level_max(); ++k) {
        genInfo->level_size[k] = (uint64_t) pow(2,level_max() - k);
    }

    //
    // STEP.1 (Apply equivalence optimization, and then calculate the total number of particles required in each row to allow allocation of datastructures)
    //

    APRTimer apr_timer(true);

    uint8_t min_type = apr_parameters.neighborhood_optimization ? 1 : 2;

    initialize_xz_linear();

    // ========================================================================
    apr_timer.start_timer("first_step");

    const uint8_t UPSAMPLING_SEED_TYPE = 4;
    const uint8_t seed_us = UPSAMPLING_SEED_TYPE; //deal with the equivalence optimization
    for (size_t i = level_min()+1; i < level_max(); ++i) {
        const size_t xLen = genInfo->x_num[i];
        const size_t zLen = genInfo->z_num[i];
        const size_t yLen = genInfo->y_num[i];
        const size_t xLenUpsampled = genInfo->x_num[i - 1];
        const size_t yLenUpsampled = genInfo->y_num[i - 1];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_part_map_ds = (x / 2) * yLenUpsampled + (z / 2) * yLenUpsampled * xLenUpsampled;
                const size_t offset_part_map = x * yLen + z * yLen * xLen;

                for (size_t y = 0; y < yLenUpsampled; ++y) {
                    uint8_t status = p_map[i - 1].mesh[offset_part_map_ds + y];

                    if (status > 0 && status <= min_type) {
                        size_t y2p = std::min(2*y+1,yLen-1);
                        p_map[i].mesh[offset_part_map + 2 * y] = seed_us;
                        p_map[i].mesh[offset_part_map + y2p] = seed_us;
                    }
                }
            }
        }
    }
    apr_timer.stop_timer();

    // ========================================================================
    apr_timer.start_timer("second_step");


    for (size_t i = (level_min());i < (level_max()-1); ++i) {
        const size_t xLen = genInfo->x_num[i];
        const size_t zLen = genInfo->z_num[i];
        const size_t yLen = genInfo->y_num[i];

        const auto level_start = level_end_vec[i-1];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_pc_data = z * xLen + x;
                const size_t offset_part_map = yLen * offset_pc_data;

                uint64_t counter = 0;

                for (size_t y = 0; y < yLen; ++y) {
                    uint8_t status = p_map[i].mesh[offset_part_map + y];
                    if ((status > min_type) && (status <= UPSAMPLING_SEED_TYPE)) {
                        counter++;
                    }

                }

                xz_end_vec[level_start + offset_pc_data] = counter;

            }
        }
    }

    std::vector<uint64_t> temp_max_xz;
    temp_max_xz.resize(genInfo->z_num[genInfo->l_max - 1]*genInfo->x_num[genInfo->l_max - 1],0);

    /*
     * l_max - 1 is special as it also has the l_max information that then needs to be upsampled.
     *
     */

    size_t i = genInfo->l_max - 1;
    const size_t xLen = genInfo->x_num[i];
    const size_t zLen = genInfo->z_num[i];
    const size_t yLen = genInfo->y_num[i];

    const size_t yLen_m = genInfo->y_num[i+1];

    auto level_start = level_end_vec[i-1];

#ifdef HAVE_OPENMP
#pragma omp parallel for  schedule(dynamic) default(shared)
#endif
    for (size_t z = 0; z < zLen; ++z) {
        for (size_t x = 0; x < xLen; ++x) {
            const size_t offset_pc_data = z * xLen + x;
            const size_t offset_part_map = yLen * offset_pc_data;

            uint64_t counter = 0;
            uint64_t counter_l = 0;

            for (size_t y = 0; y < yLen; ++y) {
                uint8_t status = p_map[i].mesh[offset_part_map + y];
                if ((status > min_type) && (status <= UPSAMPLING_SEED_TYPE)) {
                    counter++;
                }
                else if (status > 0 && status <= min_type) {
                    counter_l++;
                    if(2*y<(yLen_m-1)){
                        counter_l++;
                    }
                }
            }

            xz_end_vec[level_start + offset_pc_data] = counter;
            temp_max_xz[offset_pc_data] = counter_l;

        }
    }

    /*
     * Now need to copy across the values for the level_max
     */


    const size_t xLen_m = genInfo->x_num[i+1];
    const size_t zLen_m = genInfo->z_num[i+1];

    level_start = level_end_vec[i];

#ifdef HAVE_OPENMP
#pragma omp parallel  for default(shared) schedule(dynamic)
#endif
    for (size_t z = 0; z < zLen_m; ++z) {
        for (size_t x = 0; x < xLen_m; ++x) {
            const size_t offset_pc_data_m = z * xLen_m + x;
            const size_t offset_pc_data = (z/2) * xLen + x/2;

            xz_end_vec[level_start + offset_pc_data_m] = temp_max_xz[offset_pc_data];

        }
    }


    apr_timer.stop_timer();

    //
    //  Serial Portion (Cumulative Sum)
    //

    apr_timer.start_timer("serial cumsum");

    //now run over and create the sum
    std::partial_sum(xz_end_vec.begin(),xz_end_vec.end(),xz_end_vec.begin());

    apr_timer.stop_timer();

    apr_timer.start_timer("init y");

    genInfo->total_number_particles = xz_end_vec.back();

    y_vec.resize(genInfo->total_number_particles);

    apr_timer.stop_timer();

    apr_timer.start_timer("get y");

    //
    // STEP.2 Now get the y-values.
    //

    for (size_t i = (level_min());i < (level_max()-1); ++i) {
        const size_t xLen = genInfo->x_num[i];
        const size_t zLen = genInfo->z_num[i];
        const size_t yLen = genInfo->y_num[i];

        const auto level_start = level_end_vec[i-1];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_pc_data = z * xLen + x;
                const size_t offset_part_map = yLen * offset_pc_data;

                uint64_t counter = 0;
                auto offset_y = xz_end_vec[level_start + offset_pc_data-1];

                for (uint16_t y = 0; y < yLen; ++y) {
                    uint8_t status = p_map[i].mesh[offset_part_map + y];
                    if ((status > min_type) && (status <= UPSAMPLING_SEED_TYPE)) {
                        y_vec[counter + offset_y] = y;
                        counter++;
                    }
                }
            }
        }
    }

    /*
     * l_max - 1 is special as it also has the l_max information that then needs to be upsampled.
     *
     */

    const auto level_start_m = level_end_vec[i-1];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
    for (size_t z = 0; z < zLen; ++z) {
        for (size_t x = 0; x < xLen; ++x) {
            const size_t offset_pc_data = z * xLen + x;

            const size_t offset_pc_data_m = (z*2) * xLen_m + x*2;

            const size_t offset_part_map = yLen * offset_pc_data;

            uint64_t counter = 0;
            uint64_t counter_l = 0;

            auto offset_y = xz_end_vec[level_start + offset_pc_data-1];
            auto offset_y_m = xz_end_vec[level_start_m + offset_pc_data_m-1];

            for (uint16_t y = 0; y < yLen; ++y) {
                uint8_t status = p_map[i].mesh[offset_part_map + y];
                if ((status > min_type) && (status <= UPSAMPLING_SEED_TYPE)) {
                    y_vec[counter + offset_y] = y;
                    counter++;
                }
                else if (status > 0 && status <= min_type) {
                    y_vec[counter_l + offset_y_m] = 2*y;
                    counter_l++;
                    if(2*y<(yLen_m-1)){
                        y_vec[counter_l + offset_y_m] = 2*y+1;
                        counter_l++;
                    }
                }
            }
        }
    }

    apr_timer.stop_timer();

    apr_timer.start_timer("max y");

    //now need to spread the maximum level y
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
    for (size_t z = 0; z < zLen_m; ++z) {
        for (size_t x = 0; x < xLen_m; ++x) {

            //first check if its not already there.
            if(((z % 2) != 0) || ((x % 2) != 0)) {

                const size_t offset_pc_data_m = z * xLen_m + x;

                const size_t offset_pc_data_m_f = (z/2)*2 * xLen_m + (x/2)*2;
                auto offset_y_b_f = xz_end_vec[level_start_m + offset_pc_data_m_f - 1];

                auto offset_y_b = xz_end_vec[level_start_m + offset_pc_data_m - 1];
                auto offset_y_e = xz_end_vec[level_start_m + offset_pc_data_m];

                //now we can loop over only the gaps
                for (int j = 0; j < (offset_y_e-offset_y_b); ++j) {
                    y_vec[offset_y_b + j] = y_vec[offset_y_b_f + j];
                }
            }
        }
    }

    apr_timer.stop_timer();

}


#endif //PARTPLAY_APRACCESS_HPP
