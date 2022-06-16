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

#include "../GenInfo.hpp"
#include "GenAccess.hpp"

class LinearAccess : public GenAccess {

public:


    void copy(const LinearAccess& access2copy){
        y_vec.copy(access2copy.y_vec);
        xz_end_vec.copy(access2copy.xz_end_vec);
        level_xz_vec.copy(access2copy.level_xz_vec);
    }

    //New Linear Access Structures

    VectorData<uint16_t> y_vec; // explicit storage of the sparse dimension (y)
    VectorData<uint64_t> xz_end_vec; // total number of particles up to and including the current sparse row
    VectorData<uint64_t> level_xz_vec; // the starting location of each level in the xz_end_vec structure

    void initialize_xz_linear(){

        uint64_t counter_total = 1; //the buffer val to allow -1 calls without checking.

        level_xz_vec.resize(level_max()+2,0); //includes a buffer for -1 calls, and therefore needs to be called with level + 1;

        level_xz_vec[0] = 1; //allowing for the offset.

        for (int i = 0; i <= level_max(); ++i) {

            counter_total += x_num(i)*z_num(i);
            level_xz_vec[i+1] = counter_total;

        }

        xz_end_vec.resize(counter_total,0);

    }

    void initialize_linear_structure(APRParameters& apr_parameters,std::vector<PixelData<uint8_t>> &p_map);

    void initialize_tree_access_sparse(std::vector<std::vector<SparseParticleCellMap>> &p_map);

    void initialize_linear_structure_sparse(APRParameters& apr_parameters,SparseGaps<SparseParticleCellMap>& p_map);

    void initialize_tree_access_dense(std::vector<PixelData<uint8_t>> &p_map);

};


inline void LinearAccess::initialize_tree_access_dense(std::vector<PixelData<uint8_t>> &p_map) {
    APRTimer timer(false);

    initialize_xz_linear();

    // fill number of particles in each row
    timer.start_timer("init tree dense - fill number particles per row");
    for(int level = level_max(); level >= level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int z = 0; z < z_num(level); ++z) {
            for(int x = 0; x < x_num(level); ++x) {
                size_t offset = (size_t) z * x_num(level) * y_num(level) + (size_t) x * y_num(level);
                size_t counter = 0;

                for(int y = 0; y < y_num(level); ++y) {
                    counter += p_map[level].mesh[offset + y];
                }
                size_t row_index = level_xz_vec[level] + z * x_num(level) + x;
                xz_end_vec[row_index] = counter;
            }
        }
    }
    timer.stop_timer();

    timer.start_timer("init tree dense - cumulative sum");
    std::partial_sum(xz_end_vec.begin(),xz_end_vec.end(),xz_end_vec.begin());
    timer.stop_timer();

    // set total number of particles
    genInfo->total_number_particles = xz_end_vec.back();

    timer.start_timer("init tree dense - allocate and fill y_vec");
    y_vec.resize(genInfo->total_number_particles);

    for(int level = level_max(); level >= level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int z = 0; z < z_num(level); ++z) {
            for (int x = 0; x < x_num(level); ++x) {
                size_t offset = (size_t) z * x_num(level) * y_num(level) + (size_t) x * y_num(level);
                size_t row_index = level_xz_vec[level] + z * x_num(level) + x;
                size_t particle_index = xz_end_vec[row_index-1];

                for(int y = 0; y < y_num(level); ++y) {
                    if(p_map[level].mesh[offset + y]) {
                        y_vec[particle_index++] = y;
                    }
                }
            }
        }
    }
    timer.stop_timer();
}


inline void LinearAccess::initialize_tree_access_sparse(std::vector<std::vector<SparseParticleCellMap>> &p_map) {
    APRTimer apr_timer(false);

    //initialize loop variables
    uint64_t x_;
    uint64_t z_;

    initialize_xz_linear();

    apr_timer.start_timer("create gaps");

    for(int i = (level_min());i <= level_max();i++) {

        const uint64_t x_num_ = genInfo->x_num[i];
        const uint64_t z_num_ = genInfo->z_num[i];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared) private(z_, x_)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {
            for (x_ = 0; x_ < x_num_; x_++) {

                const size_t offset_pc_data = x_num_ * z_ + x_;

                const auto level_start = level_xz_vec[i];

                auto &map = p_map[i][offset_pc_data].mesh;

                uint64_t counter = 0;

                for(auto it = map.begin(); it != map.end(); ++it) {
                    counter++;
                }

                xz_end_vec[level_start + offset_pc_data] = counter;
            }

        }
    }

    apr_timer.stop_timer();

    std::partial_sum(xz_end_vec.begin(),xz_end_vec.end(),xz_end_vec.begin());

    genInfo->total_number_particles = xz_end_vec.back();

    y_vec.resize(genInfo->total_number_particles);

    apr_timer.start_timer("create gaps");

    for(int i = (level_min());i <= level_max();i++) {

        const uint64_t x_num_ = genInfo->x_num[i];
        const uint64_t z_num_ = genInfo->z_num[i];

        const auto level_start = level_xz_vec[i];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared) private(z_, x_)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {
            for (x_ = 0; x_ < x_num_; x_++) {

                const size_t offset_pc_data = x_num_ * z_ + x_;

                uint16_t counter = 0;

                auto &map = p_map[i][offset_pc_data].mesh;

                auto offset_y = xz_end_vec[level_start + offset_pc_data-1];

                for (auto it = map.begin(); it != map.end(); ++it) {
                    const auto y = it->first;
                    y_vec[counter + offset_y] = y;
                    counter++;
                }
            }
        }
    }

    apr_timer.stop_timer();

}


inline void LinearAccess::initialize_linear_structure(APRParameters& apr_parameters,std::vector<PixelData<uint8_t>> &p_map) {
    /*
     * This function direclty intiitalizes the linear access data structure with explicit y.
     *
     * The algorithm logic has been designed such that it is portable to the GPU (i.e. pre-allocation of memory)
     *
     */

    //
    // STEP.1 (Apply equivalence optimization, and then calculate the total number of particles required in each row to allow allocation of datastructures)
    //

    APRTimer apr_timer(false);

    uint8_t min_type = apr_parameters.neighborhood_optimization ? 1 : 2;

    initialize_xz_linear();

    //edge case
    if(level_max()<=2){
        // For performance reasons and clarity of the code, it doesn't make sense here to handle these cases. Below assumes there is atleast levels <=2;

        //just initialize full resolution
        const auto level_start = level_xz_vec[level_max()];
        uint64_t counter = 0;
        for (int z = 0; z < z_num(level_max()); ++z) {
            for (int x = 0; x < x_num(level_max()); ++x) {
                const size_t offset_pc_data = z * x_num(level_max()) + x;
                for (int y = 0; y < y_num(level_max()); ++y) {

                    counter++;
                }
                xz_end_vec[level_start + offset_pc_data] = counter;
            }
        }
        y_vec.resize(counter);
        counter = 0;

        for (int z = 0; z < z_num(level_max()); ++z) {
            for (int x = 0; x < x_num(level_max()); ++x) {

                for (int y = 0; y < y_num(level_max()); ++y) {
                    y_vec[counter] = y;
                    counter++;
                }
            }
        }


        return;
    }

    // ========================================================================
    apr_timer.start_timer("first_step");

    const uint8_t UPSAMPLING_SEED_TYPE = 4;
    const uint8_t seed_us = UPSAMPLING_SEED_TYPE; //deal with the equivalence optimization
    for (int level = level_min()+1; level < level_max(); ++level) {
        const size_t xLen = genInfo->x_num[level];
        const size_t zLen = genInfo->z_num[level];
        const size_t yLen = genInfo->y_num[level];
        const size_t xLenUpsampled = genInfo->x_num[level - 1];
        const size_t yLenUpsampled = genInfo->y_num[level - 1];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_part_map_ds = (x / 2) * yLenUpsampled + (z / 2) * yLenUpsampled * xLenUpsampled;
                const size_t offset_part_map = x * yLen + z * yLen * xLen;

                for (size_t y = 0; y < yLenUpsampled; ++y) {
                    uint8_t status = p_map[level - 1].mesh[offset_part_map_ds + y];

                    if (status > 0 && status <= min_type) {
                        size_t y2p = std::min(2*y+1,yLen-1);
                        p_map[level].mesh[offset_part_map + 2 * y] = seed_us;
                        p_map[level].mesh[offset_part_map + y2p] = seed_us;
                    }
                }
            }
        }
    }
    apr_timer.stop_timer();

    // ========================================================================
    apr_timer.start_timer("second_step");


    for (int level = (level_min());level < (level_max()-1); ++level) {
        const size_t xLen = genInfo->x_num[level];
        const size_t zLen = genInfo->z_num[level];
        const size_t yLen = genInfo->y_num[level];

        const auto level_start = level_xz_vec[level];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_pc_data = z * xLen + x;
                const size_t offset_part_map = yLen * offset_pc_data;

                uint64_t counter = 0;

                for (size_t y = 0; y < yLen; ++y) {
                    uint8_t status = p_map[level].mesh[offset_part_map + y];
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

    size_t l_minus_1 = genInfo->l_max - 1;
    const size_t xLen = genInfo->x_num[l_minus_1];
    const size_t zLen = genInfo->z_num[l_minus_1];
    const size_t yLen = genInfo->y_num[l_minus_1];

    const size_t yLen_m = genInfo->y_num[l_minus_1+1];

    auto level_start_minus_1 = level_xz_vec[l_minus_1];

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
                uint8_t status = p_map[l_minus_1].mesh[offset_part_map + y];
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

            xz_end_vec[level_start_minus_1 + offset_pc_data] = counter;
            temp_max_xz[offset_pc_data] = counter_l;

        }
    }

    /*
     * Now need to copy across the values for the level_max
     */

    const size_t xLen_m = genInfo->x_num[level_max()];
    const size_t zLen_m = genInfo->z_num[level_max()];
    auto level_start_m = level_xz_vec[level_max()];

#ifdef HAVE_OPENMP
#pragma omp parallel  for default(shared) schedule(dynamic)
#endif
    for (size_t z = 0; z < zLen_m; ++z) {
        for (size_t x = 0; x < xLen_m; ++x) {
            const size_t offset_pc_data_m = z * xLen_m + x;
            const size_t offset_pc_data = (z/2) * xLen + x/2;

            xz_end_vec[level_start_m + offset_pc_data_m] = temp_max_xz[offset_pc_data];

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

    y_vec.resize(genInfo->total_number_particles,0);

    apr_timer.stop_timer();

    apr_timer.start_timer("get y");

    //
    // STEP.2 Now get the y-values.
    //

    for (int level = (level_min());level < (level_max()-1); ++level) {
        const size_t xLen = genInfo->x_num[level];
        const size_t zLen = genInfo->z_num[level];
        const size_t yLen = genInfo->y_num[level];

        const auto level_start = level_xz_vec[level];

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
                    uint8_t status = p_map[level].mesh[offset_part_map + y];
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


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
    for (size_t z = 0; z < zLen; ++z) {
        for (size_t x = 0; x < xLen; ++x) {
            const size_t offset_pc_data = z * xLen + x;

            const size_t offset_pc_data_m = (z*2) * xLen_m + x*2; //max level
            const size_t offset_part_map = yLen * offset_pc_data; //current level

            uint64_t counter = 0;
            uint64_t counter_l = 0;

            auto offset_y = xz_end_vec[level_start_minus_1 + offset_pc_data-1];
            auto offset_y_m = xz_end_vec[level_start_m + offset_pc_data_m-1];

            for (uint16_t y = 0; y < yLen; ++y) {
                uint8_t status = p_map[l_minus_1].mesh[offset_part_map + y];
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
                auto offset_y_e_f = xz_end_vec[level_start_m + offset_pc_data_m_f];
                auto offset_y_b = xz_end_vec[level_start_m + offset_pc_data_m - 1];

                std::copy(y_vec.begin() + offset_y_b_f,y_vec.begin() + offset_y_e_f,y_vec.begin() + offset_y_b);
            }
        }
    }

    apr_timer.stop_timer();

}


inline void LinearAccess::initialize_linear_structure_sparse(APRParameters& apr_parameters,SparseGaps<SparseParticleCellMap>& p_map) {
    /*
     * This function direclty intiitalizes the linear access data structure with explicit y.
     *
     * The algorithm logic has been designed such that it is portable to the GPU (i.e. pre-allocation of memory)
     *
     */

    //
    // STEP.1 (Apply equivalence optimization, and then calculate the total number of particles required in each row to allow allocation of datastructures)
    //

    APRTimer apr_timer(false);

    uint8_t min_type = apr_parameters.neighborhood_optimization ? 1 : 2;

    initialize_xz_linear();

    // ========================================================================
    apr_timer.start_timer("first_step");

    const uint8_t UPSAMPLING_SEED_TYPE = 4;
    const uint8_t seed_us = UPSAMPLING_SEED_TYPE; //deal with the equivalence optimization
    for (int level = level_min()+1; level < level_max(); ++level) {
        const size_t xLen = genInfo->x_num[level];
        const size_t zLen = genInfo->z_num[level];
        const size_t yLen = genInfo->y_num[level];
        const size_t xLenUpsampled = genInfo->x_num[level - 1];


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_part_map_ds = (x / 2)  + (z / 2) * xLenUpsampled;
                const size_t offset_part_map = x + z * xLen;

                auto& mesh_ds = p_map.data[level-1][offset_part_map_ds][0].mesh;

                //SPARSE iteration
                for (auto it=mesh_ds.begin(); it!=mesh_ds.end(); ++it){
                    size_t y = it->first;
                    uint8_t status = it->second;

                    if (status > 0 && status <= min_type) {
                        uint16_t y2p = std::min(2*y+1,yLen-1);

                        p_map.data[level][offset_part_map][0].mesh[ 2 * y] = seed_us;
                        p_map.data[level][offset_part_map][0].mesh[ y2p] = seed_us;

                    }
                }

            }
        }
    }
    apr_timer.stop_timer();

    // ========================================================================
    apr_timer.start_timer("second_step");


    for (int level = (level_min());level < (level_max()-1); ++level) {
        const size_t xLen = genInfo->x_num[level];
        const size_t zLen = genInfo->z_num[level];

        const auto level_start = level_xz_vec[level];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_pc_data = z * xLen + x;

                uint64_t counter = 0;

                auto& mesh_ds = p_map.data[level][offset_pc_data][0].mesh;

                //SPARSE iteration
                for (auto it=mesh_ds.begin(); it!=mesh_ds.end(); ++it){

                    uint8_t status = it->second;

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

    size_t l_minus_1 = genInfo->l_max - 1;
    const size_t xLen = genInfo->x_num[l_minus_1];
    const size_t zLen = genInfo->z_num[l_minus_1];


    const size_t yLen_m = genInfo->y_num[l_minus_1+1];

    auto level_start_minus_1 = level_xz_vec[l_minus_1];

#ifdef HAVE_OPENMP
#pragma omp parallel for  schedule(dynamic) default(shared)
#endif
    for (size_t z = 0; z < zLen; ++z) {
        for (size_t x = 0; x < xLen; ++x) {
            const size_t offset_pc_data = z * xLen + x;

            uint64_t counter = 0;
            uint64_t counter_l = 0;

            auto& mesh_ds = p_map.data[l_minus_1][offset_pc_data][0].mesh;

            //SPARSE iteration
            for (auto it=mesh_ds.begin(); it!=mesh_ds.end(); ++it){

                uint8_t status = it->second;
                size_t y = it->first;

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


            xz_end_vec[level_start_minus_1 + offset_pc_data] = counter;
            temp_max_xz[offset_pc_data] = counter_l;

        }
    }

    /*
     * Now need to copy across the values for the level_max
     */


    const size_t xLen_m = genInfo->x_num[level_max()];
    const size_t zLen_m = genInfo->z_num[level_max()];

    auto level_start_m = level_xz_vec[level_max()];

#ifdef HAVE_OPENMP
#pragma omp parallel  for default(shared) schedule(dynamic)
#endif
    for (size_t z = 0; z < zLen_m; ++z) {
        for (size_t x = 0; x < xLen_m; ++x) {
            const size_t offset_pc_data_m = z * xLen_m + x;
            const size_t offset_pc_data = (z/2) * xLen + x/2;

            xz_end_vec[level_start_m + offset_pc_data_m] = temp_max_xz[offset_pc_data];

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

    for (int level = (level_min());level < (level_max()-1); ++level) {
        const size_t xLen = genInfo->x_num[level];
        const size_t zLen = genInfo->z_num[level];

        const auto level_start = level_xz_vec[level];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_pc_data = z * xLen + x;

                uint64_t counter = 0;
                auto offset_y = xz_end_vec[level_start + offset_pc_data-1];

                auto& mesh_ds = p_map.data[level][offset_pc_data][0].mesh;

                //SPARSE iteration
                for (auto it=mesh_ds.begin(); it!=mesh_ds.end(); ++it) {

                    uint8_t status = it->second;
                    uint16_t y = it->first;

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

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
    for (size_t z = 0; z < zLen; ++z) {
        for (size_t x = 0; x < xLen; ++x) {
            const size_t offset_pc_data = z * xLen + x;

            const size_t offset_pc_data_m = (z*2) * xLen_m + x*2; //max level

            uint64_t counter = 0;
            uint64_t counter_l = 0;

            auto offset_y = xz_end_vec[level_start_minus_1 + offset_pc_data-1];
            auto offset_y_m = xz_end_vec[level_start_m + offset_pc_data_m-1];

            auto& mesh_ds = p_map.data[l_minus_1][offset_pc_data][0].mesh;

            //SPARSE iteration
            for (auto it=mesh_ds.begin(); it!=mesh_ds.end(); ++it) {

                uint8_t status = it->second;
                size_t y = it->first;

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
                auto offset_y_e_f = xz_end_vec[level_start_m + offset_pc_data_m_f];
                auto offset_y_b = xz_end_vec[level_start_m + offset_pc_data_m - 1];

                std::copy(y_vec.begin() + offset_y_b_f,y_vec.begin() + offset_y_e_f,y_vec.begin() + offset_y_b);
            }
        }
    }

    apr_timer.stop_timer();

}



#endif //PARTPLAY_APRACCESS_HPP
