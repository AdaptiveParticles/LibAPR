//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APRACCESS_HPP
#define PARTPLAY_APRACCESS_HPP


#include <map>
#include <utility>
#include "data_structures/Mesh/PixelData.hpp"
#include "ExtraParticleData.hpp"
#include "ExtraPartCellData.hpp"
#include "algorithm/APRParameters.hpp"

#include "APRAccessStructures.hpp"

class APRAccess {

public:

    uint64_t l_min;
    uint64_t l_max;
    uint64_t org_dims[3]={0,0,0};

    uint8_t number_dimensions = 3;

    // TODO: SHould they be also saved as uint64 in HDF5? (currently int is used)
    std::vector<uint64_t> x_num;
    std::vector<uint64_t> y_num;
    std::vector<uint64_t> z_num;

    uint64_t total_number_particles;
    uint64_t total_number_gaps;
    std::vector<uint64_t> global_index_by_level_begin;                   // note: 0..n-1 based numbering
    std::vector<uint64_t> global_index_by_level_end;                     // note: 0..n-1 based numbering
    std::vector<std::vector<uint64_t>> global_index_by_level_and_zx_end; // note: 1..n based numbering

    uint64_t total_number_non_empty_rows;
    ExtraPartCellData<ParticleCellGapMap> gap_map;

    uint64_t level_max() const { return l_max; }
    uint64_t level_min() const { return l_min; }
    uint64_t spatial_index_x_max(const unsigned int level) const { return x_num[level]; }
    uint64_t spatial_index_y_max(const unsigned int level) const { return y_num[level]; }
    uint64_t spatial_index_z_max(const unsigned int level) const { return z_num[level]; }

    inline bool get_neighbour_coordinate(const ParticleCell& input,ParticleCell& neigh,const unsigned int& face,const uint16_t& level_delta,const uint16_t& index){

        static constexpr int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
        static constexpr int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
        static constexpr int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

        static constexpr uint8_t children_index_offsets[4][2] = {{0,0},{1,0},{0,1},{1,1}};

        unsigned int dir;

        switch (level_delta){
            case _LEVEL_SAME:
                //Same Level Particle Cell
                neigh.x = input.x + dir_x[face];
                neigh.y = input.y + dir_y[face];
                neigh.z = input.z + dir_z[face];
                neigh.level = input.level;

                if(neigh.level < level_max()) {
                    neigh.pc_offset = gap_map.x_num[neigh.level] * neigh.z + neigh.x;
                }
                else {
                    neigh.pc_offset = gap_map.x_num[neigh.level] * (neigh.z/2) + (neigh.x/2);
                }

                return true;
            case _LEVEL_DECREASE:
                //Larger Particle Cell (Lower Level)
                neigh.level = input.level - 1;
                neigh.x = (input.x+ dir_x[face])/2;
                neigh.y = (input.y+ dir_y[face])/2;
                neigh.z = (input.z+ dir_z[face])/2;

                neigh.pc_offset =  gap_map.x_num[neigh.level] * neigh.z + neigh.x;

                return true;
            case _LEVEL_INCREASE:
                //Higher Level Particle Cell (Smaller/Higher Resolution), there is a maximum of 4 (conditional on boundary conditions)
                neigh.level = input.level + 1;
                neigh.x = (input.x + dir_x[face])*2 + (dir_x[face]<0);
                neigh.y = (input.y + dir_y[face])*2 + (dir_y[face]<0);
                neigh.z = (input.z + dir_z[face])*2 + (dir_z[face]<0);

                dir = (face/2);

                switch (dir){
                    case 0:
                        //y+ and y-
                        neigh.x = neigh.x + children_index_offsets[index][0];
                        neigh.z = neigh.z + children_index_offsets[index][1];

                        break;

                    case 1:
                        //x+ and x-
                        neigh.y = neigh.y + children_index_offsets[index][0];
                        neigh.z = neigh.z + children_index_offsets[index][1];

                        break;
                    case 2:
                        //z+ and z-
                        neigh.y = neigh.y + children_index_offsets[index][0];
                        neigh.x = neigh.x + children_index_offsets[index][1];

                        break;
                }

                if(neigh.level < level_max()) {
                    neigh.pc_offset = gap_map.x_num[neigh.level] * neigh.z + neigh.x;
                }
                else {
                    neigh.pc_offset = gap_map.x_num[neigh.level] * (neigh.z/2) + (neigh.x/2);
                }

                return true;
            case _NO_NEIGHBOUR:

                return false;
        }

        return false;
    }

    inline bool check_neighbours_flag(const uint16_t& x,const uint16_t& z,const uint16_t& level){
        // 0 1 2 .............. x_num-3 x_num-2 x_num-1 (x_num)
        //                              ......(x-1)..........
        //                                      ..(x)........
        return ((uint16_t)(x-1)>(x_num[level]-3)) | ((uint16_t)(z-1)>(z_num[level]-3));
    }

    /*
    inline uint8_t number_neighbours_in_direction(const uint8_t& level_delta){
        //
        //  Gives the maximum number of neighbours in a direction given the level_delta.
        //

        switch (level_delta){

            case _NO_NEIGHBOUR:
                return 0;

            case _LEVEL_INCREASE:

                switch (number_dimensions) {
                    case 1:
                        return 1;
                    case 2:
                        return 2;
                    case 3:
                        return 4;
                }
        }
        return 1;
    }
    */

    bool find_particle_cell(ParticleCell& part_cell,MapIterator& map_iterator){

        if(part_cell.pc_offset > gap_map.data[part_cell.level].size()) { // out of bounds
            return false;
        }

        if(gap_map.data[part_cell.level][part_cell.pc_offset].size() > 0) {

            ParticleCellGapMap& current_pc_map = gap_map.data[part_cell.level][part_cell.pc_offset][0];

            //this is required due to utilization of the equivalence optimization

            if((map_iterator.pc_offset != part_cell.pc_offset) || (map_iterator.level != part_cell.level) ){
                map_iterator.iterator = gap_map.data[part_cell.level][part_cell.pc_offset][0].map.begin();
                map_iterator.pc_offset = part_cell.pc_offset;
                map_iterator.level = part_cell.level;

                if(part_cell.pc_offset == 0){
                    if(part_cell.level == level_min()){
                        map_iterator.global_offset = 0;
                    } else {
                        map_iterator.global_offset = global_index_by_level_and_zx_end[part_cell.level-1].back();
                    }
                } else {
                    map_iterator.global_offset = global_index_by_level_and_zx_end[part_cell.level][part_cell.pc_offset-1];
                }


                if(part_cell.level == level_max()) {

                    auto it = (gap_map.data[part_cell.level][part_cell.pc_offset][0].map.rbegin());

                    map_iterator.max_offset = ((it->second.global_index_begin_offset + (it->second.y_end - it->first)) + 1 -
                                               map_iterator.iterator->second.global_index_begin_offset);

                    map_iterator.max_offset = ((it->second.global_index_begin_offset + (it->second.y_end-it->first))+1);

                } else {
                    map_iterator.max_offset = 0;
                }
            }

            uint64_t offset = 0;
            //deals with the different xz in the same access tree at highest resolution
            if(part_cell.level == level_max()) {
                offset = ((part_cell.x % 2) + (part_cell.z % 2) * 2)*map_iterator.max_offset +
                         map_iterator.global_offset;
            } else {
                offset = map_iterator.global_offset;
            }


            if(map_iterator.iterator == current_pc_map.map.end()){
                //check if pointing to a valid key
                map_iterator.iterator = current_pc_map.map.begin();
            }



            if ((part_cell.y >= map_iterator.iterator->first) && (part_cell.y <= map_iterator.iterator->second.y_end)) {
                // already pointing to the correct place
                part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset +
                                         (part_cell.y - map_iterator.iterator->first) + offset; //#fixme

                return true;
            } else {

                //first try next element
                //if(map_iterator.iterator != current_pc_map.map.end()){
                map_iterator.iterator++;
                //check if there
                if(map_iterator.iterator != current_pc_map.map.end()) {
                    if ((part_cell.y >= map_iterator.iterator->first) &
                        (part_cell.y <= map_iterator.iterator->second.y_end)) {
                        // already pointing to the correct place
                        part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset +
                                                 (part_cell.y - map_iterator.iterator->first) + offset; //#fixme

                        return true;
                    }
                }

                //otherwise search for it (points to first key that is greater than the y value)
                map_iterator.iterator = current_pc_map.map.upper_bound(part_cell.y);

                if((map_iterator.iterator == current_pc_map.map.begin()) || (map_iterator.iterator == current_pc_map.map.end())){
                    //less then the first value //check the last element

                    map_iterator.iterator = current_pc_map.map.end();

                }

                    map_iterator.iterator--;

                if ((part_cell.y >= map_iterator.iterator->first) & (part_cell.y <= map_iterator.iterator->second.y_end)) {
                    // already pointing to the correct place
                    part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset +
                                             (part_cell.y - map_iterator.iterator->first) + offset; //#fixme

                    return true;
                }
            }
        }

        return false;
    }

    bool find_particle_cell_tree(APRAccess &apr_access,ParticleCell& part_cell,MapIterator& map_iterator){
        if(part_cell.level == level_max()){

            if (apr_access.gap_map.data[part_cell.level+1][part_cell.pc_offset].size() > 0) {
                //shared access data
                ParticleCellGapMap &current_pc_map = apr_access.gap_map.data[part_cell.level+1][part_cell.pc_offset][0];

                //this is required due to utilization of the equivalence optimization

                if ((map_iterator.pc_offset != part_cell.pc_offset) || (map_iterator.level != part_cell.level)) {
                    map_iterator.iterator = apr_access.gap_map.data[part_cell.level+1][part_cell.pc_offset][0].map.begin();
                    map_iterator.pc_offset = part_cell.pc_offset;
                    map_iterator.level = part_cell.level;

                    if (part_cell.pc_offset == 0) {
                        if (part_cell.level == level_min()) {
                            map_iterator.global_offset = 0;
                        } else {
                            map_iterator.global_offset = global_index_by_level_and_zx_end[part_cell.level - 1].back();
                        }
                    } else {
                        map_iterator.global_offset = global_index_by_level_and_zx_end[part_cell.level][
                                part_cell.pc_offset - 1];
                    }

                    map_iterator.max_offset = 0;
                }

                uint64_t offset = 0;
                //deals with the different xz in the same access tree at highest resolution
                offset = map_iterator.global_offset;


                if (map_iterator.iterator == current_pc_map.map.end()) {
                    //check if pointing to a valid key
                    map_iterator.iterator = current_pc_map.map.begin();
                }

                if ((2*part_cell.y >= map_iterator.iterator->first) &&
                    (2*part_cell.y <= map_iterator.iterator->second.y_end)) {
                    // already pointing to the correct place
                    part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset/2 +
                                             (part_cell.y - map_iterator.iterator->first)/2 + offset;
                    return true;
                } else {
                    //first try next element
                    //if(map_iterator.iterator != current_pc_map.map.end()){
                    map_iterator.iterator++;
                    //check if there
                    if (map_iterator.iterator != current_pc_map.map.end()) {
                        if ((2*part_cell.y >= map_iterator.iterator->first) &
                            (2*part_cell.y <= map_iterator.iterator->second.y_end)) {
                            // already pointing to the correct place
                            part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset/2 +
                                                     (part_cell.y - map_iterator.iterator->first)/2 + offset;

                            return true;
                        }
                    }

                    //otherwise search for it (points to first key that is greater than the y value)
                    map_iterator.iterator = current_pc_map.map.upper_bound(2*part_cell.y);

                    if ((map_iterator.iterator == current_pc_map.map.begin()) ||
                        (map_iterator.iterator == current_pc_map.map.end())) {
                        //less then the first value
                        return false;
                    } else {
                        map_iterator.iterator--;
                    }

                    if ((2*part_cell.y >= map_iterator.iterator->first) &
                        (2*part_cell.y <= map_iterator.iterator->second.y_end)) {
                        // already pointing to the correct place
                        part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset/2 +
                                                 (part_cell.y - map_iterator.iterator->first)/2 + offset;

                        return true;
                    }
                }
            }
        } else {

            if (gap_map.data[part_cell.level][part_cell.pc_offset].size() > 0) {

                ParticleCellGapMap &current_pc_map = gap_map.data[part_cell.level][part_cell.pc_offset][0];

                //this is required due to utilization of the equivalence optimization

                if ((map_iterator.pc_offset != part_cell.pc_offset) || (map_iterator.level != part_cell.level)) {
                    map_iterator.iterator = gap_map.data[part_cell.level][part_cell.pc_offset][0].map.begin();
                    map_iterator.pc_offset = part_cell.pc_offset;
                    map_iterator.level = part_cell.level;

                    if (part_cell.pc_offset == 0) {
                        if (part_cell.level == level_min()) {
                            map_iterator.global_offset = 0;
                        } else {
                            map_iterator.global_offset = global_index_by_level_and_zx_end[part_cell.level - 1].back();
                        }
                    } else {
                        map_iterator.global_offset = global_index_by_level_and_zx_end[part_cell.level][
                                part_cell.pc_offset - 1];
                    }

                    map_iterator.max_offset = 0;
                }

                uint64_t offset = 0;
                //deals with the different xz in the same access tree at highest resolution
                offset = map_iterator.global_offset;


                if (map_iterator.iterator == current_pc_map.map.end()) {
                    //check if pointing to a valid key
                    map_iterator.iterator = current_pc_map.map.begin();
                }

                if ((part_cell.y >= map_iterator.iterator->first) &&
                    (part_cell.y <= map_iterator.iterator->second.y_end)) {
                    // already pointing to the correct place
                    part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset +
                                             (part_cell.y - map_iterator.iterator->first) + offset;

                    return true;
                } else {
                    map_iterator.iterator++;
                    //check if there
                    if (map_iterator.iterator != current_pc_map.map.end()) {
                        if ((part_cell.y >= map_iterator.iterator->first) &
                            (part_cell.y <= map_iterator.iterator->second.y_end)) {
                            // already pointing to the correct place
                            part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset +
                                                     (part_cell.y - map_iterator.iterator->first) + offset;

                            return true;
                        }
                    }

                    //otherwise search for it (points to first key that is greater than the y value)
                    map_iterator.iterator = current_pc_map.map.upper_bound(part_cell.y);

                    if ((map_iterator.iterator == current_pc_map.map.begin()) ||
                        (map_iterator.iterator == current_pc_map.map.end())) {
                        //less then the first value
                        return false;
                    } else {
                        map_iterator.iterator--;
                    }

                    if ((part_cell.y >= map_iterator.iterator->first) &
                        (part_cell.y <= map_iterator.iterator->second.y_end)) {
                        // already pointing to the correct place
                        part_cell.global_index = map_iterator.iterator->second.global_index_begin_offset +
                                                 (part_cell.y - map_iterator.iterator->first) + offset;

                        return true;
                    }
                }
            }
        }

        return false;
    }

    void initialize_structure_from_particle_cell_tree(APRParameters& apr_parameters,std::vector<PixelData<uint8_t>>& layers){
       x_num.resize(l_max+1);
       y_num.resize(l_max+1);
       z_num.resize(l_max+1);

        for(size_t i = l_min;i < l_max; ++i) {
            x_num[i] = layers[i].x_num;
            y_num[i] = layers[i].y_num;
            z_num[i] = layers[i].z_num;
        }
        y_num[l_max] = org_dims[0];
        x_num[l_max] = org_dims[1];
        z_num[l_max] = org_dims[2];

        //transfer over data-structure to make the same (re-use of function for read-write)
        std::vector<ArrayWrapper<uint8_t>> p_map(l_max);
        for (size_t k = 0; k < l_max; ++k) {
            p_map[k].swap(layers[k].mesh);
        }

        initialize_structure_from_particle_cell_tree( apr_parameters,p_map);
    }

    void initialize_structure_from_particle_cell_tree(APRParameters& apr_parameters,std::vector<ArrayWrapper<uint8_t>> &p_map);

    void allocate_map_insert( ExtraPartCellData<std::pair<uint16_t,YGap_map>>& y_begin) {
        // TODO: This whole method copies just y_begin converting std::pair into std::map entries, why we do not do that
        //       from the beginning (using map instead of pair, then we can skip this whole part of code).
        //

        APRTimer apr_timer;
        apr_timer.start_timer("initialize map");

        gap_map.depth_max = level_max();
        gap_map.depth_min = level_min();
        gap_map.z_num.resize(y_begin.depth_max+1);
        gap_map.x_num.resize(y_begin.depth_max+1);
        gap_map.data.resize(y_begin.depth_max+1);

        // Initialize sizes for each level with max level of size of level(max-1)
        for (uint64_t i = y_begin.depth_min; i < y_begin.depth_max; ++i) {
            gap_map.x_num[i] = x_num[i];
            gap_map.z_num[i] = z_num[i];
            gap_map.data[i].resize(z_num[i]*x_num[i]);
        }
        gap_map.x_num[y_begin.depth_max] = x_num[y_begin.depth_max-1];
        gap_map.z_num[y_begin.depth_max] = z_num[y_begin.depth_max-1];
        gap_map.data[y_begin.depth_max].resize(z_num[y_begin.depth_max-1]*x_num[y_begin.depth_max-1]);

        uint64_t counter_rows = 0;

        for (uint64_t level = (level_min()); level <= level_max(); ++level) {
            const unsigned int xLen = y_begin.x_num[level];
            const unsigned int zLen = y_begin.z_num[level];

            #ifdef HAVE_OPENMP
            #pragma omp parallel for default(shared) schedule(dynamic) reduction(+:counter_rows) if(zLen*xLen > 100)
            #endif
            for (uint64_t z = 0; z < zLen; ++z) {
                for (uint64_t x = 0; x < xLen; ++x) {
                    const size_t offset_pc_data = xLen * z + x;
                    if (y_begin.data[level][offset_pc_data].size() > 0) {
                        gap_map.data[level][offset_pc_data].resize(1); // create one empty std::map
                        gap_map.data[level][offset_pc_data][0].map.insert(y_begin.data[level][offset_pc_data].begin(),
                                                                          y_begin.data[level][offset_pc_data].end());
                        counter_rows++;
                    }
                }
            }
        }
        total_number_non_empty_rows = counter_rows;

        apr_timer.stop_timer();
    }


    void allocate_map(MapStorageData& map_data,std::vector<uint64_t>& cumsum,bool tree = false){

        //first add the layers
        gap_map.depth_max = level_max();
        gap_map.depth_min = level_min();

        gap_map.z_num.resize(gap_map.depth_max+1);
        gap_map.x_num.resize(gap_map.depth_max+1);
        gap_map.data.resize(gap_map.depth_max+1);

        global_index_by_level_and_zx_end.resize(gap_map.depth_max+1);


        if(tree) {
            for (uint64_t i = gap_map.depth_min; i < gap_map.depth_max; i++) {
                gap_map.z_num[i] = z_num[i];
                gap_map.x_num[i] = x_num[i];
                gap_map.data[i].resize(z_num[i] * x_num[i]);
                global_index_by_level_and_zx_end[i].resize(z_num[i] * x_num[i], 0);
            }

        } else {
            for (uint64_t i = gap_map.depth_min; i < gap_map.depth_max; i++) {
                gap_map.z_num[i] = z_num[i];
                gap_map.x_num[i] = x_num[i];
                gap_map.data[i].resize(z_num[i] * x_num[i]);
                global_index_by_level_and_zx_end[i].resize(z_num[i] * x_num[i], 0);
            }

            gap_map.z_num[level_max()] = z_num[level_max() - 1];
            gap_map.x_num[level_max()] = x_num[level_max() - 1];
            gap_map.data[level_max()].resize(z_num[level_max() - 1] * x_num[level_max() - 1]);
            global_index_by_level_and_zx_end[level_max()].resize(z_num[level_max() - 1] * x_num[level_max() - 1], 0);
        }
        uint64_t j;
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) schedule(dynamic) private(j)
#endif
        for (j = 0; j < total_number_non_empty_rows; ++j) {

            const uint64_t level = map_data.level[j];
            const uint64_t offset_pc_data =  gap_map.x_num[level]* map_data.z[j] + map_data.x[j];
            const uint64_t global_begin = cumsum[j];
            const uint64_t number_gaps = map_data.number_gaps[j];

            YGap_map gap;

            gap_map.data[level][offset_pc_data].resize(1);

            uint64_t global_index = 0;
            global_index_by_level_and_zx_end[level][offset_pc_data] = map_data.global_index[j];

            for (uint64_t i = global_begin; i < (global_begin + number_gaps) ; ++i) {
                gap.y_end = map_data.y_end[i];
                gap.global_index_begin_offset = global_index;

                auto hint = gap_map.data[level][offset_pc_data][0].map.end();
                gap_map.data[level][offset_pc_data][0].map.insert(hint,{map_data.y_begin[i],gap});
                global_index += map_data.y_end[i] - map_data.y_begin[i] + 1;
            }
        }
    }

    void rebuild_map_tree(MapStorageData& map_data,APRAccess& ARPOwn_access){

        uint64_t z_;
        uint64_t x_;
        APRTimer apr_timer;
        apr_timer.verbose_flag = false;
        apr_timer.start_timer("rebuild map");

        std::vector<uint64_t> cumsum;
        cumsum.reserve(total_number_non_empty_rows);
        uint64_t counter=0;

        uint64_t j;

        for (j = 0; j < total_number_non_empty_rows; ++j) {
            cumsum.push_back(counter);
            counter+=(map_data.number_gaps[j]);
        }

        allocate_map(map_data,cumsum,true);

        apr_timer.stop_timer();

        apr_timer.start_timer("forth loop");
        //////////////////
        ///
        /// Recalculate the iteration helpers
        ///
        //////////////////////

        //iteration helpers for by level
        global_index_by_level_begin.resize(level_max()+1,1);
        global_index_by_level_end.resize(level_max()+1,0);

        unsigned int curr_level = map_data.level[0];

        for (j = 0; j < total_number_non_empty_rows; ++j) {
            if(curr_level!=map_data.level[j]) {
                global_index_by_level_end[curr_level] = map_data.global_index[j]-1;
            }
            curr_level = map_data.level[j];
        }

        global_index_by_level_end[level_max() - 1] = map_data.global_index.back() - 1;
        global_index_by_level_end[level_max()] = total_number_particles-1;
        for (int i = 0; i <= level_max(); ++i) {
            if(global_index_by_level_end[i]>0) {
                global_index_by_level_begin[i] = global_index_by_level_end[i-1]+1;
            }
        }

        global_index_by_level_begin[map_data.level[0]] = map_data.global_index[0];

        uint64_t prev = 0;

        //now xz iteration helpers (need to fill in the gaps)
        for (int i = 0; i <= level_max(); ++i) {
            for (int k = 0; k < global_index_by_level_and_zx_end[i].size(); ++k) {
                if(global_index_by_level_and_zx_end[i][k]==0){
                    global_index_by_level_and_zx_end[i][k] =  prev;
                }
                prev = global_index_by_level_and_zx_end[i][k];
            }
        }


        //this is required for reading across the information from the APR
        const unsigned int x_num_ = x_num[level_max()];
        const unsigned int z_num_ = z_num[level_max()];

        uint64_t cumsum_l = map_data.global_index.back();

        global_index_by_level_and_zx_end[level_max()].resize(z_num_ * x_num_, 0);

        for (z_ = 0; z_ < z_num_; z_++) {

            for (x_ = 0; x_ < x_num_; x_++) {
                const size_t offset_pc_data = x_num_ * z_ + x_;

                if (ARPOwn_access.gap_map.data[level_max()+1][offset_pc_data].size() > 0){
                    auto it = (ARPOwn_access.gap_map.data[level_max()+1][offset_pc_data][0].map.rbegin());
                    cumsum_l += it->second.global_index_begin_offset/2 + (it->second.y_end - it->first+1)/2 + (it->second.y_end+1)%2;
                    //need to deal with odd domains where the last particle cell is required, hence the modulo
                }

                global_index_by_level_and_zx_end[level_max()][offset_pc_data] = cumsum_l;
            }
        }

        apr_timer.stop_timer();
    }

    void flatten_structure(MapStorageData &map_data)  {
        //
        //  Flatten the map access structure for writing the output
        //

        map_data.y_begin.reserve(total_number_gaps);
        map_data.y_end.reserve(total_number_gaps);
        map_data.global_index.reserve(total_number_non_empty_rows);

        //total_number_non_empty_rows
        map_data.x.reserve(total_number_non_empty_rows);
        map_data.z.reserve(total_number_non_empty_rows);
        map_data.level.reserve(total_number_non_empty_rows);
        map_data.number_gaps.reserve(total_number_non_empty_rows);

        uint64_t z_;
        uint64_t x_;

        for(uint64_t i = (level_min());i <= level_max();i++) {

            unsigned int x_num_ = (unsigned int )x_num[i];
            unsigned int z_num_ = (unsigned int )z_num[i];

            if(gap_map.data[i].size()>0) {
                //account for tree where the last level doesn't exist
                if (i == level_max()) {
                    //account for the APR, where the maximum level is down-sampled one
                    x_num_ = (unsigned int) x_num[i - 1];
                    z_num_ = (unsigned int) z_num[i - 1];
                }

                for (z_ = 0; z_ < z_num_; z_++) {
                    for (x_ = 0; x_ < x_num_; x_++) {
                        const uint64_t offset_pc_data = x_num_ * z_ + x_;
                        if (gap_map.data[i][offset_pc_data].size() > 0) {
                            map_data.x.push_back(x_);
                            map_data.z.push_back(z_);
                            map_data.level.push_back(i);
                            map_data.number_gaps.push_back(gap_map.data[i][offset_pc_data][0].map.size());

                            map_data.global_index.push_back(global_index_by_level_and_zx_end[i][offset_pc_data]);

                            for (auto const &element : gap_map.data[i][offset_pc_data][0].map) {
                                map_data.y_begin.push_back(element.first);
                                map_data.y_end.push_back(element.second.y_end);
                            }
                        }

                    }
                }
            }
        }
    }

    void rebuild_map(MapStorageData& map_data){

        APRTimer apr_timer;
        apr_timer.verbose_flag = true;
        apr_timer.start_timer("rebuild map");

        std::vector<uint64_t> cumsum;
        cumsum.reserve(total_number_non_empty_rows);
        uint64_t counter=0;

        uint64_t j;

        for (j = 0; j < total_number_non_empty_rows; ++j) {
            cumsum.push_back(counter);
            counter+=(map_data.number_gaps[j]);
        }

        allocate_map(map_data,cumsum);

        apr_timer.stop_timer();

        apr_timer.start_timer("forth loop");
        //////////////////
        ///
        /// Recalculate the iteration helpers
        ///
        //////////////////////

        //iteration helpers for by level
        global_index_by_level_begin.resize(level_max()+1,1);
        global_index_by_level_end.resize(level_max()+1,0);

        unsigned int curr_level = map_data.level[0];

        for (j = 0; j < total_number_non_empty_rows; ++j) {
            if(curr_level!=map_data.level[j]) {
                global_index_by_level_end[curr_level] = map_data.global_index[j]-1;
            }
            curr_level = map_data.level[j];
        }

        global_index_by_level_end[level_max()] = total_number_particles-1;

        for (int i = 0; i <= level_max(); ++i) {
            if(global_index_by_level_end[i]>0) {
                global_index_by_level_begin[i] = global_index_by_level_end[i-1]+1;
            }
        }

        global_index_by_level_begin[map_data.level[0]] = map_data.global_index[0];

        uint64_t prev = 0;

        //now xz iteration helpers (need to fill in the gaps)
        for (int i = 0; i <= level_max(); ++i) {
            for (int k = 0; k < global_index_by_level_and_zx_end[i].size(); ++k) {
                if(global_index_by_level_and_zx_end[i][k]==0){
                    global_index_by_level_and_zx_end[i][k] =  prev;
                }
                prev = global_index_by_level_and_zx_end[i][k];
            }
        }

        apr_timer.stop_timer();
    }

    void initialize_tree_access(APRAccess& APROwn_access, std::vector<PixelData<uint8_t>> &p_map);


};

inline void APRAccess::initialize_structure_from_particle_cell_tree(APRParameters& apr_parameters,std::vector<ArrayWrapper<uint8_t>> &p_map) {
    APRTimer apr_timer(false);

    uint8_t min_type = apr_parameters.neighborhood_optimization ? 1 : 2;

    // ========================================================================
    apr_timer.start_timer("first_step");
    const uint8_t UPSAMPLING_SEED_TYPE = 4;
    const uint8_t seed_us = UPSAMPLING_SEED_TYPE; //deal with the equivalence optimization
    for (size_t i = level_min()+1; i < level_max(); ++i) {
        const size_t xLen = x_num[i];
        const size_t zLen = z_num[i];
        const size_t yLen = y_num[i];
        const size_t xLenUpsampled = x_num[i - 1];
        const size_t yLenUpsampled = y_num[i - 1];

        #ifdef HAVE_OPENMP
        #pragma omp parallel for default(shared) if (zLen*xLen > 100)
        #endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_part_map_ds = (x / 2) * yLenUpsampled + (z / 2) * yLenUpsampled * xLenUpsampled;
                const size_t offset_part_map = x * yLen + z * yLen * xLen;

                for (size_t y = 0; y < yLenUpsampled; ++y) {
                    uint8_t status = p_map[i - 1][offset_part_map_ds + y];

                    if (status > 0 && status <= min_type) {
                        size_t y2p = std::min(2*y+1,yLen-1);
                        p_map[i][offset_part_map + 2 * y] = seed_us;
                        p_map[i][offset_part_map + y2p] = seed_us;
                    }
                }
            }
        }
    }
    apr_timer.stop_timer();

    // ========================================================================
    apr_timer.start_timer("second_step");

    ExtraPartCellData<std::pair<uint16_t, YGap_map>> y_begin;

    y_begin.depth_max = level_max();
    y_begin.depth_min = level_min();
    y_begin.z_num.resize(y_begin.depth_max+1);
    y_begin.x_num.resize(y_begin.depth_max+1);
    y_begin.data.resize(y_begin.depth_max+1); // [level][x_num(level) * z + x][y]

    // Initialize sizes for each level with max level of size of level(max-1)
    for (uint64_t i = y_begin.depth_min; i < y_begin.depth_max; ++i) {
        y_begin.x_num[i] = x_num[i];
        y_begin.z_num[i] = z_num[i];
        y_begin.data[i].resize(z_num[i]*x_num[i]);
    }
    y_begin.x_num[y_begin.depth_max] = x_num[y_begin.depth_max-1];
    y_begin.z_num[y_begin.depth_max] = z_num[y_begin.depth_max-1];
    y_begin.data[y_begin.depth_max].resize(z_num[y_begin.depth_max-1]*x_num[y_begin.depth_max-1]);

    for (size_t i = (level_min());i < level_max(); ++i) {
        const size_t xLen = x_num[i];
        const size_t zLen = z_num[i];
        const size_t yLen = y_num[i];

        #ifdef HAVE_OPENMP
        #pragma omp parallel for default(shared) if(zLen*xLen > 100)
        #endif
        for (size_t z = 0; z < zLen; ++z) {
            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_pc_data = z * xLen + x;
                const size_t offset_part_map = yLen * offset_pc_data;
                uint16_t current = 0;

                YGap_map gap;
                gap.global_index_begin_offset = 0;
                uint64_t counter = 0;

                for (size_t y = 0; y < yLen; ++y) {
                    uint8_t status = p_map[i][offset_part_map + y];
                    if ((status > min_type) && (status <= UPSAMPLING_SEED_TYPE)) {
                        if (current == 0) {
                            y_begin.data[i][offset_pc_data].push_back({y, gap});
                        }
                        current = 1;
                    }
                    else {
                        if (current == 1) {
                            (y_begin.data[i][offset_pc_data][counter]).second.y_end = (y-1);
                            counter++;
                        }
                        current = 0;
                    }
                }
                //end node
                if (current == 1) {
                    (y_begin.data[i][offset_pc_data][counter]).second.y_end = (yLen-1);
                }
            }
        }
    }
    apr_timer.stop_timer();

    // ========================================================================
    apr_timer.start_timer("third loop");

    size_t i = level_max()-1;

    const size_t xLen = x_num[i];
    const size_t zLen = z_num[i];
    const size_t yLen = y_num[i];
    const size_t yLenUpsampled = y_num[i + 1];

    #ifdef HAVE_OPENMP
    #pragma omp parallel for default(shared) if(zLen*xLen > 100)
    #endif
    for (size_t z = 0; z < zLen; ++z) {
        for (size_t x = 0; x < xLen; ++x) {
            const size_t offset_part_map = x * yLen + z * yLen * xLen;
            const size_t offset_pc_data1 = std::min(xLen*(z) + (x), xLen*zLen - 1);
            uint16_t current = 0;

            YGap_map gap;
            gap.global_index_begin_offset = 0;
            uint64_t counter = 0;

            for (size_t y = 0; y < yLen; ++y) {
                uint8_t status = p_map[i][offset_part_map + y];
                if (status > 0 && status <= min_type) {
                    if (current == 0) {
                        y_begin.data[i+1][offset_pc_data1].push_back({2*y,gap});
                    }
                    current = 1;
                }
                else {
                    if (current == 1) {
                        y_begin.data[i+1][offset_pc_data1][counter].second.y_end = std::min((uint16_t)(2*(y-1)+1),(uint16_t)(yLenUpsampled-1));
                        counter++;
                    }
                    current = 0;
                }
            }

            //last gap
            if (current == 1) {
                y_begin.data[i+1][offset_pc_data1][counter].second.y_end = (yLenUpsampled-1);
            }
        }
    }

    apr_timer.stop_timer();

    // ========================================================================
    apr_timer.start_timer("forth loop");

    // Initialize global structures
    total_number_gaps=0;

    global_index_by_level_begin.resize(level_max()+1, 1); // TODO: This is dirty hack since later when compute number of particles
    global_index_by_level_end.resize(level_max()+1, 0);   // TODO: we get 0-1 which will give max num for size_t, change it!!!!

    global_index_by_level_and_zx_end.resize(level_max()+1);

    size_t cumsum= 0;
    size_t min_level_find = level_max();

    for (size_t level = level_min(); level <= level_max(); ++level) {
        const size_t xLen = y_begin.x_num[level];
        const size_t zLen = y_begin.z_num[level];

        uint64_t cumsum_begin = cumsum;

        global_index_by_level_and_zx_end[level].resize(zLen * xLen, 0);

        for (size_t z = 0; z < zLen; ++z) {

            for (size_t x = 0; x < xLen; ++x) {
                const size_t offset_xz_data = xLen * z + x;

                uint16_t localSizeOfGaps = 0;

                // process all gaps for given xz coordinates
                for (size_t y = 0; y < y_begin.data[level][offset_xz_data].size(); ++y) {
                    min_level_find = std::min(level, min_level_find);

                    y_begin.data[level][offset_xz_data][y].second.global_index_begin_offset = localSizeOfGaps;
                    const auto idxOfLastElementInGap = y_begin.data[level][offset_xz_data][y].second.y_end;
                    const auto idxOfFirstElementInGap = y_begin.data[level][offset_xz_data][y].first;
                    const auto sizeOfGap = idxOfLastElementInGap - idxOfFirstElementInGap + 1;
                    localSizeOfGaps += sizeOfGap;
                    total_number_gaps++;
                }

                cumsum += localSizeOfGaps;

                if (level == level_max()) {
                    // need to deal with the boundary conditions: on upsampled level we have rows X Y
                    //                                                                            Y Y
                    // where X gaps are calculated above and all Y gaps (if included in domain) are handled below
                    int number_rows = 1 * ((2*x+1) < x_num[level]) +
                                      1 * ((2*z+1) < z_num[level]) +
                                      1 * ((2*z+1) < z_num[level]) * ((2*x+1) < x_num[level]);
                    cumsum += localSizeOfGaps * number_rows;
                }

                global_index_by_level_and_zx_end[level][offset_xz_data] = cumsum;
            }
        }

        if (cumsum != cumsum_begin) {
            global_index_by_level_begin[level] = cumsum_begin;
            global_index_by_level_end[level] = cumsum-1;
        }
    }
    total_number_particles = cumsum;
    l_min = min_level_find; //set minimum level to the first non-empty level.
    apr_timer.stop_timer();

    // ========================================================================
    apr_timer.start_timer("insert");
    allocate_map_insert(y_begin);
    apr_timer.stop_timer();
}

inline void APRAccess::initialize_tree_access(APRAccess& APROwn_access, std::vector<PixelData<uint8_t>> &p_map) {
    APRTimer apr_timer(false);


    x_num.resize(level_max()+1);
    y_num.resize(level_max()+1);
    z_num.resize(level_max()+1);

    for(int i = level_min();i < level_max();i++){
        x_num[i] = p_map[i].x_num;
        y_num[i] = p_map[i].y_num;
        z_num[i] = p_map[i].z_num;
    }

    x_num[level_max()] = APROwn_access.x_num[level_max()];
    y_num[level_max()] = APROwn_access.y_num[level_max()];
    z_num[level_max()] = APROwn_access.z_num[level_max()];

    //initialize loop variables
    uint64_t x_;
    uint64_t z_;
    uint64_t y_,status;

    apr_timer.start_timer("init structure");

    ExtraPartCellData<std::pair<uint16_t,YGap_map>> y_begin;

    y_begin.depth_min = level_min();
    y_begin.depth_max = level_max();

    y_begin.z_num.resize(y_begin.depth_max+1);
    y_begin.x_num.resize(y_begin.depth_max+1);
    y_begin.data.resize(y_begin.depth_max+1);

    for (uint64_t i = y_begin.depth_min; i < y_begin.depth_max; ++i) {
        y_begin.z_num[i] = z_num[i];
        y_begin.x_num[i] = x_num[i];
        y_begin.data[i].resize(z_num[i]*x_num[i]);
    }

    apr_timer.stop_timer();

    apr_timer.start_timer("create gaps");

    for(uint64_t i = (level_min());i < level_max();i++) {

        const uint64_t x_num_ = x_num[i];
        const uint64_t z_num_ = z_num[i];
        const uint64_t y_num_ = y_num[i];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) default(shared) private(z_, x_, y_, status) if(z_num_*x_num_ > 100)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {

            for (x_ = 0; x_ < x_num_; x_++) {
                const size_t offset_part_map = x_ * y_num_ + z_ * y_num_ * x_num_;
                const size_t offset_pc_data = x_num_*z_ + x_;

                uint16_t current = 0;
                uint16_t previous = 0;

                YGap_map gap;
                gap.global_index_begin_offset = 0;
                uint64_t counter = 0;

                for (y_ = 0; y_ < y_num_; y_++) {

                    status = p_map[i].mesh[offset_part_map + y_];
                    if(status > 0) {
                        current = 1;

                        if(previous == 0){
                            y_begin.data[i][offset_pc_data].push_back({y_,gap});

                        }
                    } else {
                        current = 0;

                        if(previous == 1){

                            (y_begin.data[i][offset_pc_data][counter]).second.y_end = (y_-1);
                            counter++;
                        }
                    }

                    previous = current;

                }
                //end node
                if(previous==1) {

                    (y_begin.data[i][offset_pc_data][counter]).second.y_end = (y_num_-1);
                }
            }
        }
    }

    apr_timer.stop_timer();

    uint64_t cumsum = 0;

    apr_timer.start_timer("forth loop");

    //iteration helpers for by level
    global_index_by_level_begin.resize(level_max()+1,1);
    global_index_by_level_end.resize(level_max()+1,0);

    cumsum= 0;

    total_number_gaps=0;

    uint64_t min_level_find = level_max();
    uint64_t max_level_find = level_min();

    global_index_by_level_and_zx_end.resize(level_max()+1);

    for(uint64_t i = (level_min());i < level_max();i++) {

        const unsigned int x_num_ = x_num[i];
        const unsigned int z_num_ = z_num[i];

        //set up the levels here.
        uint64_t cumsum_begin = cumsum;

        global_index_by_level_and_zx_end[i].resize(z_num_ * x_num_, 0);

        for (z_ = 0; z_ < z_num_; z_++) {
            for (x_ = 0; x_ < x_num_; x_++) {
                const size_t offset_pc_data = x_num_ * z_ + x_;

                uint16_t local_sum = 0;

                for (size_t j = 0; j < y_begin.data[i][offset_pc_data].size(); ++j) {

                    min_level_find = std::min(i,min_level_find);
                    max_level_find = std::max(i,max_level_find);

                    y_begin.data[i][offset_pc_data][j].second.global_index_begin_offset = local_sum;
                    local_sum+=(y_begin.data[i][offset_pc_data][j].second.y_end-y_begin.data[i][offset_pc_data][j].first)+1;
                    total_number_gaps++;
                }

                cumsum += local_sum;

                global_index_by_level_and_zx_end[i][offset_pc_data] = cumsum;
            }
        }

        if(cumsum!=cumsum_begin){
            //cumsum_begin++;
            global_index_by_level_begin[i] = cumsum_begin;
        }

        if(cumsum!=cumsum_begin){
            global_index_by_level_end[i] = cumsum-1;
        }
    }

    total_number_particles = cumsum;

    const unsigned int x_num_ = x_num[level_max()];
    const unsigned int z_num_ = z_num[level_max()];

    //set up the levels here

    global_index_by_level_and_zx_end[level_max()].resize(z_num_ * x_num_, 0);

    for (z_ = 0; z_ < z_num_; z_++) {
        for (x_ = 0; x_ < x_num_; x_++) {
            const size_t offset_pc_data = x_num_ * z_ + x_;

            if (APROwn_access.gap_map.data[level_max()+1][offset_pc_data].size() > 0){
                auto it = (APROwn_access.gap_map.data[level_max()+1][offset_pc_data][0].map.rbegin());
                cumsum += it->second.global_index_begin_offset/2 + (it->second.y_end - it->first+1)/2 + (it->second.y_end+1)%2;
                //need to deal with odd domains where the last particle cell is required, hence the modulo
            }

            global_index_by_level_and_zx_end[level_max()][offset_pc_data] = cumsum;
        }
    }

    apr_timer.stop_timer();

    total_number_particles = cumsum;

    //std::cout << "Lower level, interior tree PC: " << total_number_particles << std::endl;

    total_number_non_empty_rows=0;

    apr_timer.start_timer("set up gapmap");

    gap_map.depth_min = level_min();
    gap_map.depth_max = level_max();

    gap_map.z_num.resize(y_begin.depth_max+1);
    gap_map.x_num.resize(y_begin.depth_max+1);
    gap_map.data.resize(y_begin.depth_max+1);

    for (uint64_t i = gap_map.depth_min; i < gap_map.depth_max; ++i) {
        gap_map.z_num[i] = z_num[i];
        gap_map.x_num[i] = x_num[i];
        gap_map.data[i].resize(z_num[i]*x_num[i]);
    }

    uint64_t counter_rows = 0;

    for (uint64_t i = (level_min()); i < level_max(); i++) {
        const unsigned int x_num_ = x_num[i];
        const unsigned int z_num_ = z_num[i];
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) schedule(dynamic) private(z_, x_) reduction(+:counter_rows)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {
            for (x_ = 0; x_ < x_num_; x_++) {
                const size_t offset_pc_data = x_num_ * z_ + x_;
                if (y_begin.data[i][offset_pc_data].size() > 0) {
                    gap_map.data[i][offset_pc_data].resize(1);


                    gap_map.data[i][offset_pc_data][0].map.insert(y_begin.data[i][offset_pc_data].begin(),
                                                                  y_begin.data[i][offset_pc_data].end());

                    counter_rows++;
                }
            }
        }
    }

    apr_timer.stop_timer();

    total_number_non_empty_rows = counter_rows;
}

#endif //PARTPLAY_APRACCESS_HPP
