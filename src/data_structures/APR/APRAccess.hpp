//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APRACCESS_HPP
#define PARTPLAY_APRACCESS_HPP



#include <map>
#include <utility>

#define _NO_NEIGHBOUR ((uint16_t)3)
#define _LEVEL_SAME ((uint16_t)1)
#define _LEVEL_DECREASE ((uint16_t)0)
#define _LEVEL_INCREASE ((uint16_t)2)

#define _EMPTY ((uint16_t)0)
#define _SEED ((uint16_t)1)
#define _BOUNDARY ((uint16_t)2)
#define _FILLER ((uint16_t)3)

#define YP_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 1
#define YP_LEVEL_SHIFT (uint16_t)  1

#define YM_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 3
#define YM_LEVEL_SHIFT (uint16_t) 3

#define XP_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 5
#define XP_LEVEL_SHIFT 5

#define XM_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 7
#define XM_LEVEL_SHIFT 7

#define ZP_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 9
#define ZP_LEVEL_SHIFT 9

#define ZM_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 11
#define ZM_LEVEL_SHIFT 11

#define PC_TYPE_MASK ((((uint16_t)1) << 2) - 1) << 13
#define PC_TYPE_SHIFT 13

template<typename T>
class ExtraParticleData;

template<typename ImageType>
class APRIterator;

#include "src/data_structures/APR/APR.hpp"

#include "src/data_structures/APR/ExtraParticleData.hpp"
#include "src/data_structures/APR/ExtraPartCellData.hpp"


struct ParticleCell {
    uint16_t x,y,z,level,type;
    uint64_t pc_offset,global_index;
};

struct YGap {
    uint64_t y_begin;
    uint64_t y_end;
    uint64_t global_index_begin;

//    YGap():y_begin(0),y_end(0),global_index_begin(0){
//
//    }
};

struct YGap_map {
    uint16_t y_end;
    uint64_t global_index_begin;
};

struct ParticleCellGapMap{
    std::map<uint16_t,YGap_map> map;
};

struct YIterators{
    ExtraPartCellData<std::map<uint16_t,YGap_map>::iterator> gap_map_it;
};


struct MapIterator{
    std::map<uint16_t,YGap_map>::iterator iterator;
    uint64_t pc_offset;
    uint16_t level;
};

struct LocalMapIterators{
    std::vector<MapIterator>  same_level;
    std::vector<std::vector<MapIterator>>  child_level;
    std::vector<MapIterator>  parent_level;

    LocalMapIterators(){
        //initialize them to be set to pointing to no-where
        MapIterator init;
        init.pc_offset = -1;
        init.level = -1;

        same_level.resize(6,init);
        parent_level.resize(6,init);
        child_level.resize(6);
        for (int i = 0; i < 6; ++i) {
            child_level[i].resize(4,init);
        }
    }
};


struct MapStorageData{

    std::vector<uint16_t> y_begin;
    std::vector<uint16_t> y_end;
    std::vector<uint64_t> global_index;
    std::vector<uint16_t> z;
    std::vector<uint16_t> x;
    std::vector<uint8_t> level;
    std::vector<uint16_t> number_gaps;

};


class APRAccess {

public:

    ExtraPartCellData<ParticleCellGapMap> gap_map;
    //ExtraPartCellData<std::map<uint16_t,YGap_map>::iterator> gap_map_it;

    ExtraParticleData<uint8_t> particle_cell_type;

    uint64_t level_max;
    uint64_t level_min;

    uint64_t org_dims[3]={0,0,0};

    std::vector<uint64_t> x_num;
    std::vector<uint64_t> y_num;
    std::vector<uint64_t> z_num;

    uint64_t total_number_particles;

    uint64_t total_number_gaps;

    uint64_t total_number_non_empty_rows;

    std::vector<uint64_t> global_index_by_level_begin;
    std::vector<uint64_t> global_index_by_level_end;

    std::vector<std::vector<uint64_t>> global_index_by_level_and_z_begin;
    std::vector<std::vector<uint64_t>> global_index_by_level_and_z_end;

    APRAccess(){

    };

    MapIterator& get_local_iterator(LocalMapIterators& local_iterators,const uint16_t& level_delta,const uint16_t& face,const uint16_t& index){
        //
        //  Chooses the local iterator required
        //

        switch (level_delta){
            case _LEVEL_SAME:

                return local_iterators.same_level[face];

            case _LEVEL_DECREASE:

                return local_iterators.parent_level[face];

            case _LEVEL_INCREASE:

                return local_iterators.child_level[face][index];
        }

        return local_iterators.same_level[0];
    }





    inline bool get_neighbour_coordinate(const ParticleCell& input,ParticleCell& neigh,const unsigned int& face,const uint16_t& level_delta,const uint16_t& index){
        //
        //

        static constexpr int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
        static constexpr int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
        static constexpr int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

        static constexpr uint8_t children_index_offsets[4][2] = {{0,0},{0,1},{1,0},{1,1}};

        unsigned int dir;

        switch (level_delta){
            case _LEVEL_SAME:
                //Same Level Particle Cell
                neigh.x = input.x + dir_x[face];
                neigh.y = input.y + dir_y[face];
                neigh.z = input.z + dir_z[face];
                neigh.level = input.level;

                neigh.pc_offset =  x_num[neigh.level] * neigh.z + neigh.x;

                return true;
            case _LEVEL_DECREASE:
                //Larger Particle Cell (Lower Level)
                neigh.level = input.level - 1;
                neigh.x = (input.x+ dir_x[face])/2;
                neigh.y = (input.y+ dir_y[face])/2;
                neigh.z = (input.z+ dir_z[face])/2;

                neigh.pc_offset =  x_num[neigh.level] * neigh.z + neigh.x;

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

                neigh.pc_offset =  x_num[neigh.level] * neigh.z + neigh.x;

                return true;
            case _NO_NEIGHBOUR:

                return false;
        }

        return false;

    }

    inline uint64_t get_parts_start(const uint16_t& x,const uint16_t& z,const uint16_t& level){

        const uint64_t offset = x_num[level] * z + x;
        if(gap_map.data[level][offset].size() > 0){
            auto it = (gap_map.data[level][offset][0].map.begin());
            return it->second.global_index_begin;
        } else {
            return (-1);
        }

    }

    inline uint64_t get_parts_end(const uint16_t& x,const uint16_t& z,const uint16_t& level){
        const uint64_t offset = x_num[level] * z + x;
        if(gap_map.data[level][offset].size() > 0){
            auto it = (gap_map.data[level][offset][0].map.rbegin());
            return (it->second.global_index_begin + (it->second.y_end-it->first));
        } else {
            return (0);
        }

    }

    inline uint64_t global_index_end(MapIterator& it){
        return (it.iterator->second.global_index_begin + (it.iterator->second.y_end-it.iterator->first));
    }

    inline bool check_neighbours_flag(const uint16_t& x,const uint16_t& z,const uint16_t& level){
        return ((uint16_t)(x-1)>(x_num[level]-3)) | ((uint16_t)(z-1)>(z_num[level]-3));
    }

    inline uint8_t number_neighbours_in_direction(const uint8_t& level_delta){
        //
        //  Gives the maximum number of neighbours in a direction given the level_delta.
        //

        switch (level_delta){
            case _LEVEL_INCREASE:
                return 4;
            case _NO_NEIGHBOUR:
                return 0;
        }
        return 1;
    }

    bool find_particle_cell(ParticleCell& part_cell,MapIterator& map_iterator){

        if(gap_map.data[part_cell.level][part_cell.pc_offset].size() > 0) {

            ParticleCellGapMap& current_pc_map = gap_map.data[part_cell.level][part_cell.pc_offset][0];

            if((map_iterator.pc_offset != part_cell.pc_offset) || (map_iterator.level != part_cell.level) ){
                map_iterator.iterator = gap_map.data[part_cell.level][part_cell.pc_offset][0].map.begin();
                map_iterator.pc_offset = part_cell.pc_offset;
                map_iterator.level = part_cell.level;
            }

            if(map_iterator.iterator == current_pc_map.map.end()){
                //check if pointing to a valid key
                map_iterator.iterator = current_pc_map.map.begin();
            }

            if ((part_cell.y >= map_iterator.iterator->first) && (part_cell.y <= map_iterator.iterator->second.y_end)) {
                // already pointing to the correct place
                part_cell.global_index = map_iterator.iterator->second.global_index_begin +
                                         (part_cell.y - map_iterator.iterator->first);

                return true;
            } else {
                //first try next element
                if(map_iterator.iterator != current_pc_map.map.end()){
                    map_iterator.iterator++;
                    //check if there
                    if(map_iterator.iterator != current_pc_map.map.end()) {
                        if ((part_cell.y >= map_iterator.iterator->first) &
                            (part_cell.y <= map_iterator.iterator->second.y_end)) {
                            // already pointing to the correct place
                            part_cell.global_index = map_iterator.iterator->second.global_index_begin +
                                                     (part_cell.y - map_iterator.iterator->first);

                            return true;
                        }
                    }

                }

                //otherwise search for it (points to first key that is greater than the y value)
                map_iterator.iterator = current_pc_map.map.upper_bound(part_cell.y);

                if(map_iterator.iterator == current_pc_map.map.begin()){
                    //less then the first value
                    return false;
                } else{
                    map_iterator.iterator--;
                }

                if ((part_cell.y >= map_iterator.iterator->first) & (part_cell.y <= map_iterator.iterator->second.y_end)) {
                    // already pointing to the correct place
                    part_cell.global_index = map_iterator.iterator->second.global_index_begin +
                                             (part_cell.y - map_iterator.iterator->first);

                    return true;
                }
            }
        }

        return false;

    }


    template<typename T>
    void generate_pmap(APR<T>& apr,std::vector<std::vector<uint8_t>>& p_map){

        uint64_t z_ = 0;
        uint64_t x_ = 0;
        uint64_t j_ = 0;

        uint64_t node_val;
        uint64_t status,y_coord;

        level_min = apr.level_min();
        level_max = apr.level_max();

        p_map.resize(apr.level_max()+1);

        x_num.resize(apr.level_max()+1);
        y_num.resize(apr.level_max()+1);
        z_num.resize(apr.level_max()+1);


        for (int level = apr.level_min(); level <= apr.level_max(); level++) {
            x_num[level] = apr.spatial_index_x_max(level);
            y_num[level] = apr.spatial_index_y_max(level);
            z_num[level] = apr.spatial_index_z_max(level);
        }
        org_dims[1] = x_num[level_max];
        org_dims[0] = y_num[level_max];
        org_dims[2] = z_num[level_max];


        for(uint64_t i = apr.level_min();i < apr.level_max();i++){


            unsigned int x_num_ = apr.pc_data.x_num[i];
            unsigned int z_num_ = apr.pc_data.z_num[i];
            unsigned int y_num_ = apr.pc_data.y_num[i];

            uint64_t node_val;

            p_map[i].resize(x_num_*z_num_*y_num_,0);

            std::fill(p_map[i].begin(), p_map[i].end(), 0);

            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;

                    const size_t j_num =apr.pc_data.data[i][offset_pc_data].size();

                    y_coord = 0;

                    for(j_ = 0;j_ < j_num;j_++){

                        node_val =apr.pc_data.data[i][offset_pc_data][j_];

                        if (!(node_val&1)){
                            //get the index gap node
                            y_coord++;

                            status =apr.pc_data.get_status(node_val);

                            if(status > 1) {

                                p_map[i][offset_p_map + y_coord] = status;

                            }

                        } else {

                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure

                        }

                    }

                }

            }


            x_num_ =apr.pc_data.x_num[i+1];
            z_num_ =apr.pc_data.z_num[i+1];
            y_num_ =apr.pc_data.y_num[i+1];

            int x_num_d =apr.pc_data.x_num[i];
            int z_num_d =apr.pc_data.z_num[i];
            int y_num_d =apr.pc_data.y_num[i];

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_p_map = y_num_d*x_num_d*(z_/2) + y_num_d*(x_/2);

                    const size_t j_num =apr.pc_data.data[i+1][offset_pc_data].size();

                    y_coord = 0;

                    for(j_ = 0;j_ < j_num;j_++){

                        node_val =apr.pc_data.data[i+1][offset_pc_data][j_];

                        if (!(node_val&1)){
                            //get the index gap node
                            y_coord++;

                            status =apr.pc_data.get_status(node_val);

                            if(status == 1) {

                                p_map[i][offset_p_map + y_coord/2] = status;

                            }

                        } else {

                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                        }

                    }

                }

            }



        }



    }

    template<typename T>
    void initialize_structure_from_particle_cell_tree(APR<T>& apr,std::vector<MeshData<uint8_t>>& layers){
       x_num.resize(level_max+1);
       y_num.resize(level_max+1);
       z_num.resize(level_max+1);

        for(int i = level_min;i < level_max;i++){
            x_num[i] = layers[i].x_num;
            y_num[i] = layers[i].y_num;
            z_num[i] = layers[i].z_num;
        }

        y_num[level_max] = org_dims[0];
        x_num[level_max] = org_dims[1];
        z_num[level_max] = org_dims[2];

        //transfer over data-structure to make the same (re-use of function for read-write)

        std::vector<std::vector<uint8_t>> p_map;
        p_map.resize(level_max);

        for (int k = 0; k < level_max; ++k) {
            std::swap(p_map[k],layers[k].mesh);
        }

        initialize_structure_from_particle_cell_tree(apr, p_map);


    }


    template<typename T>
    void initialize_structure_from_particle_cell_tree(APR<T>& apr,std::vector<std::vector<uint8_t>>& p_map) {
        //
        //  Initialize the new structure;
        //

        APRTimer apr_timer;
        apr_timer.verbose_flag = false;


        apr_timer.start_timer("first_step");

        //initialize loop variables
        uint64_t x_;
        uint64_t z_;
        uint64_t y_,status;

        const uint8_t seed_us = 4; //deal with the equivalence optimization

        for(uint64_t i = (apr.level_min()+1);i < apr.level_max();i++) {

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            const unsigned int x_num_ds = x_num[i - 1];
            const unsigned int z_num_ds = z_num[i - 1];
            const unsigned int y_num_ds = y_num[i - 1];

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_, x_, y_, status) if(z_num_*x_num_ > 100)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {
                    const size_t offset_part_map_ds = (x_ / 2) * y_num_ds + (z_ / 2) * y_num_ds * x_num_ds;
                    const size_t offset_part_map = x_ * y_num_ + z_ * y_num_ * x_num_;

                    for (y_ = 0; y_ < y_num_ds; y_++) {

                        status = p_map[i - 1][offset_part_map_ds + y_];

                        if (status == SEED) {
                            p_map[i][offset_part_map + 2 * y_] = seed_us;
                            p_map[i][offset_part_map + 2 * y_ + 1] = seed_us;
                        }
                    }
                }

            }
        }

        apr_timer.stop_timer();


        apr_timer.start_timer("second_step");



        ExtraPartCellData<std::pair<uint16_t,YGap_map>> y_begin;
        y_begin.initialize_structure_parts_empty(apr);

        for(uint64_t i = (apr.level_min());i < apr.level_max();i++) {

            const uint64_t x_num_ = x_num[i];
            const uint64_t z_num_ = z_num[i];
            const uint64_t y_num_ = y_num[i];

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_, x_, y_, status) if(z_num_*x_num_ > 100)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {
                    const size_t offset_part_map = x_ * y_num_ + z_ * y_num_ * x_num_;
                    const size_t offset_pc_data = x_num_*z_ + x_;

                    uint16_t current = 0;
                    uint16_t previous = 0;

                    YGap_map gap;
                    gap.global_index_begin = 0;
                    uint64_t counter = 0;

                    for (y_ = 0; y_ < y_num_; y_++) {

                        status = p_map[i][offset_part_map + y_];
                        if((status > 1) & (status < 5)) {
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

        apr_timer.start_timer("third loop");

        unsigned int i = apr.level_max()-1;

        const unsigned int x_num_ = x_num[i];
        const unsigned int z_num_ = z_num[i];
        const unsigned int y_num_ = y_num[i];

        const unsigned int x_num_us = x_num[i + 1];
        const unsigned int z_num_us = z_num[i + 1];
        const unsigned int y_num_us = y_num[i + 1];

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_, x_, y_, status) if(z_num_*x_num_ > 100)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {

            for (x_ = 0; x_ < x_num_; x_++) {
                const uint64_t offset_part_map = x_ * y_num_ + z_ * y_num_ * x_num_;
                const uint64_t offset_pc_data = x_num_*z_ + x_;

                const uint64_t offset_pc_data1 = std::min((uint64_t)x_num_us*(2*z_) + (2*x_),(uint64_t) x_num_us*z_num_us - 1);


                uint16_t current = 0;
                uint16_t previous = 0;

                YGap_map gap;
                gap.global_index_begin = 0;

                uint64_t counter = 0;

                for (y_ = 0; y_ < y_num_; y_++) {

                    status = p_map[i][offset_part_map + y_];
                    if(status ==SEED) {
                        current = 1;

                        if(previous == 0){
                            y_begin.data[i+1][offset_pc_data1].push_back({2*y_,gap});

                        }
                    } else {
                        current = 0;

                        if(previous == 1){

                            y_begin.data[i+1][offset_pc_data1][counter].second.y_end = std::min((uint16_t)(2*(y_-1)+1),(uint16_t)(y_num_us-1));

                            counter++;

                        }
                    }

                    previous = current;

                }
                //last gap
                if(previous == 1){

                    y_begin.data[i+1][offset_pc_data1][counter].second.y_end = (y_num_us-1);

                }

            }

        }

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_, x_, y_, status) if(z_num_*x_num_ > 100)
#endif
        for (z_ = 0; z_ < z_num_; z_++) {

            for (x_ = 0; x_ < x_num_; x_++) {
                const uint64_t offset_part_map = x_ * y_num_ + z_ * y_num_ * x_num_;
                const uint64_t offset_pc_data = x_num_*z_ + x_;

                const uint64_t offset_pc_data1 = std::min((uint64_t)x_num_us*(2*z_) + (2*x_),(uint64_t) x_num_us*z_num_us - 1);
                const uint64_t offset_pc_data2 = std::min((uint64_t)x_num_us*(2*z_) + (2*x_+1),(uint64_t) x_num_us*z_num_us - 1);
                const uint64_t offset_pc_data3 = std::min((uint64_t)x_num_us*(2*z_+1) + (2*x_),(uint64_t) x_num_us*z_num_us - 1);
                const uint64_t offset_pc_data4 = std::min((uint64_t)x_num_us*(2*z_+1) + (2*x_+1),(uint64_t) x_num_us*z_num_us - 1);

                uint16_t current = 0;
                uint16_t previous = 0;

                YGap_map gap;
                gap.global_index_begin = 0;

                size_t size_v = y_begin.data[i+1][offset_pc_data1].size();

                y_begin.data[i+1][offset_pc_data2].resize(size_v);
                std::copy(y_begin.data[i+1][offset_pc_data1].begin(),y_begin.data[i+1][offset_pc_data1].end(),y_begin.data[i+1][offset_pc_data2].begin());

                y_begin.data[i+1][offset_pc_data3].resize(size_v);
                std::copy(y_begin.data[i+1][offset_pc_data1].begin(),y_begin.data[i+1][offset_pc_data1].end(),y_begin.data[i+1][offset_pc_data3].begin());

                y_begin.data[i+1][offset_pc_data4].resize(size_v);
                std::copy(y_begin.data[i+1][offset_pc_data1].begin(),y_begin.data[i+1][offset_pc_data1].end(),y_begin.data[i+1][offset_pc_data4].begin());

                //end data copy


            }

        }

        //then need to loop over and then do a copy.
        apr_timer.stop_timer();

        uint64_t cumsum = 0;

        apr_timer.start_timer("forth loop");

        //iteration helpers for by level
        global_index_by_level_begin.resize(apr.level_max()+1,0);
        global_index_by_level_end.resize(apr.level_max()+1,0);

        cumsum= 0;

        total_number_gaps=0;

        uint64_t min_level_find = apr.level_max();

        //set up the iteration helpers for by zslice
        global_index_by_level_and_z_begin.resize(apr.level_max()+1);
        global_index_by_level_and_z_end.resize(apr.level_max()+1);

        for(uint64_t i = (apr.level_min());i <= apr.level_max();i++) {

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];
            //set up the levels here.

            uint64_t cumsum_begin = cumsum;

            global_index_by_level_and_z_begin[i].resize(z_num_,(-1));
            global_index_by_level_and_z_end[i].resize(z_num_,0);

            for (z_ = 0; z_ < z_num_; z_++) {
                uint64_t cumsum_begin_z = cumsum;


                for (x_ = 0; x_ < x_num_; x_++) {
                    const size_t offset_pc_data = x_num_ * z_ + x_;
                    for (int j = 0; j < y_begin.data[i][offset_pc_data].size(); ++j) {

                        min_level_find = std::min(i,min_level_find);

                        y_begin.data[i][offset_pc_data][j].second.global_index_begin = cumsum;

                        cumsum+=(y_begin.data[i][offset_pc_data][j].second.y_end-y_begin.data[i][offset_pc_data][j].first)+1;
                        total_number_gaps++;
                    }
                }
                if(cumsum!=cumsum_begin_z) {
                    global_index_by_level_and_z_end[i][z_] = cumsum - 1;
                    global_index_by_level_and_z_begin[i][z_] = cumsum_begin_z;
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

        apr_timer.stop_timer();


        //set minimum level now to the first non-empty level.
        level_min = min_level_find;

        total_number_non_empty_rows=0;

//        apr_timer.start_timer("initialize map");
//
//        gap_map.initialize_structure_parts_empty(apr);
//
//        uint64_t counter_rows=0;
//
//        for(uint64_t i = (apr.level_min());i <= apr.level_max();i++) {
//
//            const unsigned int x_num_ = x_num[i];
//            const unsigned int z_num_ = z_num[i];
//            const unsigned int y_num_ = y_num[i];
//#ifdef HAVE_OPENMP
//	#pragma omp parallel for default(shared) private(z_, x_) reduction(+:counter_rows)if(z_num_*x_num_ > 100)
//#endif
//            for (z_ = 0; z_ < z_num_; z_++) {
//                for (x_ = 0; x_ < x_num_; x_++) {
//                    const size_t offset_pc_data = x_num_ * z_ + x_;
//                    if(y_begin.data[i][offset_pc_data].size() > 0) {
//                        gap_map.data[i][offset_pc_data].resize(1);
//
//
//                        gap_map.data[i][offset_pc_data][0].map.insert(y_begin.data[i][offset_pc_data].begin(),y_begin.data[i][offset_pc_data].end());
//
//                        counter_rows++;
//                    }
//                }
//            }
//        }
//        total_number_non_empty_rows = counter_rows;
//        apr_timer.stop_timer();

        allocate_map_insert(apr,y_begin);


        APRIterator<T> apr_iterator(*this);

        particle_cell_type.data.resize(global_index_by_level_end[level_max-1]+1,0);

        uint64_t particle_number;

        for (uint64_t level = apr_iterator.level_min(); level < apr_iterator.level_max(); ++level) {

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = apr_iterator.particles_level_begin(level); particle_number <  apr_iterator.particles_level_end(level); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);
                const uint64_t offset_part_map = apr_iterator.x() * apr_iterator.spatial_index_y_max(apr_iterator.level()) + apr_iterator.z() * apr_iterator.spatial_index_y_max(apr_iterator.level()) * apr_iterator.spatial_index_x_max(apr_iterator.level());

                apr_iterator(particle_cell_type) = p_map[apr_iterator.level()][offset_part_map + apr_iterator.y()];

            }
        }


    }

    template<typename T>
    void allocate_map_insert(APR<T>& apr,ExtraPartCellData<std::pair<uint16_t,YGap_map>>& y_begin) {
        //
        //  Seperated for checking memory allocation
        //

        APRTimer apr_timer;

        apr_timer.start_timer("initialize map");

        gap_map.initialize_structure_parts_empty(apr);

        uint64_t counter_rows = 0;

        uint64_t z_,x_;

        for (uint64_t i = (apr.level_min()); i <= apr.level_max(); i++) {

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(z_, x_) reduction(+:counter_rows)if(z_num_*x_num_ > 100)
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
        total_number_non_empty_rows = counter_rows;
        apr_timer.stop_timer();

    }


    template<typename T>
    void allocate_map(APR<T>& apr,MapStorageData& map_data,std::vector<uint64_t>& cumsum){

        //first add the layers
        gap_map.depth_max = level_max;
        gap_map.depth_min = level_min;

        gap_map.z_num.resize(gap_map.depth_max+1);
        gap_map.x_num.resize(gap_map.depth_max+1);
        gap_map.y_num.resize(gap_map.depth_max+1);

        gap_map.data.resize(gap_map.depth_max+1);

        gap_map.org_dims.resize(3);
        gap_map.org_dims[0] = org_dims[0];
        gap_map.org_dims[1] = org_dims[1];
        gap_map.org_dims[2] = org_dims[2];

        for(uint64_t i = gap_map.depth_min;i <= gap_map.depth_max;i++){
            gap_map.z_num[i] = z_num[i];
            gap_map.x_num[i] = x_num[i];
            gap_map.y_num[i] = y_num[i];

            gap_map.data[i].resize(z_num[i]*x_num[i]);

        }

        uint64_t j;
#ifdef HAVE_OPENMP
        #pragma omp parallel for default(shared) schedule(static) private(j)
#endif
        for (j = 0; j < total_number_non_empty_rows; ++j) {

            const uint64_t level = map_data.level[j];

            const uint64_t offset_pc_data =  x_num[level]* map_data.z[j] + map_data.x[j];

            const uint64_t global_begin = cumsum[j];

            const uint64_t number_gaps = map_data.number_gaps[j];

            YGap_map gap;

            gap_map.data[level][offset_pc_data].resize(1);

            for (uint64_t i = global_begin; i < (global_begin + number_gaps) ; ++i) {
                gap.y_end = map_data.y_end[i];
                gap.global_index_begin = map_data.global_index[i];

                auto hint = gap_map.data[level][offset_pc_data][0].map.end();
                gap_map.data[level][offset_pc_data][0].map.insert(hint,{map_data.y_begin[i],gap});
            }

        }



    }

    template<typename T>
    void rebuild_map(APR<T>& apr,MapStorageData& map_data){

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


        allocate_map(apr,map_data,cumsum);

        apr_timer.start_timer("forth loop");
        //////////////////
        ///
        /// Recalculate the iteration helpers
        ///
        //////////////////////

        //iteration helpers for by level
        global_index_by_level_begin.resize(level_max+1,0);
        global_index_by_level_end.resize(level_max+1,0);

        uint64_t cumsum_parts= 0;

        //set up the iteration helpers for by zslice
        global_index_by_level_and_z_begin.resize(level_max+1);
        global_index_by_level_and_z_end.resize(level_max+1);

        for(uint64_t i = level_min;i <= level_max;i++) {

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];
            //set up the levels here.

            uint64_t cumsum_begin = cumsum_parts;

            global_index_by_level_and_z_begin[i].resize(z_num_,(-1));
            global_index_by_level_and_z_end[i].resize(z_num_,0);

            for (z_ = 0; z_ < z_num_; z_++) {
                uint64_t cumsum_begin_z = cumsum_parts;

                for (x_ = 0; x_ < x_num_; x_++) {
                    const size_t offset_pc_data = x_num_ * z_ + x_;
                    if(gap_map.data[i][offset_pc_data].size() > 0) {
                        for (auto const &element : gap_map.data[i][offset_pc_data][0].map) {
                            //count the number of particles in each gap
                            cumsum_parts += (element.second.y_end - element.first) + 1;
                        }
                    }
                }
                if(cumsum_parts!=cumsum_begin_z) {
                    global_index_by_level_and_z_end[i][z_] = cumsum_parts - 1;
                    global_index_by_level_and_z_begin[i][z_] = cumsum_begin_z;
                }
            }

            if(cumsum_parts!=cumsum_begin){
                //cumsum_begin++;
                global_index_by_level_begin[i] = cumsum_begin;
            }

            if(cumsum_parts!=cumsum_begin){
                global_index_by_level_end[i] = cumsum_parts-1;
            }
        }

        apr_timer.stop_timer();


    }


    template<typename T>
    void flatten_structure(APR<T>& apr,MapStorageData& map_data){
        //
        //  Flatten the map access structure for writing the output
        //

        map_data.y_begin.reserve(total_number_gaps);
        map_data.y_end.reserve(total_number_gaps);
        map_data.global_index.reserve(total_number_gaps);

        //total_number_non_empty_rows
        map_data.x.reserve(total_number_non_empty_rows);
        map_data.z.reserve(total_number_non_empty_rows);
        map_data.level.reserve(total_number_non_empty_rows);
        map_data.number_gaps.reserve(total_number_non_empty_rows);

        uint64_t z_;
        uint64_t x_;

        for(uint64_t i = (apr.level_min());i <= apr.level_max();i++) {

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            for (z_ = 0; z_ < z_num_; z_++) {
                for (x_ = 0; x_ < x_num_; x_++) {
                    const uint64_t offset_pc_data = x_num_ * z_ + x_;
                    if(gap_map.data[i][offset_pc_data].size()>0) {
                        map_data.x.push_back(x_);
                        map_data.z.push_back(z_);
                        map_data.level.push_back(i);
                        map_data.number_gaps.push_back(gap_map.data[i][offset_pc_data][0].map.size());

                        for (auto const &element : gap_map.data[i][offset_pc_data][0].map) {
                            map_data.y_begin.push_back(element.first);
                            map_data.y_end.push_back(element.second.y_end);
                            map_data.global_index.push_back(element.second.global_index_begin);
                        }
                    }

                }
            }
        }

    }



    template<typename T>
    void test_method(APR<T>& apr){

        x_num.resize(apr.depth_max()+1);
        y_num.resize(apr.depth_max()+1);
        z_num.resize(apr.depth_max()+1);

        for (int level = apr.depth_min(); level <= apr.depth_max(); level++) {
            x_num[level] = apr.spatial_index_x_max(level);
            y_num[level] = apr.spatial_index_y_max(level);
            z_num[level] = apr.spatial_index_z_max(level);
        }


        //initialize variables required
        uint64_t node_val_pc; // node variable encoding neighbour and cell information

        uint64_t x_; // iteration variables
        uint64_t z_; // iteration variables
        uint64_t j_; // index variable
        uint64_t curr_key = 0; // key used for accessing and particles and cells
        PartCellNeigh<uint64_t> neigh_cell_keys;

        uint64_t y_coord = 0;

        std::vector<uint16_t> neighbours;

        apr.get_part_numbers();

        neighbours.resize(apr.num_parts_total);

        ExtraPartCellData<YGap> ygaps;
        ygaps.initialize_structure_parts_empty(apr.particles_int_old);

//    ExtraPartCellData<uint64_t> gaps_end;
//    gaps_end.initialize_structure_parts_empty(apr.particles_intensities);
//
//    ExtraPartCellData<uint64_t> index;
//    index.initialize_structure_parts_empty(apr.particles_intensities);

        std::cout << "Number of particles: " << apr.num_parts_total << std::endl;


        uint64_t count_gaps=0;
        uint64_t count_parts = 0;

        PartCellData<uint64_t> pc_data;

        for(uint64_t i = apr.depth_min();i <= apr.depth_max();i++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr.spatial_index_x_max(i);
            const unsigned int z_num_ = apr.spatial_index_z_max(i);

//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                curr_key = 0;

                pc_data.pc_key_set_z(curr_key,z_);
                pc_data.pc_key_set_depth(curr_key,i);

                for(x_ = 0;x_ < x_num_;x_++){

                    pc_data.pc_key_set_x(curr_key,x_);

                    const uint64_t offset_pc_data = x_num_*z_ + x_;

                    const uint64_t j_num = apr.pc_data.data[i][offset_pc_data].size();


                    uint64_t prev = 0;

                    YGap gap;

                        gap.y_begin = 0;
                        gap.y_end = 0;
                        gap.global_index_begin = 0;

                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < j_num;j_++){

                        float part_int= 0;

                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        node_val_pc = apr.pc_data.data[i][offset_pc_data][j_];



                        if (!(node_val_pc&1)){
                            //Indicates this is a particle cell node
                            y_coord++;
                            count_parts++;

                            uint16_t status = (node_val_pc & STATUS_MASK) >> STATUS_SHIFT;
                            uint16_t type = (node_val_pc & TYPE_MASK) >> TYPE_SHIFT;

                            uint16_t xp_j= (node_val_pc & XP_INDEX_MASK) >> XP_INDEX_SHIFT;
                            uint16_t xp_dep = (node_val_pc & XP_DEPTH_MASK) >> XP_DEPTH_SHIFT;

                            uint16_t zp_j = (node_val_pc & ZP_INDEX_MASK) >> ZP_INDEX_SHIFT;
                            uint16_t zp_dep = (node_val_pc & ZP_DEPTH_MASK) >> ZP_DEPTH_SHIFT;

                            uint16_t m_j = (node_val_pc & XM_INDEX_MASK) >> XM_INDEX_SHIFT;
                            uint16_t xm_dep = (node_val_pc & XM_DEPTH_MASK) >> XM_DEPTH_SHIFT;

                            uint16_t zm_j = (node_val_pc & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
                            uint16_t zm_dep = (node_val_pc & ZM_DEPTH_MASK) >> ZM_DEPTH_SHIFT;

//
//                        pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_GAP;
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= (((uint64_t)y_) << NEXT_COORD_SHIFT);
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= ( prev_coord << PREV_COORD_SHIFT);
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);

                            neighbours[count_parts-1] |= (xp_dep << XP_LEVEL_SHIFT);
                            neighbours[count_parts-1] |= (xm_dep << XM_LEVEL_SHIFT);
                            neighbours[count_parts-1] |= (zp_dep << ZP_LEVEL_SHIFT);
                            neighbours[count_parts-1] |= (zm_dep << ZM_LEVEL_SHIFT);
                            neighbours[count_parts-1] |= (status << PC_TYPE_SHIFT);

                            if(prev == 0){
                                //add a y same flag

                                neighbours[count_parts-1] |= (_LEVEL_SAME << YM_LEVEL_SHIFT);

                                neighbours[count_parts-2] |= (_LEVEL_SAME << YP_LEVEL_SHIFT);

                            }

                            //NEED TO SET YP
                            prev = 0;

                        } else {
                            // Inidicates this is not a particle cell node, and is a gap node

                            prev = 1;

                            uint16_t type = (node_val_pc & TYPE_MASK) >> TYPE_SHIFT;

                            uint16_t yp_j = (node_val_pc & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
                            uint16_t yp_dep = (node_val_pc & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;

                            uint16_t ym_j = (node_val_pc & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
                            uint16_t ym_dep = (node_val_pc & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;

                            uint64_t next_y = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;

                            uint64_t prev_y = (node_val_pc & PREV_COORD_MASK) >> PREV_COORD_SHIFT;

                            //yp_dep = yp_dep + 2;


                            if((j_ == 0) & (j_num > 1)){
                                //first node (do forward) (YM)
                                neighbours[count_parts] |= (ym_dep << YM_LEVEL_SHIFT);

                            } else if ((j_ == (j_num-1)) & (j_num > 1)){
                                //last node (do behind) (YP)
                                neighbours[count_parts-1] |= (yp_dep << YP_LEVEL_SHIFT);



                            } else if (j_num > 1){
                                // front (YM) and behind (YP)



                                neighbours[count_parts] |= (ym_dep << YM_LEVEL_SHIFT);
                                neighbours[count_parts-1] |= (yp_dep << YP_LEVEL_SHIFT);

                            }


                            if(j_>0){
                                //gaps_end.data[i][offset_pc_data].push_back(y_coord);
                                gap.y_end =  (uint16_t)(y_coord);
                                ygaps.data[i][offset_pc_data].push_back(gap);

                            }

                            y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                            if(j_num > 1) {
                                if(j_ < (j_num - 1)) {
                                    count_gaps++;

                                    gap.y_begin = (uint16_t)(y_coord + 1);
                                    gap.global_index_begin = count_parts;
                                    //gaps.data[i][offset_pc_data].push_back(y_coord+1);
                                    //index.data[i][offset_pc_data].push_back(count_parts);
                                }

                            }
                        }
                    }
                }
            }
        }

        std::cout << count_gaps << std::endl;
        std::cout << count_parts << std::endl;

        std::vector<uint16_t> pint;
        pint.reserve(count_parts);

        std::vector<uint16_t> px;
        px.reserve(count_parts);

        std::vector<uint16_t> py;
        py.reserve(count_parts);

        std::vector<uint16_t> pz;
        pz.reserve(count_parts);

        APRTimer timer;

        timer.verbose_flag = true;

        timer.start_timer("iterate old");

        for (apr.begin();apr.end()!=0 ;apr.it_forward()) {
            pint.push_back(apr(apr.particles_int_old));
            px.push_back(apr.x());
            py.push_back(apr.y());
            pz.push_back(apr.z());
        }

        timer.stop_timer();

        std::vector<uint16_t> pint2;
        pint2.reserve(count_parts);

        std::vector<uint16_t> px2;
        px2.reserve(count_parts);

        std::vector<uint16_t> py2;
        py2.reserve(count_parts);

        std::vector<uint16_t> pz2;
        pz2.reserve(count_parts);

        //now how do I check they are right?, compare with the old structure and request the information out.

        std::vector<uint16_t> shift = {YP_LEVEL_SHIFT,YM_LEVEL_SHIFT,XP_LEVEL_SHIFT,XM_LEVEL_SHIFT,ZP_LEVEL_SHIFT,ZM_LEVEL_SHIFT};
        std::vector<uint16_t> mask = {YP_LEVEL_MASK,YM_LEVEL_MASK,XP_LEVEL_MASK,XM_LEVEL_MASK,ZP_LEVEL_MASK,ZM_LEVEL_MASK};

        apr.set_part_numbers_xz();

        APRIteratorOld<uint16_t> neighbour_iterator(apr);

        uint64_t c = 0;
        uint64_t nn= 0;

        std::vector<unsigned int > dir_vec = {0,1,2,3,4,5};

        for (apr.begin();apr.end()!=0 ;apr.it_forward()) {

            uint16_t node = neighbours[c];

            //now we only update the neighbours, and directly access them through a neighbour iterator
            apr.update_all_neighbours();

            uint16_t type = (node & PC_TYPE_MASK) >> PC_TYPE_SHIFT;

            if(type!= apr.type()){
                std::cout << "type broke" << std::endl;
            }

            //loop over all the neighbours and set the neighbour iterator to it
            for (int f = 0; f < dir_vec.size(); ++f) {
                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                unsigned int dir = dir_vec[f];

                int node_depth_dif = (node & mask[dir]) >> shift[dir];
                int depth_dif = _NO_NEIGHBOUR;

                for (int index = 0; index < apr.number_neighbours_in_direction(dir); ++index) {
                    // on each face, there can be 0-4 neighbours accessed by index
                    if(neighbour_iterator.set_neighbour_iterator(apr, dir, index)){
                        //will return true if there is a neighbour defined

                        depth_dif =  neighbour_iterator.level() - apr.level() + 1;

                    }
                }

                if(depth_dif < 3){
                    nn++;
                    //std::cout << "number wrong" << std::endl;
                }


                if(node_depth_dif!=depth_dif){
                    std::cout << depth_dif << " " << node_depth_dif << std::endl;
                }


                //compare with new neighbour structure;

            }

            c++;
        }
        std::cout << nn << std::endl;



        gap_map.initialize_structure_parts_empty(apr.particles_int_old);

        YIterators it;
        //ExtraPartCellData<std::map<uint16_t,YGap_map>::iterator> gap_map_it;

        it.gap_map_it.initialize_structure_parts_empty(apr.particles_int_old);

        uint64_t counter_stop = 0;

        for(uint64_t i = apr.depth_min();i <= apr.depth_max();i++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr.spatial_index_x_max(i);
            const unsigned int z_num_ = apr.spatial_index_z_max(i);

//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
            for (z_ = 0; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure
                curr_key = 0;

                for (x_ = 0; x_ < x_num_; x_++) {


                    const uint64_t offset_pc_data = x_num_ * z_ + x_;

                    const uint64_t gap_num = ygaps.data[i][offset_pc_data].size();

                    //YGap_map ygap;
                    YGap old_gap;

                    if(gap_num > 0){

                        gap_map.data[i][offset_pc_data].resize(1);


                        for (int j = 0; j < gap_num; ++j) {
                            old_gap = ygaps.data[i][offset_pc_data][j];
                            YGap_map ygap;
                            ygap.global_index_begin = old_gap.global_index_begin;
                            ygap.y_end = old_gap.y_end;

                            uint16_t y_begin = old_gap.y_begin;

                            if(counter_stop==10000){
                                std::cout << y_begin << " " << ygap.y_end << " " << ygap.global_index_begin << std::endl;
                            }

                            //gap_map.data[i][offset_pc_data][0].map[old_gap.y_begin] = ygap;

                            gap_map.data[i][offset_pc_data][0].map.insert(std::pair<uint16_t,YGap_map>(y_begin,ygap));

                            counter_stop++;
                        }
                        //initialize the iterator

                    }

                }
            }
        }

//        timer.start_timer("iterate new");
//
//        uint64_t counter_new = -1;
//
//        for(uint64_t i = apr.depth_min();i <= apr.depth_max();i++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = apr.spatial_index_x_max(i);
//            const unsigned int z_num_ = apr.spatial_index_z_max(i);
//
////#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
//            for (z_ = 0; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                for (x_ = 0; x_ < x_num_; x_++) {
//
//                    const uint64_t offset_pc_data = x_num_*z_ + x_;
//
//                    if(gap_map.data[i][offset_pc_data].size() > 0){
//
//                        MapIterator it;
//
//
//                        for ( it.iterator = gap_map.data[i][offset_pc_data][0].map.begin(); it.iterator != gap_map.data[i][offset_pc_data][0].map.end();it.iterator++) {
//
//                            YGap_map gap = it.iterator->second;
//                            uint16_t y_begin = it.iterator->first;
//
//                            uint64_t curr_index = gap.global_index_begin;
//
//                            curr_index--;
//
//                            for (uint16_t y = y_begin;
//                                 y <= gap.y_end; y++) {
//
//                                curr_index++;
//                                counter_new++;
//
//                                pint2.push_back(pint[curr_index]);
//                                px2.push_back(x_);
//                                pz2.push_back(z_);
//                                py2.push_back(y);
//
//                            }
//
//                        }
//                    }
//                }
//            }
//        }
//
//        timer.stop_timer();
//
//        std::cout << counter_new << std::endl;
//
//        /////////////////
//        ///
//        /// Checking everything is okay here..
//        ///
//        ///////////////////
//
//        bool broken = false;
//
//        for (int k = 0; k < count_parts; ++k) {
//
//            if(pint[k] != pint2[k]){
//                std::cout << "intbroke" << std::endl;
//            }
//
//            if(py[k] != py2[k]){
//                std::cout << "ybroke" << std::endl;
//                broken = true;
//            }
//
//            if(px[k] != px2[k]){
//                std::cout << "xbroke" << std::endl;
//            }
//
//            if(pz[k] != pz2[k]){
//                std::cout << "zbroke" << std::endl;
//            }
//
//        }
//
//        if(broken){
//            std::cout << "ybroken" << std::endl;
//        }
//
//        std::cout << py2.size() << std::endl;
//        std::cout << py.size() << std::endl;
//
//        //return;
//
//        float num_rep = 20;
//
//        uint64_t counter_n1= 0;
//
//        uint64_t q = 0;
//
//        std::vector<int> neigh_count;
//        neigh_count.resize(apr.num_parts_total,0);
//
//        std::vector<int> neigh_count2;
//        neigh_count2.resize(apr.num_parts_total,0);
//
//        ExtraPartCellData<float> neigh_sum(apr);
//
//        std::vector<float> neigh_sum_new;
//        neigh_sum_new.resize(apr.num_parts_total,0);
//
//
//        //initialization of the iteration structures
//        APRIterator<uint16_t> apr_parallel_iterator(apr); //this is required for parallel access
//        uint64_t part; //declare parallel iteration variable
//
//        ExtraPartCellData<float> neigh_xm(apr);
//
//        timer.start_timer("APR parallel iterator neighbour loop");
//
//        for (int l = 0; l < num_rep; ++l) {
//
//#pragma omp parallel for schedule(static) private(part) firstprivate(apr_parallel_iterator,neighbour_iterator)
//            for (part = 0; part < apr.num_parts_total; ++part) {
//                //needed step for any parallel loop (update to the next part)
//
//                apr_parallel_iterator.set_iterator_to_particle_by_number(part);
//
//                //compute neighbours as previously, now using the apr_parallel_iterator (APRIterator), instead of the apr class for access.
//                apr_parallel_iterator.update_all_neighbours();
//
//                float temp = 0;
//                float counter = 0;
//
//                //loop over all the neighbours and set the neighbour iterator to it
//                for (int dir = 0; dir < 6; ++dir) {
//                    for (int index = 0; index < apr_parallel_iterator.number_neighbours_in_direction(dir); ++index) {
//
//                        if (neighbour_iterator.set_neighbour_iterator(apr_parallel_iterator, dir, index)) {
//                            //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
//                            temp += neighbour_iterator(apr.particles_intensities);
//                            counter++;
//                        }
//
//                    }
//                }
//
//                apr_parallel_iterator(neigh_xm) = temp / counter;
//
//            }
//        }
//
//        timer.stop_timer();
//
////        timer.start_timer("APR serial iterator neighbours loop");
////
////        for (int l = 0; l < num_rep; ++l) {
////
////            //Basic serial iteration over all particles
////            for (apr.begin(); apr.end() != 0; apr.it_forward()) {
////
////                //now we only update the neighbours, and directly access them through a neighbour iterator
////                apr.update_all_neighbours();
////
////                float counter = 0;
////                float temp = 0;
////
////                //loop over all the neighbours and set the neighbour iterator to it
////                for (int dir = 0; dir < 6; ++dir) {
////                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
////
////                    for (int index = 0; index < apr.number_neighbours_in_direction(dir); ++index) {
////                        // on each face, there can be 0-4 neighbours accessed by index
////                        if (neighbour_iterator.set_neighbour_iterator(apr, dir, index)) {
////                            //will return true if there is a neighbour defined
////                            temp+=neighbour_iterator(apr.particles_int);
////
////                            counter++;
////
////                        }
////                    }
////                }
////
////                apr(neigh_sum) = temp/counter;
////
////            }
////
////        }
////
////        timer.stop_timer();
//
//        std::cout << q << std::endl;
//
//        ////////////////////////////
//        ///
//        /// Prototype neighbour access
//        ///
//        //////////////////////////
//
//        ParticleCell input;
//        ParticleCell neigh;
//
//        //initialize_neigh(gap_map);
//
//        timer.start_timer("new neighbour loop");
//
//        uint64_t counter_n= 0;
//        uint64_t counter_t = 0;
//
//        LocalMapIterators local_iterators;
//
//        for (int l = 0; l < num_rep; ++l) {
//
//            uint64_t neigh_count = 0;
//
//            for (uint64_t i = apr.depth_min(); i <= apr.depth_max(); i++) {
//                //loop over the resolutions of the structure
//                const unsigned int x_num_ = apr.spatial_index_x_max(i);
//                const unsigned int z_num_ = apr.spatial_index_z_max(i);
//
//                input.level = i;
//
//#pragma omp parallel for schedule(static) default(shared) private(z_,x_) reduction(+:neigh_count) firstprivate(input,neigh,local_iterators)
//                for (z_ = 0; z_ < z_num_; z_++) {
//                    //both z and x are explicitly accessed in the structure
//
//                    input.z = z_;
//
//                    for (x_ = 0; x_ < x_num_; x_++) {
//
//                        const size_t offset_pc_data = x_num_ * z_ + x_;
//
//                        input.x = x_;
//
//                        if(gap_map.data[i][offset_pc_data].size() > 0){
//
//                            for ( const auto &p : gap_map.data[i][offset_pc_data][0].map ) {
//
//                                YGap_map gap = p.second;
//                                uint16_t y_begin = p.first;
//
//                                uint64_t curr_index = gap.global_index_begin;
//
//                                curr_index--;
//
//                                for (int y = y_begin;
//                                     y <= gap.y_end; y++) {
//
//                                    curr_index++;
//
//                                    input.y = y;
//
//                                    uint16_t node = neighbours[curr_index];
//
//                                    float counter = 0;
//                                    float temp = 0;
//
//                                    for (int f = 0; f < dir_vec.size(); ++f) {
//                                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                                        unsigned int face = dir_vec[f];
//
//                                        uint16_t level_delta = (node & mask[face]) >> shift[face];
//
//                                        for (int n = 0; n < number_neighbours_in_direction(level_delta); ++n) {
//                                            get_neighbour_coordinate(input, neigh, face, level_delta, n);
//                                            if(n> 0) {
//                                                if (neigh.x < apr.spatial_index_x_max(neigh.level) ){
//                                                    if (neigh.z < apr.spatial_index_z_max(neigh.level)) {
//
//                                                        neigh.pc_offset =
//                                                                apr.spatial_index_x_max(neigh.level)* neigh.z + neigh.x;
//
//                                                        if (find_particle_cell(neigh, get_local_iterator(local_iterators,level_delta,face,n))) {
//                                                            // do something;
//                                                            temp+=pint[neigh.global_index];
//                                                            counter++;
//                                                            neigh_count++;
//                                                        }
//
//                                                    }
//                                                }
//                                            } else {
//
//                                                neigh.pc_offset =
//                                                        apr.spatial_index_x_max(neigh.level) * neigh.z + neigh.x;
//                                                if (find_particle_cell(neigh, get_local_iterator(local_iterators,level_delta,face,n))) {
//                                                    // do something;
//                                                    temp+=pint[neigh.global_index];
//                                                    counter++;
//                                                    neigh_count++;
//
//                                                }
//
//                                            }
//
//                                        }
//
//                                    }
//
//                                    neigh_sum_new[curr_index] = temp/counter;
//
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//
//            std::cout << neigh_count << std::endl;
//
//        }
//
//        timer.stop_timer();
//
//        std::cout << counter_n << std::endl;
//        std::cout << counter_t << std::endl;
//
//
//        for (int m = 0; m < neigh_count.size(); ++m) {
//            if(neigh_count[m]!=neigh_count2[m]){
//                //std::cout << neigh_count[m] << " " << neigh_count2[m] << std::endl;
//            }
//        }
//
//
//        //////////////////////////////////////////
//        ///
//        ///
//        /// Check the loop
//        ///
//        ///
//        ///////////////////////////////////
//
//        ExtraPartCellData<uint64_t> index_vec(apr);
//
//        uint64_t cp = 0;
//
//        for (apr.begin();apr.end()!=0 ;apr.it_forward()) {
//            apr(index_vec) = cp;
//            cp++;
//        }
//
//
//        MeshData<uint64_t> index_image;
//
//        index_image.initialize(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2));
//
//
//        //CHECK THE CHECKING SCHEME FIRST
//
//        //Basic serial iteration over all particles
//        for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//            //now we only update the neighbours, and directly access them through a neighbour iterator
//            apr.update_all_neighbours();
//
//            float counter = 0;
//            float temp = 0;
//
//            uint16_t x_global = apr.x_nearest_pixel();
//            uint16_t y_global = apr.y_nearest_pixel();
//            uint16_t z_global = apr.z_nearest_pixel();
//
//            index_image(y_global,x_global,z_global) = apr(index_vec);
//
//        }
//
//        //Basic serial iteration over all particles
//        for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//            //now we only update the neighbours, and directly access them through a neighbour iterator
//            apr.update_all_neighbours();
//
//            float counter = 0;
//            float temp = 0;
//
//            //loop over all the neighbours and set the neighbour iterator to it
//            for (int dir = 0; dir < 6; ++dir) {
//                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                for (int index = 0; index < apr.number_neighbours_in_direction(dir); ++index) {
//                    // on each face, there can be 0-4 neighbours accessed by index
//                    if(neighbour_iterator.set_neighbour_iterator(apr, dir, index)){
//                        //will return true if there is a neighbour defined
//
//                        uint16_t x_global = neighbour_iterator.x_nearest_pixel();
//                        uint16_t y_global = neighbour_iterator.y_nearest_pixel();
//                        uint16_t z_global = neighbour_iterator.z_nearest_pixel();
//
//                        uint64_t neigh_index = index_image(y_global,x_global,z_global);
//                        uint64_t neigh_truth = neighbour_iterator(index_vec);
//
//                        if(neigh_index != neigh_truth){
//                            std::cout << "test still broke" << std::endl;
//                        }
//                    }
//                }
//            }
//
//        }
//
//
//        for (uint64_t i = apr.depth_min(); i <= apr.depth_max(); i++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = apr.spatial_index_x_max(i);
//            const unsigned int z_num_ = apr.spatial_index_z_max(i);
//
//            input.level = i;
//
////#pragma omp parallel for schedule(static) default(shared) private(z_,x_)  firstprivate(input,neigh,local_iterators) if(z_num_*x_num_ > 100)
//            for (z_ = 0; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                input.z = z_;
//
//                for (x_ = 0; x_ < x_num_; x_++) {
//
//                    const size_t offset_pc_data = x_num_ * z_ + x_;
//
//                    input.x = x_;
//
//                    if(gap_map.data[i][offset_pc_data].size() > 0){
//
//                        for ( const auto &p : gap_map.data[i][offset_pc_data][0].map ) {
//
//                            YGap_map gap = p.second;
//                            uint16_t y_begin = p.first;
//
//                            uint64_t curr_index = gap.global_index_begin;
//
//                            curr_index--;
//
//                            for (int y = y_begin;
//                                 y <= gap.y_end; y++) {
//
//                                curr_index++;
//
//                                input.y = y;
//
//                                uint16_t node = neighbours[curr_index];
//
//                                for (int f = 0; f < dir_vec.size(); ++f) {
//                                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                                    unsigned int face = dir_vec[f];
//
//                                    uint16_t level_delta = (node & mask[face]) >> shift[face];
//
//                                    for (int n = 0; n < number_neighbours_in_direction(level_delta); ++n) {
//                                        get_neighbour_coordinate(input, neigh, face, level_delta, n);
//                                        if(number_neighbours_in_direction(level_delta)==4) {
//                                            if (neigh.x < apr.pc_data.x_num[neigh.level]) {
//                                                if (neigh.z < apr.pc_data.z_num[neigh.level]) {
//
//                                                    neigh.pc_offset =
//                                                            apr.pc_data.x_num[neigh.level] * neigh.z + neigh.x;
//
//
//                                                    if (find_particle_cell(neigh, get_local_iterator(local_iterators,level_delta,face,n))) {
//                                                        // do something;
//
//                                                        uint16_t x_global = floor((neigh.x+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                        uint16_t y_global = floor((neigh.y+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                        uint16_t z_global = floor((neigh.z+0.5)*pow(2, apr.level_max() - neigh.level));
//
//                                                        uint64_t neigh_index = index_image(y_global,x_global,z_global);
//                                                        uint64_t neigh_truth = neigh.global_index;
//
//                                                        if(neigh_index != neigh_truth){
//                                                            std::cout << "neigh broke" << std::endl;
//                                                        }
//
//                                                    }
//
//                                                }
//                                            }
//                                        } else {
//
//
//                                            neigh.pc_offset =
//                                                    apr.pc_data.x_num[neigh.level] * neigh.z + neigh.x;
//                                            if (find_particle_cell(neigh, get_local_iterator(local_iterators,level_delta,face,n))) {
//                                                // do something;
//
//                                                uint16_t x_global = floor((neigh.x+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                uint16_t y_global = floor((neigh.y+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                uint16_t z_global = floor((neigh.z+0.5)*pow(2, apr.level_max() - neigh.level));
//
//                                                uint64_t neigh_index = index_image(y_global,x_global,z_global);
//                                                uint64_t neigh_truth = neigh.global_index;
//
//                                                if(neigh_index != neigh_truth){
//                                                    std::cout << "neigh broke" << std::endl;
//                                                }
//
//
//                                            }
//
//                                        }
//
//                                    }
//
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//
//        initialize_pointers(it.gap_map_it);
//
//        timer.start_timer("new neighbour loop");
//
//        std::vector<float> neigh_sum_new2;
//        neigh_sum_new2.resize(apr.num_parts_total,0);
//
//        for (int l = 0; l < num_rep; ++l) {
//
//            uint64_t neigh_count = 0;
//
//            for (uint64_t i = apr.depth_min(); i <= apr.depth_max(); i++) {
//                //loop over the resolutions of the structure
//                const unsigned int x_num_ = apr.spatial_index_x_max(i);
//                const unsigned int z_num_ = apr.spatial_index_z_max(i);
//
//                input.level = i;
//
//                std::vector<uint16_t> level_check;
//
//                if(i == apr.level_max()){
//                    level_check = {_LEVEL_SAME,_LEVEL_DECREASE};
//                } else if (i == apr.level_min()){
//                    level_check = {_LEVEL_SAME,_LEVEL_INCREASE};
//                } else {
//                    level_check = {_LEVEL_SAME,_LEVEL_DECREASE,_LEVEL_INCREASE};
//                }
//
//
//#pragma omp parallel for schedule(static) default(shared) private(z_,x_)  firstprivate(input,neigh,local_iterators) reduction(+:neigh_count) if(z_num_*x_num_ > 100)
//                for (z_ = 0; z_ < z_num_; z_++) {
//                    //both z and x are explicitly accessed in the structure
//
//                    input.z = z_;
//
//                    for (x_ = 0; x_ < x_num_; x_++) {
//
//                        const size_t offset_pc_data = x_num_ * z_ + x_;
//
//                        input.x = x_;
//
//                        if(gap_map.data[i][offset_pc_data].size() > 0){
//
//                            for ( const auto &p : gap_map.data[i][offset_pc_data][0].map ) {
//
//                                YGap_map gap = p.second;
//                                uint16_t y_begin = p.first;
//
//                                uint64_t curr_index = gap.global_index_begin;
//
//                                curr_index--;
//
//                                for (int y = y_begin;
//                                     y <= gap.y_end; y++) {
//
//                                    curr_index++;
//
//                                    input.y = y;
//
//
//                                    float counter = 0;
//                                    float temp = 0;
//
//                                    bool neigh_check = check_neighbours_flag(x_,z_,i);
//
//                                    for (int f = 0; f < dir_vec.size(); ++f) {
//                                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                                        unsigned int face = dir_vec[f];
//
//                                        bool found = false;
//
//                                        for (int j = 0; j < level_check.size(); ++j) {
//                                            uint16_t level_delta = level_check[j];
//
//
//                                            for (int n = 0; n < number_neighbours_in_direction(level_delta); ++n) {
//                                                get_neighbour_coordinate(input, neigh, face, level_delta, n);
//                                                if (neigh_check) {
//                                                    if ((neigh.x < apr.spatial_index_x_max(neigh.level)) & (neigh.x >= 0)) {
//                                                        if ((neigh.z < apr.spatial_index_z_max(neigh.level)) & (neigh.z >= 0)) {
//
//                                                            neigh.pc_offset =
//                                                                    apr.spatial_index_x_max(neigh.level) * neigh.z +
//                                                                    neigh.x;
//
//                                                            if (find_particle_cell(neigh,
//                                                                                   get_local_iterator(local_iterators,
//                                                                                                      level_delta, face,
//                                                                                                      n))) {
//                                                                // do something;
//                                                                temp += pint[neigh.global_index];
//                                                                counter++;
//                                                                found = true;
//                                                                neigh_count++;
//                                                            }
//
//                                                        }
//                                                    } else {
//                                                        found = true;
//                                                    }
//                                                } else {
//
//                                                    neigh.pc_offset =
//                                                            apr.spatial_index_x_max(neigh.level) * neigh.z + neigh.x;
//                                                    if (find_particle_cell(neigh, get_local_iterator(local_iterators,
//                                                                                                     level_delta, face,
//                                                                                                     n))) {
//                                                        // do something;
//                                                        temp += pint[neigh.global_index];
//                                                        counter++;
//                                                        found = true;
//                                                        neigh_count++;
//                                                    }
//
//                                                }
//
//                                            }
//
//                                            if(found){
//                                                break;
//                                            }
//
//                                        }
//                                    }
//
//                                    neigh_sum_new2[curr_index] = temp/counter;
//
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//
//            std::cout << neigh_count << std::endl;
//        }
//
//        timer.stop_timer();
//
//
//        for (int i1 = 0; i1 < neigh_sum_new2.size(); ++i1) {
//            if(floor(neigh_sum_new[i1])!=floor(neigh_sum_new2[i1])){
//                std::cout << " ns broke " << std::endl;
//            }
//        }
//
//
//
//
//
//        for (uint64_t i = apr.depth_min(); i <= apr.depth_max(); i++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = apr.spatial_index_x_max(i);
//            const unsigned int z_num_ = apr.spatial_index_z_max(i);
//
//            input.level = i;
//
//            std::vector<uint16_t> level_check;
//
//            if(i == apr.level_max()){
//                level_check = {_LEVEL_SAME,_LEVEL_DECREASE};
//            } else if (i == apr.level_min()){
//                level_check = {_LEVEL_SAME,_LEVEL_INCREASE};
//            } else {
//                level_check = {_LEVEL_SAME,_LEVEL_DECREASE,_LEVEL_INCREASE};
//            }
//
//
//#pragma omp parallel for schedule(static) default(shared) private(z_,x_)  firstprivate(input,neigh,local_iterators) if(z_num_*x_num_ > 100)
//            for (z_ = 0; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                input.z = z_;
//
//                for (x_ = 0; x_ < x_num_; x_++) {
//
//                    const size_t offset_pc_data = x_num_ * z_ + x_;
//
//                    input.x = x_;
//
//                    if(gap_map.data[i][offset_pc_data].size() > 0){
//
//                        for ( const auto &p : gap_map.data[i][offset_pc_data][0].map ) {
//
//                            YGap_map gap = p.second;
//                            uint16_t y_begin = p.first;
//
//                            uint64_t curr_index = gap.global_index_begin;
//
//                            curr_index--;
//
//                            for (int y = y_begin;
//                                 y <= gap.y_end; y++) {
//
//                                curr_index++;
//
//                                input.y = y;
//
//
//                                float counter = 0;
//                                float temp = 0;
//
//                                bool neigh_check = check_neighbours_flag(x_,z_,i);
//
//                                for (int f = 0; f < dir_vec.size(); ++f) {
//                                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                                    unsigned int face = dir_vec[f];
//
//                                    bool found = false;
//
//                                    for (int j = 0; j < level_check.size(); ++j) {
//                                        uint16_t level_delta = level_check[j];
//
//
//                                        for (int n = 0; n < number_neighbours_in_direction(level_delta); ++n) {
//                                            get_neighbour_coordinate(input, neigh, face, level_delta, n);
//                                            if (neigh_check) {
//                                                if ((neigh.x < apr.spatial_index_x_max(neigh.level)) & (neigh.x >= 0)) {
//                                                    if ((neigh.z < apr.spatial_index_z_max(neigh.level)) & (neigh.z >= 0)) {
//
//                                                        neigh.pc_offset =
//                                                                apr.spatial_index_x_max(neigh.level) * neigh.z +
//                                                                neigh.x;
//
//                                                        if (find_particle_cell(neigh,
//                                                                               get_local_iterator(local_iterators,
//                                                                                                  level_delta, face,
//                                                                                                  n))) {
//                                                            // do something;
//                                                            uint16_t x_global = floor((neigh.x+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                            uint16_t y_global = floor((neigh.y+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                            uint16_t z_global = floor((neigh.z+0.5)*pow(2, apr.level_max() - neigh.level));
//
//                                                            uint64_t neigh_index = index_image(y_global,x_global,z_global);
//                                                            uint64_t neigh_truth = neigh.global_index;
//
//                                                            if(neigh_index != neigh_truth){
//                                                                std::cout << "neigh broke without" << std::endl;
//                                                            }
//                                                            found = true;
//
//                                                        }
//
//                                                    }
//                                                } else {
//                                                    found = true;
//                                                }
//                                            } else {
//
//                                                neigh.pc_offset =
//                                                        apr.spatial_index_x_max(neigh.level) * neigh.z + neigh.x;
//                                                if (find_particle_cell(neigh, get_local_iterator(local_iterators,
//                                                                                                 level_delta, face,
//                                                                                                 n))) {
//                                                    uint16_t x_global = floor((neigh.x+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                    uint16_t y_global = floor((neigh.y+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                    uint16_t z_global = floor((neigh.z+0.5)*pow(2, apr.level_max() - neigh.level));
//
//                                                    uint64_t neigh_index = index_image(y_global,x_global,z_global);
//                                                    uint64_t neigh_truth = neigh.global_index;
//
//                                                    if(neigh_index != neigh_truth){
//                                                        std::cout << "neigh broke without" << std::endl;
//                                                    }
//                                                    found = true;
//
//                                                }
//
//                                            }
//
//                                        }
//
//                                        if(found){
//                                            break;
//                                        }
//
//                                    }
//                                }
//
//
//
//                            }
//                        }
//                    }
//                }
//            }
//        }



    }




};


#endif //PARTPLAY_APRACCESS_HPP
