//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APRACCESS_HPP
#define PARTPLAY_APRACCESS_HPP



#include <map>
#include <utility>
#include "src/data_structures/Mesh/MeshData.hpp"

//TODO: IT SHOULD NOT BE DEFINDED HERE SINCE IT DUPLICATES FROM PullingScheme
#define SEED_TYPE 1

#define _NO_NEIGHBOUR ((uint16_t)3)
#define _LEVEL_SAME ((uint16_t)1)
#define _LEVEL_DECREASE ((uint16_t)0)
#define _LEVEL_INCREASE ((uint16_t)2)

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


#include "APR.hpp"
#include "ExtraParticleData.hpp"
#include "ExtraPartCellData.hpp"


struct ParticleCell {
    uint16_t x,y,z,level,type;
    uint64_t pc_offset,global_index;
};

struct YGap_map {
    uint16_t y_end;
    uint64_t global_index_begin;
};

struct ParticleCellGapMap{
    std::map<uint16_t,YGap_map> map;
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
                //if(map_iterator.iterator != current_pc_map.map.end()){
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

                //}

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

        std::vector<ArrayWrapper<uint8_t>> p_map;
        p_map.resize(level_max);

        for (int k = 0; k < level_max; ++k) {
            p_map[k].swap(layers[k].mesh);
        }

        initialize_structure_from_particle_cell_tree(apr, p_map);
    }


    template<typename T>
    void initialize_structure_from_particle_cell_tree(APR<T>& apr,std::vector<ArrayWrapper<uint8_t>>& p_map) {
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

                        if (status == SEED_TYPE) {
                            p_map[i][offset_part_map + 2 * y_] = seed_us;
                            p_map[i][offset_part_map + 2 * y_ + 1] = seed_us;
                        }
                    }
                }

            }
        }

        apr_timer.stop_timer();

        apr_timer.start_timer("second_step");

        ExtraPartCellData<std::pair<uint16_t,YGap_map>> y_begin(apr);

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
                const uint64_t offset_pc_data1 = std::min((uint64_t)x_num_us*(2*z_) + (2*x_),(uint64_t) x_num_us*z_num_us - 1);


                uint16_t current = 0;
                uint16_t previous = 0;

                YGap_map gap;
                gap.global_index_begin = 0;

                uint64_t counter = 0;

                for (y_ = 0; y_ < y_num_; y_++) {

                    status = p_map[i][offset_part_map + y_];
                    if(status ==SEED_TYPE) {
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
                const uint64_t offset_pc_data1 = std::min((uint64_t)x_num_us*(2*z_) + (2*x_),(uint64_t) x_num_us*z_num_us - 1);
                const uint64_t offset_pc_data2 = std::min((uint64_t)x_num_us*(2*z_) + (2*x_+1),(uint64_t) x_num_us*z_num_us - 1);
                const uint64_t offset_pc_data3 = std::min((uint64_t)x_num_us*(2*z_+1) + (2*x_),(uint64_t) x_num_us*z_num_us - 1);
                const uint64_t offset_pc_data4 = std::min((uint64_t)x_num_us*(2*z_+1) + (2*x_+1),(uint64_t) x_num_us*z_num_us - 1);

                YGap_map gap;
                gap.global_index_begin = 0;

                size_t size_v = y_begin.data[i+1][offset_pc_data1].size();

                y_begin.data[i+1][offset_pc_data2].resize(size_v);
                std::copy(y_begin.data[i+1][offset_pc_data1].begin(),y_begin.data[i+1][offset_pc_data1].end(),y_begin.data[i+1][offset_pc_data2].begin());

                y_begin.data[i+1][offset_pc_data3].resize(size_v);
                std::copy(y_begin.data[i+1][offset_pc_data1].begin(),y_begin.data[i+1][offset_pc_data1].end(),y_begin.data[i+1][offset_pc_data3].begin());

                y_begin.data[i+1][offset_pc_data4].resize(size_v);
                std::copy(y_begin.data[i+1][offset_pc_data1].begin(),y_begin.data[i+1][offset_pc_data1].end(),y_begin.data[i+1][offset_pc_data4].begin());
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

                particle_cell_type[apr_iterator] = p_map[apr_iterator.level()][offset_part_map + apr_iterator.y()];
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
        gap_map.data.resize(gap_map.depth_max+1);

        for(uint64_t i = gap_map.depth_min;i <= gap_map.depth_max;i++){
            gap_map.z_num[i] = z_num[i];
            gap_map.x_num[i] = x_num[i];
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
};


#endif //PARTPLAY_APRACCESS_HPP
