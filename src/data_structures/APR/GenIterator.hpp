//
// Created by cheesema on 29.06.18.
//

#ifndef LIBAPR_GENITERATOR_HPP
#define LIBAPR_GENITERATOR_HPP

class APRAccess;

class GenIterator {

protected:

    LocalMapIterators local_iterators;

    ParticleCell neighbour_particle_cell{ 0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };

    ParticleCell current_particle_cell{0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };

    APRAccess* apr_access;

    uint16_t level_delta{};

    MapIterator current_gap;

    bool check_neigh_flag = false;

    bool check_neighbours_particle_cell_in_bounds();

public:

    uint64_t end_index = 0;

    uint64_t total_number_particles();

    uint64_t particles_level_begin(const uint16_t& level_);
    uint64_t particles_level_end(const uint16_t& level_);

    uint64_t particles_z_begin(const uint16_t& level_,const uint64_t& z_);
    uint64_t particles_z_end(const uint16_t& level_,const uint64_t& z_);

    uint64_t particles_zx_begin(const uint16_t& level_,const uint64_t& z_,const uint64_t& x_);
    uint64_t particles_zx_end(const uint16_t& level_,const uint64_t& z_,const uint64_t& x_);

    uint64_t particles_offset_end(const uint16_t& level,const uint64_t& offset);

    uint16_t x() const { return current_particle_cell.x; }
    uint16_t y() const { return current_particle_cell.y; }
    uint16_t z() const { return current_particle_cell.z; }
    uint16_t level() const { return current_particle_cell.level; }
    uint64_t global_index() const { return current_particle_cell.global_index; }
    operator uint64_t() { return current_particle_cell.global_index; }

    ParticleCell get_neigh_particle_cell();
    ParticleCell get_current_particle_cell();
    MapIterator& get_current_gap();

    uint8_t number_neighbours_in_direction(const uint8_t& face);

    unsigned int x_nearest_pixel();
    float x_global();
    unsigned int y_nearest_pixel();
    float y_global();
    unsigned int z_nearest_pixel();
    float z_global();

    uint16_t level_min();
    uint16_t level_max();

    uint64_t spatial_index_x_max(const unsigned int level);
    uint64_t spatial_index_y_max(const unsigned int level);
    uint64_t spatial_index_z_max(const unsigned int level);

    inline void set_neighbour_flag();

    GenIterator(){
        //default constructor, for use by inherited classes
    }
};


#include "APRAccess.hpp"

inline uint64_t GenIterator::total_number_particles(){
    return apr_access->total_number_particles;
}

inline uint64_t GenIterator::particles_level_begin(const uint16_t& level_){
    //
    //  Used for finding the starting particle on a given level
    //
    return apr_access->global_index_by_level_begin[level_];
}

inline uint64_t GenIterator::particles_level_end(const uint16_t& level_){
    //
    //  Find the last particle on a given level
    //
    return (apr_access->global_index_by_level_end[level_]+1l);
}

inline uint64_t GenIterator::particles_zx_begin(const uint16_t& level_,const uint64_t& z_,const uint64_t& x_){
    //
    //  Used for finding the starting particle on a given level
    //

    return apr_access->get_parts_start(x_,z_,level_);
}

inline uint64_t GenIterator::particles_zx_end(const uint16_t& level_,const uint64_t& z_,const uint64_t& x_){
    //
    //  Used for finding the starting particle on a given level
    //

    return apr_access->get_parts_end(x_,z_,level_)+1l;
}

inline uint64_t GenIterator::particles_offset_end(const uint16_t& level,const uint64_t& offset){
    //
    //  Used for finding the starting particle on a given level
    //

    if(apr_access->gap_map.data[level][offset].size() > 0){
        auto it = apr_access->gap_map.data[level][offset][0].map.rbegin();
        return (it->second.global_index_begin_offset + (it->second.y_end-it->first)); //#check me
    } else {
        return 0;
    }
}

inline ParticleCell GenIterator::get_neigh_particle_cell(){
    return neighbour_particle_cell;
}

inline ParticleCell GenIterator::get_current_particle_cell(){
    return current_particle_cell;
}

inline MapIterator& GenIterator::get_current_gap(){
    return current_gap;
}

inline uint8_t GenIterator::number_neighbours_in_direction(const uint8_t& face){
    switch (level_delta){
        case _LEVEL_INCREASE:
            return 4;
        case _NO_NEIGHBOUR:
            return 0;
    }
    return 1;
}

inline unsigned int GenIterator::x_nearest_pixel(){
    return floor((current_particle_cell.x+0.5)*pow(2, apr_access->l_max - current_particle_cell.level));
}

inline float GenIterator::x_global(){
    return (current_particle_cell.x+0.5)*pow(2, apr_access->l_max - current_particle_cell.level);
}

inline unsigned int GenIterator::y_nearest_pixel(){
    return floor((current_particle_cell.y+0.5)*pow(2, apr_access->l_max - current_particle_cell.level));
}

inline float GenIterator::y_global(){
    return (current_particle_cell.y+0.5)*pow(2, apr_access->l_max - current_particle_cell.level);
}

inline unsigned int GenIterator::z_nearest_pixel(){
    return floor((current_particle_cell.z+0.5)*pow(2, apr_access->l_max - current_particle_cell.level));
}

inline float GenIterator::z_global(){
    return (current_particle_cell.z+0.5)*pow(2, apr_access->l_max - current_particle_cell.level);
}

inline uint16_t GenIterator::level_min(){
    return apr_access->l_min;
}

inline uint16_t GenIterator::level_max(){
    return apr_access->l_max;
}

inline uint64_t GenIterator::spatial_index_x_max(const unsigned int level){
    return apr_access->x_num[level];
}

inline uint64_t GenIterator::spatial_index_y_max(const unsigned int level){
    return apr_access->y_num[level];
}

inline uint64_t GenIterator::spatial_index_z_max(const unsigned int level){
    return apr_access->z_num[level];
}

inline bool GenIterator::check_neighbours_particle_cell_in_bounds(){
    //uses the fact that the coordinates have unsigned type, and therefore if they are negative they will be above the bound
    if(check_neigh_flag) {
        return (neighbour_particle_cell.x < apr_access->x_num[neighbour_particle_cell.level]) &
               (neighbour_particle_cell.z < apr_access->z_num[neighbour_particle_cell.level]);
    } else {
        return true;
    }
}


inline void GenIterator::set_neighbour_flag(){
    check_neigh_flag = apr_access->check_neighbours_flag(current_particle_cell.x,current_particle_cell.z,current_particle_cell.level);
}



#endif //LIBAPR_GENITERATOR_HPP
