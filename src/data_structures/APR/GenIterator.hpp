//
// Created by cheesema on 29.06.18.
//

#ifndef LIBAPR_GENITERATOR_HPP
#define LIBAPR_GENITERATOR_HPP

#include "APR.hpp"
#include "APRAccess.hpp"
#include "APRTree.hpp"

template<typename ImageType>
class GenIterator {

protected:

    LocalMapIterators local_iterators;

    const uint8_t level_check_max[2] = {_LEVEL_SAME,_LEVEL_DECREASE};

    const uint8_t level_check_min[2] = {_LEVEL_SAME,_LEVEL_INCREASE};

    const uint8_t level_check_middle[3] = {_LEVEL_SAME,_LEVEL_DECREASE,_LEVEL_INCREASE};

    ParticleCell neighbour_particle_cell{ 0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };

    ParticleCell current_particle_cell{0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };

    APR<ImageType>* aprOwn;
    APRAccess* apr_access;

    uint16_t level_delta{};

    MapIterator current_gap;

    uint8_t highest_resolution_type;

    bool check_neigh_flag = false;

    const uint16_t shift[6] = {YP_LEVEL_SHIFT,YM_LEVEL_SHIFT,XP_LEVEL_SHIFT,XM_LEVEL_SHIFT,ZP_LEVEL_SHIFT,ZM_LEVEL_SHIFT};
    const uint16_t mask[6] = {YP_LEVEL_MASK,YM_LEVEL_MASK,XP_LEVEL_MASK,XM_LEVEL_MASK,ZP_LEVEL_MASK,ZM_LEVEL_MASK};


public:

    uint64_t end_index = 0;



    unsigned long number_gaps(){
        if(apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset].size() > 0) {
            return apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.size();
        } else {
            return 0;
        }
    }

    uint64_t current_gap_y_begin(){
        return current_gap.iterator->first;
    }

    uint64_t current_gap_y_end(){
        return current_gap.iterator->second.y_end;
    }

    uint64_t current_gap_index(){
        return current_gap.iterator->second.global_index_begin_offset; //#fixme
    }


    GenIterator(){
        //default constructor, for use by inherited classes
    }

    inline uint64_t particles_level_begin(const uint16_t& level_){
        //
        //  Used for finding the starting particle on a given level
        //
        return apr_access->global_index_by_level_begin[level_];
    }

    inline uint64_t particles_level_end(const uint16_t& level_){
        //
        //  Find the last particle on a given level
        //
        return (apr_access->global_index_by_level_end[level_]+1l);
    }

    inline uint64_t particles_z_begin(const uint16_t& level_,const uint64_t& z_){
        //
        //  Used for finding the starting particle on a given level
        //
        return apr_access->global_index_by_level_and_z_begin[level_][z_];
    }

    inline uint64_t particles_z_end(const uint16_t& level_,const uint64_t& z_){
        //
        //  Used for finding the starting particle on a given level
        //
        return apr_access->global_index_by_level_and_z_end[level_][z_]+1l;
    }

    inline uint64_t current_offset(){
        return current_particle_cell.pc_offset;
    }


    inline uint64_t particles_zx_begin(const uint16_t& level_,const uint64_t& z_,const uint64_t& x_){
        //
        //  Used for finding the starting particle on a given level
        //

        return apr_access->get_parts_start(x_,z_,level_);
    }

    inline uint64_t particles_zx_end(const uint16_t& level_,const uint64_t& z_,const uint64_t& x_){
        //
        //  Used for finding the starting particle on a given level
        //

        return apr_access->get_parts_end(x_,z_,level_)+1l;
    }

    inline uint64_t particles_offset_end(const uint16_t& level,const uint64_t& offset){
        //
        //  Used for finding the starting particle on a given level
        //

        if(apr_access->gap_map.data[level][offset].size() > 0){
            auto it = apr_access->gap_map.data[level][offset][0].map.rbegin();
            return (it->second.global_index_begin_offset + (it->second.y_end-it->first)); //#fixme
        } else {
            return 0;
        }

    }


    inline uint16_t x(){
        //get x
        return current_particle_cell.x;
    }

    inline uint16_t y(){
        //get x
        return current_particle_cell.y;
    }

    inline uint16_t z(){
        //get x
        return current_particle_cell.z;
    }

    inline uint8_t type(){
        //get type of the particle cell

        return highest_resolution_type;


    }

    inline uint16_t level(){
        //get x
        return current_particle_cell.level;
    }

    inline uint64_t global_index() const {
        //get x
        return current_particle_cell.global_index;
    }

    inline ParticleCell get_current_particle_cell(){
        return current_particle_cell;
    }

    inline ParticleCell get_neigh_particle_cell(){
        return neighbour_particle_cell;
    }


    bool set_neighbour_iterator(GenIterator<ImageType> &original_iterator, const uint8_t& direction, const uint8_t& index){
        //
        //  This is sets the this iterator, to the neighbour of the particle cell that original_iterator is pointing to
        //

        if(original_iterator.level_delta!=_LEVEL_INCREASE){
            //copy the information from the original iterator
            std::swap(current_particle_cell,original_iterator.neighbour_particle_cell);

        } else {
            if(index==0){
                std::swap(current_particle_cell,original_iterator.neighbour_particle_cell);

            } else {
                bool success = original_iterator.find_next_child(direction,index);
                std::swap(current_particle_cell,original_iterator.neighbour_particle_cell);

                return success;
            }
        }

        //this needs the if clause that finds the neighbour
        return true;

    }

    inline uint8_t number_neighbours_in_direction(const uint8_t& face){

        switch (level_delta){
            case _LEVEL_INCREASE:
                return 4;
            case _NO_NEIGHBOUR:
                return 0;
        }
        return 1;

    }

    inline unsigned int x_nearest_pixel(){
        //get x
        return floor((current_particle_cell.x+0.5)*pow(2, apr_access->level_max - current_particle_cell.level));
    }

    inline float x_global(){
        //get x
        return (current_particle_cell.x+0.5)*pow(2, apr_access->level_max - current_particle_cell.level);
    }

    inline unsigned int y_nearest_pixel(){
        //get x
        return floor((current_particle_cell.y+0.5)*pow(2, apr_access->level_max - current_particle_cell.level));
    }

    inline float y_global(){
        //get x
        return (current_particle_cell.y+0.5)*pow(2, apr_access->level_max - current_particle_cell.level);
    }

    inline unsigned int z_nearest_pixel(){
        //get z nearest pixel
        return floor((current_particle_cell.z+0.5)*pow(2, apr_access->level_max - current_particle_cell.level));
    }

    inline float z_global(){
        //get z global coordinate
        return (current_particle_cell.z+0.5)*pow(2, apr_access->level_max - current_particle_cell.level);
    }

    inline uint16_t level_min(){
        return apr_access->level_min;
    }

    inline uint16_t level_max(){
        return apr_access->level_max;
    }

    inline uint64_t spatial_index_x_max(const unsigned int level){
        return apr_access->x_num[level];
    }

    inline uint64_t spatial_index_y_max(const unsigned int level){
        return apr_access->y_num[level];
    }

    inline uint64_t spatial_index_z_max(const unsigned int level){
        return apr_access->z_num[level];
    }
    /////////////////////////
    /// Random access
    ///
    /////////////////////////


protected:
    //private methods

    inline bool check_neighbours_particle_cell_in_bounds(){
        //uses the fact that the coordinates have unsigned type, and therefore if they are negative they will be above the bound
        if(check_neigh_flag) {
            return (neighbour_particle_cell.x < apr_access->x_num[neighbour_particle_cell.level]) &
                   (neighbour_particle_cell.z < apr_access->z_num[neighbour_particle_cell.level]);
        } else {
            return true;
        }
    }


    inline void set_neighbour_flag(){
        check_neigh_flag = apr_access->check_neighbours_flag(current_particle_cell.x,current_particle_cell.z,current_particle_cell.level);
    }

};


#endif //LIBAPR_GENITERATOR_HPP
