//
// Created by cheesema on 2019-05-24.
//

#ifndef LIBAPR_RANDOMITERATOR_HPP
#define LIBAPR_RANDOMITERATOR_HPP

#include "GenIterator.hpp"

class RandomIterator: public GenIterator {

protected:

    LocalMapIterators local_iterators;
    ParticleCell neighbour_particle_cell{ 0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };
    ParticleCell current_particle_cell{0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };

    uint16_t level_delta;
    MapIterator current_gap;
    bool check_neigh_flag = false;
    bool check_neighbours_particle_cell_in_bounds();

public:

    operator uint64_t() {return current_particle_cell.global_index;};

    uint16_t z() const { return current_particle_cell.z; }
    uint16_t level() const { return current_particle_cell.level; }
    uint16_t x() const { return current_particle_cell.x; }
    uint16_t y() const { return current_particle_cell.y; }

    ParticleCell get_neigh_particle_cell();
    ParticleCell get_current_particle_cell();
    MapIterator& get_current_gap();

    uint8_t number_neighbours_in_direction(const uint8_t& face);

    inline void set_neighbour_flag();

    MapIterator& get_local_iterator(LocalMapIterators& local_iterators,const uint16_t& level_delta,const uint16_t& face,const uint16_t& index) const {
        switch (level_delta) {
            case _LEVEL_SAME:
                return local_iterators.same_level[face];

            case _LEVEL_DECREASE:
                return local_iterators.parent_level[face];

            case _LEVEL_INCREASE:
                return local_iterators.child_level[face][index];
        }

        return local_iterators.same_level[0];
    }
    inline bool check_neighbours_flag(const uint16_t& x,const uint16_t& z,const uint16_t& level){
        // 0 1 2 .............. x_num-3 x_num-2 x_num-1 (x_num)
        //                              ......(x-1)..........
        //                                      ..(x)........
        return ((uint16_t)(x-1)>(this->genInfo->x_num[level]-3)) | ((uint16_t)(z-1)>(this->genInfo->z_num[level]-3));
    }


};

inline ParticleCell RandomIterator::get_current_particle_cell(){
    return current_particle_cell;
}

inline ParticleCell RandomIterator::get_neigh_particle_cell(){
    return neighbour_particle_cell;
}

inline MapIterator& RandomIterator::get_current_gap(){
    return current_gap;
}

inline uint8_t RandomIterator::number_neighbours_in_direction(const uint8_t& face){
    switch (level_delta){
        case _LEVEL_INCREASE:
            return 4;
        case _NO_NEIGHBOUR:
            return 0;
    }
    return 1;
}

inline bool RandomIterator::check_neighbours_particle_cell_in_bounds(){
    //uses the fact that the coordinates have unsigned type, and therefore if they are negative they will be above the bound
    if (check_neigh_flag) {
        return (neighbour_particle_cell.x < this->genInfo->x_num[neighbour_particle_cell.level]) &&
               (neighbour_particle_cell.z < this->genInfo->z_num[neighbour_particle_cell.level]);
    }
    return true;
}



inline void RandomIterator::set_neighbour_flag(){
    check_neigh_flag = check_neighbours_flag(current_particle_cell.x,current_particle_cell.z,current_particle_cell.level);
}




#endif //LIBAPR_RANDOMITERATOR_HPP
