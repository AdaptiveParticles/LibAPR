//
// Created by cheesema on 14.02.18.
//

#ifndef LIBAPR_APRTREEITERATOR_HPP
#define LIBAPR_APRTREEITERATOR_HPP


#include "GenIterator.hpp"

// APRIteration class, with extra methods designed for the use with APRTree

class APRTreeIterator : public GenIterator {
public:
    APRAccess* aprOwn_access;

    APRTreeIterator(APRAccess& apr_access_,APRAccess& tree_access){
        this->apr_access = &tree_access;
        aprOwn_access = &apr_access_;
    }

    uint64_t total_number_tree_particle_cells();
    uint64_t set_new_lzxy(const uint16_t level,const uint16_t z,const uint16_t x,const uint16_t y);

    bool set_iterator_to_particle_next_particle();

    bool find_neighbours_same_level(const uint8_t& direction);

    bool set_neighbour_iterator(APRTreeIterator &original_iterator, const uint8_t& direction, const uint8_t& index);

    bool find_next_child(const uint8_t& direction,const uint8_t& index);

    uint64_t start_index(const uint16_t level, const uint64_t offset);

    uint64_t set_new_lzx(const uint16_t level,const uint16_t z,const uint16_t x);

};


uint64_t APRTreeIterator::total_number_tree_particle_cells(){
    return this->apr_access->total_number_particles;
}

uint64_t APRTreeIterator::set_new_lzxy(const uint16_t level,const uint16_t z,const uint16_t x,const uint16_t y){
    //otherwise now we have to figure out where to look for the next particle cell;
    //set to the correct row
    uint64_t begin_index = set_new_lzx(level,z,x);

    this->current_particle_cell.y = y;

    //row is non-emtpy
    if(begin_index!=UINT64_MAX){

        if(level == this->level_max()) {
            ParticleCellGapMap& current_pc_map = aprOwn_access->gap_map.data[this->current_particle_cell.level+1][this->current_particle_cell.pc_offset][0];
            //Using the APR access data-structure, therefore requires y re-scaling

            //otherwise search for it (points to first key that is greater than the y value)
            this->current_gap.iterator = current_pc_map.map.upper_bound((uint16_t)2*this->current_particle_cell.y);

            bool end = false;

            if (this->current_gap.iterator == current_pc_map.map.begin()) {
                //less then the first value

                this->current_particle_cell.y = (uint16_t) (this->current_gap.iterator->first/2);
                this->current_particle_cell.global_index =
                        this->current_gap.iterator->second.global_index_begin_offset/2 + begin_index;

                this->set_neighbour_flag();

                return this->current_particle_cell.global_index;
            } else {

                if (this->current_gap.iterator == current_pc_map.map.end()) {
                    end = true;
                }
                this->current_gap.iterator--;
            }

            if (((2*this->current_particle_cell.y) >= this->current_gap.iterator->first) &
                ((2*this->current_particle_cell.y) <= this->current_gap.iterator->second.y_end)) {
                // exists
                this->current_particle_cell.global_index =
                        this->current_gap.iterator->second.global_index_begin_offset/2 +
                        (2*this->current_particle_cell.y - this->current_gap.iterator->first)/2 + begin_index;
                this->set_neighbour_flag();
                return this->current_particle_cell.global_index;
            }

            if (end) {
                //no more particles
                this->current_particle_cell.global_index = UINT64_MAX;
                return this->current_particle_cell.global_index;
            } else {
                //still within range
                this->current_particle_cell.global_index =
                        this->current_gap.iterator->second.global_index_begin_offset/2 + begin_index;
                this->current_particle_cell.y = (uint16_t) (this->current_gap.iterator->first/2);
                this->set_neighbour_flag();
                return this->current_particle_cell.global_index;
            }
        } else {
            ParticleCellGapMap& current_pc_map =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0];
            //otherwise search for it (points to first key that is greater than the y value)
            this->current_gap.iterator = current_pc_map.map.upper_bound(this->current_particle_cell.y);

            bool end = false;

            if (this->current_gap.iterator == current_pc_map.map.begin()) {
                //less then the first value

                this->current_particle_cell.y = this->current_gap.iterator->first;
                this->current_particle_cell.global_index =
                        this->current_gap.iterator->second.global_index_begin_offset + begin_index;

                this->set_neighbour_flag();

                return this->current_particle_cell.global_index;
            } else {

                if (this->current_gap.iterator == current_pc_map.map.end()) {
                    end = true;
                }
                this->current_gap.iterator--;
            }

            if ((this->current_particle_cell.y >= this->current_gap.iterator->first) &
                (this->current_particle_cell.y <= this->current_gap.iterator->second.y_end)) {
                // exists
                this->current_particle_cell.global_index =
                        this->current_gap.iterator->second.global_index_begin_offset +
                        (this->current_particle_cell.y - this->current_gap.iterator->first) + begin_index;
                this->set_neighbour_flag();
                return this->current_particle_cell.global_index;
            }

            if (end) {
                //no more particles
                this->current_particle_cell.global_index = UINT64_MAX;
                return this->current_particle_cell.global_index;
            } else {
                //still within range
                this->current_particle_cell.global_index =
                        this->current_gap.iterator->second.global_index_begin_offset + begin_index;
                this->current_particle_cell.y = this->current_gap.iterator->first;
                this->set_neighbour_flag();
                return this->current_particle_cell.global_index;
            }
        }

    } else {
        this->current_particle_cell.global_index = UINT64_MAX;
        return UINT64_MAX;
    }
}

bool APRTreeIterator::set_iterator_to_particle_next_particle(){
    //
    //  Moves the iterator to point to the particle number (global index of the particle)
    //

    if(this->current_particle_cell.level == this->level_max()) {
        if ((this->current_particle_cell.y + 1) <= (this->current_gap.iterator->second.y_end/2)) {
            //  Still in same y gap

            this->current_particle_cell.global_index++;
            this->current_particle_cell.y++;
            return true;

        } else {

            //not in the same gap
            this->current_gap.iterator++;//move the iterator forward.


            //I am in the next gap
            this->current_particle_cell.global_index++;
            this->current_particle_cell.y = (uint16_t) (this->current_gap.iterator->first /
                                                        2); // the key is the first y value for the gap
            return true;
        }
    } else {
        if ((this->current_particle_cell.y + 1) <= (this->current_gap.iterator->second.y_end) ) {
            //  Still in same y gap

            this->current_particle_cell.global_index++;
            this->current_particle_cell.y++;
            return true;

        } else {

            //not in the same gap
            this->current_gap.iterator++;//move the iterator forward.

            //I am in the next gap
            this->current_particle_cell.global_index++;
            this->current_particle_cell.y = (uint16_t) (this->current_gap.iterator->first); // the key is the first y value for the gap
            return true;
        }
    }

}

bool APRTreeIterator::find_neighbours_same_level(const uint8_t& direction){

    bool found = false;

    this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_SAME,0);

    if(this->check_neighbours_particle_cell_in_bounds()){
        if(this->apr_access->find_particle_cell_tree(*aprOwn_access,this->neighbour_particle_cell,this->local_iterators.same_level[direction])){
            //found the neighbour! :D

            this->level_delta = _LEVEL_SAME;
            return true;
        }
    };

    if(!found){
        this->level_delta=_NO_NEIGHBOUR;
    }

    return found;

}

bool APRTreeIterator::set_neighbour_iterator(APRTreeIterator &original_iterator, const uint8_t& direction, const uint8_t& index){
    //
    //  This is sets the this iterator, to the neighbour of the particle cell that original_iterator is pointing to
    //

    if(original_iterator.level_delta!=_LEVEL_INCREASE){
        //copy the information from the original iterator
        std::swap(this->current_particle_cell,original_iterator.neighbour_particle_cell);

    } else {
        if(index==0){
            std::swap(this->current_particle_cell,original_iterator.neighbour_particle_cell);

        } else {
            bool success = original_iterator.find_next_child(direction,index);
            std::swap(this->current_particle_cell,original_iterator.neighbour_particle_cell);

            return success;
        }
    }

    //this needs the if clause that finds the neighbour
    return true;

}

bool APRTreeIterator::find_next_child(const uint8_t& direction,const uint8_t& index){

    this->level_delta = _LEVEL_INCREASE;
    this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,this->level_delta,index);

    if(this->check_neighbours_particle_cell_in_bounds()){
        if(this->apr_access->find_particle_cell_tree(*aprOwn_access,this->neighbour_particle_cell,this->apr_access->get_local_iterator(this->local_iterators, this->level_delta, direction,index))){
            //found the neighbour! :D
            return true;
        }
    };
    return false;
}

uint64_t APRTreeIterator::start_index(const uint16_t level, const uint64_t offset){

    if(this->current_particle_cell.pc_offset == 0){
        if(level == this->level_min()){
            return  0;
        } else {
            return this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level-1].back();
        }
    } else {
        return this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level][this->current_particle_cell.pc_offset-1];
    }


}

uint64_t APRTreeIterator::set_new_lzx(const uint16_t level,const uint16_t z,const uint16_t x){
    //
    //  The tree iterator uses the APR high resolution information for its highest level, as they are the same.
    //

    this->current_particle_cell.level = level;
    //otherwise now we have to figure out where to look for the next particle cell;

    //back out your xz from the offset
    this->current_particle_cell.z = z;
    this->current_particle_cell.x = x;

    if(level == this->level_max()){
        this->current_particle_cell.pc_offset = aprOwn_access->x_num[level]*(z) + (x);

        //note this is different, using the APR's access datastructure
        if(aprOwn_access->gap_map.data[this->current_particle_cell.level+1][this->current_particle_cell.pc_offset].size() > 0) {

            this->current_gap.iterator =aprOwn_access->gap_map.data[level+1][this->current_particle_cell.pc_offset][0].map.begin();
            this->current_particle_cell.y = (uint16_t)((this->current_gap.iterator->first)/2);

            uint64_t begin = start_index(level,this->current_particle_cell.pc_offset);

            this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin;

            this->set_neighbour_flag();

            // IN HERE PUT THE STARTING INDEX!
            this->end_index = this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level][this->current_particle_cell.pc_offset];

            return this->current_particle_cell.global_index;
        } else {
            return UINT64_MAX;
        }

    } else {
        this->current_particle_cell.pc_offset =this->apr_access->x_num[level]*z + x;

        if(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size() > 0) {

            this->current_gap.iterator =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
            this->current_particle_cell.y = this->current_gap.iterator->first;

            uint64_t begin = start_index(level,this->current_particle_cell.pc_offset);
            this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin;

            this->set_neighbour_flag();

            // IN HERE PUT THE STARTING INDEX!
            this->end_index = this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level][this->current_particle_cell.pc_offset];

            return this->current_particle_cell.global_index;
        } else {
            return UINT64_MAX;
        }

    }


}


#include "GenIterator.hpp"


#endif //LIBAPR_APRTREEITERATOR_HPP
