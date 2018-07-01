//
// Created by cheesema on 14.02.18.
//

#ifndef LIBAPR_APRTREEITERATOR_HPP
#define LIBAPR_APRTREEITERATOR_HPP

#include "APRIterator.hpp"
#include "APRTree.hpp"
#include "APR.hpp"

// APRIteration class, with extra methods designed for the use with APRTree

template<typename ImageType>
class APRTreeIterator : public GenIterator<ImageType> {
public:
    APR<ImageType>* aprOwn;

    APRTreeIterator(APR<ImageType>& apr){
        this->apr_access = &apr.apr_tree.tree_access;
        this->current_particle_cell.global_index = UINT64_MAX;
        this->highest_resolution_type = 8;
        aprOwn = &apr;
    }

    APRTreeIterator(APRTree<ImageType>& apr_tree){
        this->apr_access = &apr_tree.tree_access;
        this->current_particle_cell.global_index = UINT64_MAX;
        this->highest_resolution_type = 8;
    }

    void set_iterator_to_child(APRIterator<ImageType>& current_iterator,uint8_t child_index){
        //empty
    };

    bool set_iterator_to_parent(GenIterator<ImageType>& current_iterator){
        //takes an input iterator and sets it THIS iterator to the parent of the particle cell that the current_iterator is pointing to.
//        this->current_particle_cell.y = (uint16_t)floor(current_iterator.y()/2.0);
//        this->current_particle_cell.x = (uint16_t)floor(current_iterator.x()/2.0);
//        this->current_particle_cell.z = (uint16_t)floor(current_iterator.z()/2.0);
//        this->current_particle_cell.level = (uint16_t)(current_iterator.level() - 1);
//
//        if(this->current_particle_cell.level >= this->level_min()) {
//            return this->set_iterator_by_particle_cell(this->current_particle_cell);
//        } else {
//            //is at a level lower then the set minimum level.
//            return false;
//        }

    }

    void set_particle_cell_no_search(APRIterator<ImageType>& current_iterator){
        this->current_particle_cell.y = (uint16_t)(current_iterator.y());
        this->current_particle_cell.x = (uint16_t)(current_iterator.x());
        this->current_particle_cell.z = (uint16_t)(current_iterator.z());
        this->current_particle_cell.level = (uint16_t)(current_iterator.level());
        this->set_neighbour_flag();

    }

    uint64_t total_number_tree_particle_cells(){
        return this->apr_access->total_number_particles;
    }

    uint64_t set_new_lzxy(const uint16_t level,const uint16_t z,const uint16_t x,const uint16_t y){
        //#FIXME
//        this->current_particle_cell.level = level;
//        //otherwise now we have to figure out where to look for the next particle cell;
//
//        //back out your xz from the offset
//        this->current_particle_cell.z = z;
//        this->current_particle_cell.x = x;
//        this->current_particle_cell.y = y;
//
//        this->current_particle_cell.pc_offset =this->apr_access->x_num[level]*z + x;
//
//        this->end_index = this->particles_zx_end(level, z,
//                                                 x);
//
//        if(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size() > 0) {
//
//            ParticleCellGapMap& current_pc_map =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0];
//
//            //otherwise search for it (points to first key that is greater than the y value)
//            this->current_gap.iterator = current_pc_map.map.upper_bound(this->current_particle_cell.y);
//
//            bool end = false;
//
//            if(this->current_gap.iterator == current_pc_map.map.begin()){
//                //less then the first value
//
//                this->current_particle_cell.y = this->current_gap.iterator->first;
//                this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset; //#fixme
//
//                this->set_neighbour_flag();
//
//                return this->current_particle_cell.global_index;
//            } else{
//
//                if(this->current_gap.iterator == current_pc_map.map.end()){
//                    end = true;
//                }
//                this->current_gap.iterator--;
//            }
//
//            if ((this->current_particle_cell.y >= this->current_gap.iterator->first) & (this->current_particle_cell.y <= this->current_gap.iterator->second.y_end)) {
//                // exists
//                this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset +
//                                                           (this->current_particle_cell.y - this->current_gap.iterator->first); //#fixme
//                this->set_neighbour_flag();
//                return this->current_particle_cell.global_index;
//            }
//
//            if(end){
//                //no more particles
//                this->current_particle_cell.global_index = UINT64_MAX;
//                return this->current_particle_cell.global_index;
//            } else {
//                //still within range
//                this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset; //#fixme
//                this->current_particle_cell.y = this->current_gap.iterator->first;
//                this->set_neighbour_flag();
//                return this->current_particle_cell.global_index;
//            }
//
//
//        } else {
//            return UINT64_MAX;
//        }

    }

    bool set_iterator_to_particle_next_particle(){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if(this->current_particle_cell.level == this->level_max()) {
            if ((this->current_particle_cell.y + 1) <= (this->current_gap.iterator->second.y_end) / 2) {
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

    bool find_neighbours_same_level(const uint8_t& direction){
        //#FIXME
        bool found = false;

        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_SAME,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.same_level[direction])){
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

    uint64_t set_new_lzx(const uint16_t level,const uint16_t z,const uint16_t x){
        //
        //  The tree iterator uses the APR high resolution information for its highest level, as they are the same.
        //

        this->current_particle_cell.level = level;
        //otherwise now we have to figure out where to look for the next particle cell;

        //back out your xz from the offset
        this->current_particle_cell.z = z;
        this->current_particle_cell.x = x;

        if(level == this->level_max()){
            this->current_particle_cell.pc_offset = aprOwn->apr_access.x_num[level]*(z) + (x);

            if(aprOwn->apr_access.gap_map.data[this->current_particle_cell.level+1][this->current_particle_cell.pc_offset].size() > 0) {

                this->current_gap.iterator =aprOwn->apr_access.gap_map.data[level+1][this->current_particle_cell.pc_offset][0].map.begin();
                this->current_particle_cell.y = (uint16_t)((this->current_gap.iterator->first)/2);

                uint64_t begin = 0;

                if(this->current_particle_cell.pc_offset == 0){
                    if(level == this->level_min()){
                        begin = 0;
                    } else {
                        begin =this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level-1].back();
                    }
                } else {
                    begin = this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level][this->current_particle_cell.pc_offset-1];
                }

                this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin;

                this->set_neighbour_flag();

                // IN HERE PUT THE STARTING INDEX!
                //auto it =(aprOwn->apr_access.gap_map.data[level+1][this->current_particle_cell.pc_offset][0].map.rbegin());
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

                uint64_t begin = 0;

                if(this->current_particle_cell.pc_offset == 0){
                    if(level == this->level_min()){
                        begin = 0;
                    } else {
                        begin =this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level-1].back();
                    }
                } else {
                    begin =this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level][this->current_particle_cell.pc_offset-1];
                }

                this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin;

                this->set_neighbour_flag();

                // IN HERE PUT THE STARTING INDEX!
                //auto it =(this->apr_access->gap_map.data[level][this->current_particle_cell.pc_offset][0].map.rbegin());
                this->end_index = this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level][this->current_particle_cell.pc_offset];

                if(this->apr_access->global_index_by_level_and_zx_end[5][0]>2388){
                    int stop = 1;
                }



                return this->current_particle_cell.global_index;
            } else {
                return UINT64_MAX;
            }

        }


    }


};


#endif //LIBAPR_APRTREEITERATOR_HPP
