//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APR_ITERATOR_NEW_HPP
#define PARTPLAY_APR_ITERATOR_NEW_HPP

#include "APRAccessStructures.hpp"

#include "RandomIterator.hpp"

class APRIterator  : public RandomIterator {

public:

    RandomAccess* apr_access;

    explicit APRIterator(RandomAccess& apr_access_,GenInfo& genInfo_) {
        apr_access = &apr_access_;
        this->genInfo =  &genInfo_;
    }

    /////////////////////////
    /// Random access
    ///
    /////////////////////////

    bool set_iterator_by_particle_cell(ParticleCell& random_particle_cell);
    bool set_iterator_by_global_coordinate(float x,float y,float z);
    bool find_neighbours_same_level(const uint8_t& direction);

    bool set_iterator_to_particle_next_particle();


    uint64_t set_new_lzx(const uint16_t level,const uint16_t z,const uint16_t x);

    uint64_t set_new_lzxy(const uint16_t level,const uint16_t z,const uint16_t x,const uint16_t y);

    bool find_neighbours_in_direction(const uint8_t& direction);

    inline bool set_neighbour_iterator(APRIterator &original_iterator, const uint8_t& direction, const uint8_t& index);

    inline PCDKey& get_pcd_key(){
        return particleCellDataKey;
    }

    inline void operator++ (int) override {
        set_iterator_to_particle_next_particle();
    }

    inline void operator++ () override {
        set_iterator_to_particle_next_particle();
    }

    inline uint64_t end() override {
        return end_index;
    }

    uint64_t begin(const uint16_t level,const uint16_t z,const uint16_t x) override {
        return set_new_lzx(level,z,x);
    }

    inline uint64_t particles_level_begin(const uint16_t& level_) override {return apr_access->global_index_by_level_and_zx_end_new[level_-1].back()+1;}
    inline uint64_t particles_level_end(const uint16_t& level_) override {return apr_access->global_index_by_level_and_zx_end_new[level_].back();}

    uint64_t set_new_lzx_old(const uint16_t level,const uint16_t z,const uint16_t x); //for backward compatability do not use!
    // Todo make various begin functions. blank(), with level, with x,z, with level,

protected:
    bool find_next_child(const uint8_t& direction,const uint8_t& index);

    uint64_t start_index(const uint16_t level, const uint64_t offset);

    uint64_t max_row_level_offset(const uint16_t x,const uint16_t z,const uint16_t num_parts);

    PCDKey particleCellDataKey;


};


uint64_t APRIterator::start_index(const uint16_t level, const uint64_t offset){

    if(offset == 0){
        if(level == this->level_min()){
            return  0;
        } else {
            return this->apr_access->global_index_by_level_and_zx_end_new[level-1].back();
        }
    } else {
        return this->apr_access->global_index_by_level_and_zx_end_new[level][offset-1];
    }

}

uint64_t APRIterator::max_row_level_offset(const uint16_t x,const uint16_t z,const uint16_t num_parts){
    return ((x%2) + (z%2)*2)*((uint64_t)num_parts) ;//calculates the number of particles in the row
}

inline uint64_t APRIterator::set_new_lzx(const uint16_t level,const uint16_t z,const uint16_t x){

        this->current_particle_cell.level = level;
        //otherwise now we have to figure out where to look for the next particle cell;

        //back out your xz from the offset
        this->current_particle_cell.z = z;
        this->current_particle_cell.x = x;

        particleCellDataKey.offset = genInfo->x_num[level]*(z) + (x);
        particleCellDataKey.local_ind = 0;
        particleCellDataKey.level = level;

        this->current_particle_cell.type = genInfo->x_num[level]*(z) + (x);

        if(level == this->level_max()){
            this->current_particle_cell.pc_offset = genInfo->x_num[level-1]*(z/2) + (x/2);

            if(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size() > 0) {

                this->current_gap.iterator =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
                this->current_particle_cell.y = this->current_gap.iterator->first;

                uint64_t begin = start_index(level,particleCellDataKey.offset);

                this->current_particle_cell.global_index = begin;

                this->end_index = this->apr_access->global_index_by_level_and_zx_end_new[level][particleCellDataKey.offset];

                this->set_neighbour_flag();

                return this->current_particle_cell.global_index;
            } else {
                this->end_index = 0;
                this->current_particle_cell.y = UINT16_MAX;

                return UINT64_MAX;
            }

        } else {
            this->current_particle_cell.pc_offset = genInfo->x_num[level]*z + x;

            if(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size() > 0) {

                this->current_gap.iterator = this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
                this->current_particle_cell.y = this->current_gap.iterator->first;

                uint64_t begin = start_index(level,this->current_particle_cell.pc_offset);

                this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin;

                this->set_neighbour_flag();

                if(level<3){
                    check_neigh_flag = true;
                }

                // IN HERE PUT THE STARTING INDEX!
                //auto it =(this->apr_access->gap_map.data[level][this->current_particle_cell.pc_offset][0].map.rbegin());
                this->end_index = this->apr_access->global_index_by_level_and_zx_end_new[this->current_particle_cell.level][this->current_particle_cell.pc_offset];

                return this->current_particle_cell.global_index;
            } else {
                this->end_index = 0;
                this->current_particle_cell.y = UINT16_MAX;
                
		        return UINT64_MAX;
            }

        }
}



uint64_t APRIterator::set_new_lzxy(const uint16_t level,const uint16_t z,const uint16_t x,const uint16_t y){

    //otherwise now we have to figure out where to look for the next particle cell;
    //set to the correct row
    uint64_t begin_index = set_new_lzx(level,z,x);

    this->current_particle_cell.y = y;

    if(begin_index!=UINT64_MAX){
        ParticleCellGapMap& current_pc_map =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0];

        //otherwise search for it (points to first key that is greater than the y value)
        this->current_gap.iterator = current_pc_map.map.upper_bound(this->current_particle_cell.y);

        bool end = false;

        if(this->current_gap.iterator == current_pc_map.map.begin()){
            //less then the first value

            this->current_particle_cell.y = this->current_gap.iterator->first;
            this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin_index;

            this->set_neighbour_flag();

            return this->current_particle_cell.global_index;
        } else{

            if(this->current_gap.iterator == current_pc_map.map.end()){
                end = true;
            }
            this->current_gap.iterator--;
        }

        if ((this->current_particle_cell.y >= this->current_gap.iterator->first) & (this->current_particle_cell.y <= this->current_gap.iterator->second.y_end)) {
            // exists
            this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset +
                                                       (this->current_particle_cell.y - this->current_gap.iterator->first) + begin_index;
            this->set_neighbour_flag();
            return this->current_particle_cell.global_index;
        }

        if(end){
            //no more particles
            this->current_particle_cell.global_index = UINT64_MAX;
            return this->current_particle_cell.global_index;
        } else {
            //still within range
            this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin_index;
            this->current_particle_cell.y = this->current_gap.iterator->first;
            this->set_neighbour_flag();
            return this->current_particle_cell.global_index;
        }

    } else {
        return UINT64_MAX;
    }
}




inline bool APRIterator::set_iterator_to_particle_next_particle(){
    //
    //  Moves the iterator to point to the particle number (global index of the particle)
    //

    if( (this->current_particle_cell.y+1) <= this->current_gap.iterator->second.y_end){
        //  Still in same y gap

        particleCellDataKey.local_ind++;

        this->current_particle_cell.global_index++;
        this->current_particle_cell.y++;
        return true;

    } else {

        this->current_particle_cell.global_index++;

        if(this->current_particle_cell.global_index >= this->end_index){
            return false;
        }

        particleCellDataKey.local_ind++;

        //not in the same gap
        this->current_gap.iterator++;//move the iterator forward.

        //I am in the next gap

        this->current_particle_cell.y = this->current_gap.iterator->first; // the key is the first y value for the gap
        return true;
    }
}


//
//inline ParticleCell APRIterator::get_neigh_particle_cell(){
//    return this->neighbour_particle_cell;
//}

inline bool APRIterator::find_neighbours_in_direction(const uint8_t& direction){

    //the three cases
    if(this->current_particle_cell.level ==this->apr_access->level_max()){
        //for (int l = 0; l < 2; ++l) {

        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_SAME,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.same_level[direction])){
                //found the neighbour! :D
                this->level_delta = _LEVEL_SAME;
                return true;
            }
        };

        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_DECREASE,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.parent_level[direction])){
                this->level_delta = _LEVEL_DECREASE;

                return true;

            }
        };

        //}

    } else if(this->current_particle_cell.level ==this->apr_access->level_min()){
        //for (int l = 0; l < 2; ++l) {

        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_SAME,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.same_level[direction])){
                //found the neighbour! :D
                this->level_delta = _LEVEL_SAME;
                return true;
            }
        };

        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_INCREASE,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.child_level[direction][0])){
                this->level_delta = _LEVEL_INCREASE;
                return true;
            }
        };

        //}
    } else {
        //for (int l = 0; l < 3; ++l) {
        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_SAME,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.same_level[direction])){
                //found the neighbour! :D
                this->level_delta = _LEVEL_SAME;
                return true;
            }
        };

        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_DECREASE,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.parent_level[direction])){
                this->level_delta = _LEVEL_DECREASE;
                return true;
            }
        };
        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_INCREASE,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->local_iterators.child_level[direction][0])){
                this->level_delta = _LEVEL_INCREASE;
                return true;
            }
        };
    }

    this->level_delta=_NO_NEIGHBOUR;

    return false;

    }

inline bool APRIterator::set_neighbour_iterator(APRIterator &original_iterator, const uint8_t& direction, const uint8_t& index){
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

inline bool APRIterator::set_iterator_by_particle_cell(ParticleCell& random_particle_cell){
    //
    //  Have to have set the particle cells x,y,z,level, and it will move the iterator to this location if it exists
    //
    if(random_particle_cell.level==this->level_max()) {
        random_particle_cell.pc_offset =
                this->apr_access->gap_map.x_num[random_particle_cell.level-1] * (random_particle_cell.z/2) +
                        (random_particle_cell.x/2);
    } else {
        random_particle_cell.pc_offset =
                this->apr_access->gap_map.x_num[random_particle_cell.level] * random_particle_cell.z +
                random_particle_cell.x;
    }

    if(this->apr_access->find_particle_cell(random_particle_cell,this->current_gap)){
        this->current_particle_cell = random_particle_cell;
        this->set_neighbour_flag();
        //exists
        return true;
    } else {
        //particle cell doesn't exist
        return false;
    }
}

inline bool APRIterator::set_iterator_by_global_coordinate(float x,float y,float z){
    //
    //  Finds the Particle Cell for which the point (x,y,z) belongs to its spatial domain and set the iterator to it
    //

    //check in bounds
    if(((uint16_t)(x)>(genInfo->org_dims[1]-1)) | ((uint16_t)(z)>(genInfo->org_dims[2]-1)) | ((uint16_t)(y)>(genInfo->org_dims[0]-1))){
        //out of bounds
        return false;
    }

    //Then check from the highest level to lowest.
    ParticleCell particle_cell;
    particle_cell.y = floor(y);
    particle_cell.x = floor(x);
    particle_cell.z = floor(z);
    particle_cell.level = this->level_max();

    particle_cell.pc_offset = this->apr_access->gap_map.x_num[particle_cell.level] * (particle_cell.z/2) + (particle_cell.x/2);

    while( (particle_cell.level > this->level_min()) & !(this->apr_access->find_particle_cell(particle_cell,this->current_gap)) ){
        particle_cell.y = particle_cell.y/2;
        particle_cell.x = particle_cell.x/2;
        particle_cell.z = particle_cell.z/2;
        particle_cell.level--;

        particle_cell.pc_offset = this->apr_access->gap_map.x_num[particle_cell.level] * particle_cell.z + particle_cell.x;
    }

    this->current_particle_cell = particle_cell; //if its in bounds it will always have a particle cell responsible

    this->set_neighbour_flag();
    return true;
}

inline bool APRIterator::find_neighbours_same_level(const uint8_t& direction){

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

inline bool APRIterator::find_next_child(const uint8_t& direction,const uint8_t& index){

    this->level_delta = _LEVEL_INCREASE;
    this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,this->level_delta,index);

    if(this->check_neighbours_particle_cell_in_bounds()){
        if(this->apr_access->find_particle_cell(this->neighbour_particle_cell, get_local_iterator(this->local_iterators, this->level_delta, direction,index))){
            //found the neighbour! :D
            return true;
        }
    };
    return false;
}

inline uint64_t APRIterator::set_new_lzx_old(const uint16_t level,const uint16_t z,const uint16_t x){

    this->current_particle_cell.level = level;
    //otherwise now we have to figure out where to look for the next particle cell;

    //back out your xz from the offset
    this->current_particle_cell.z = z;
    this->current_particle_cell.x = x;

    particleCellDataKey.offset = genInfo->x_num[level]*(z) + (x);
    particleCellDataKey.local_ind = 0;
    particleCellDataKey.level = level;

    if(level == this->level_max()){
        this->current_particle_cell.pc_offset = genInfo->x_num[level-1]*(z/2) + (x/2);

        if(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size() > 0) {

            this->current_gap.iterator =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
            this->current_particle_cell.y = this->current_gap.iterator->first;

            uint64_t begin = 0;

            if(this->current_particle_cell.pc_offset == 0){
                if(level == this->level_min()){
                    begin =   0;
                } else {
                    begin =  this->apr_access->global_index_by_level_and_zx_end[level-1].back();
                }
            } else {
                begin = this->apr_access->global_index_by_level_and_zx_end[level][this->current_particle_cell.pc_offset-1];
            }

            this->current_particle_cell.global_index = begin;

            this->set_neighbour_flag();

            //requries now an offset depending on the child position odd/even
            auto it =(this->apr_access->gap_map.data[level][this->current_particle_cell.pc_offset][0].map.rbegin());
            uint16_t num_parts = ((it->second.global_index_begin_offset + (it->second.y_end-it->first))+1);

            this->end_index =  begin + num_parts;

            //calculates the offset for the xz position
            uint64_t index_offset=0;

            if(check_neigh_flag){

                uint64_t x_factor =2;

                if((x==(x_num(level)-1)) && ((x%2)==0)){
                    x_factor = 1;
                }

                index_offset = ((x%2) + (z%2)*x_factor)*((uint64_t)num_parts);
            } else {

                //calculates the offset for the xz position
                index_offset = max_row_level_offset(x, z, num_parts);
            }

            this->end_index += index_offset;
            this->current_particle_cell.global_index += index_offset;

            return this->current_particle_cell.global_index;
        } else {
            this->end_index = 0;
            this->current_particle_cell.y = UINT16_MAX;

            return UINT64_MAX;
        }

    } else {
        this->current_particle_cell.pc_offset = genInfo->x_num[level]*z + x;

        if(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size() > 0) {

            this->current_gap.iterator = this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
            this->current_particle_cell.y = this->current_gap.iterator->first;

            uint64_t begin = start_index(level,this->current_particle_cell.pc_offset);

            this->current_particle_cell.global_index = this->current_gap.iterator->second.global_index_begin_offset + begin;

            this->set_neighbour_flag();

            if(level<3){
                check_neigh_flag = true;
            }

            // IN HERE PUT THE STARTING INDEX!
            //auto it =(this->apr_access->gap_map.data[level][this->current_particle_cell.pc_offset][0].map.rbegin());
            this->end_index = this->apr_access->global_index_by_level_and_zx_end_new[this->current_particle_cell.level][this->current_particle_cell.pc_offset];

            return this->current_particle_cell.global_index;
        } else {
            this->end_index = 0;
            this->current_particle_cell.y = UINT16_MAX;

            return UINT64_MAX;
        }

    }
}




#endif //PARTPLAY_APR_ITERATOR_NEW_HPP
