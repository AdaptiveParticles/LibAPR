//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APR_ITERATOR_NEW_HPP
#define PARTPLAY_APR_ITERATOR_NEW_HPP

#include "APR.hpp"
#include "APRAccess.hpp"
#include "APRTree.hpp"
#include "GenIterator.hpp"

template<typename ImageType>
class APRIterator  : public GenIterator<ImageType> {

protected:


    APR<ImageType>* aprOwn;


public:

    void move_gap(unsigned long& gap){
        this->current_gap.iterator++;
        gap++;
    }



    APRIterator(){
        //default constructor, for use by inherited classes
    }

    explicit APRIterator(APR<ImageType>& apr){
        aprOwn = &apr;
        this->apr_access = &apr.apr_access;
        this->current_particle_cell.global_index = UINT64_MAX;
        this->highest_resolution_type = 1;
    }

    explicit APRIterator(APRAccess& apr_access_){
        this->apr_access = &apr_access_;
        this->current_particle_cell.global_index = UINT64_MAX;
        this->highest_resolution_type = 1;
    }

    void initialize_from_apr(APR<ImageType>& apr){
        aprOwn = &apr;
        this->apr_access = &apr.apr_access;
        this->current_particle_cell.global_index = UINT64_MAX;
        this->highest_resolution_type = 1;
    }

    uint64_t total_number_particles(){
        return this->apr_access->total_number_particles;
    }

    bool set_iterator_to_particle_by_number(const uint64_t particle_number){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if(particle_number==0){
            this->current_particle_cell.level = this->level_min();
            this->current_particle_cell.pc_offset=0;

            if(move_iterator_to_next_non_empty_row(this->level_max())){
                //found and set
                this->set_neighbour_flag();
                return true;
            } else{
                return false; //no particle cells, something is wrong
            }
        } else if (particle_number <this->apr_access->total_number_particles) {

            //iterating just move to next
            if(particle_number == (this->current_particle_cell.global_index+1)){
                bool success = move_to_next_particle_cell();
                this->set_neighbour_flag();
                return success;
            }

            this->current_particle_cell.level = this->level_min();
            //otherwise now we have to figure out where to look for the next particle cell;

            //first find the level
            while((this->current_particle_cell.level <= this->level_max()) && (particle_number >this->apr_access->global_index_by_level_end[this->current_particle_cell.level])  ){
                this->current_particle_cell.level++;
            }

            //then find the offset (zx row)
            this->current_particle_cell.pc_offset=0;

            while(particle_number > this->particles_offset_end(this->current_particle_cell.level,this->current_particle_cell.pc_offset)){
                this->current_particle_cell.pc_offset++;
            }

            //back out your xz from the offset
            this->current_particle_cell.z = (this->current_particle_cell.pc_offset)/this->spatial_index_x_max(this->current_particle_cell.level);
            this->current_particle_cell.x = (this->current_particle_cell.pc_offset) - this->current_particle_cell.z*(this->spatial_index_x_max(this->current_particle_cell.level));

            this->current_gap.iterator =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
            //then find the gap.
            while((particle_number >this->apr_access->global_index_end(this->current_gap))){
                this->current_gap.iterator++;
            }

            this->current_particle_cell.y = (this->current_gap.iterator->first) + (particle_number - this->current_gap.iterator->second.global_index_begin_offset); //#fixme
            this->current_particle_cell.global_index = particle_number;
            this->set_neighbour_flag();
            return true;

        } else {
            this->current_particle_cell.global_index = -1;
            return false; // requested particle number exceeds the number of particles
        }

    }

    bool set_iterator_to_particle_by_number(const uint64_t particle_number,const uint16_t level){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if(particle_number==0){
            this->current_particle_cell.level = level;
            this->current_particle_cell.pc_offset=0;

            if(move_iterator_to_next_non_empty_row(this->level_max())){
                //found and set
                this->set_neighbour_flag();
                return true;
            } else{
                return false; //no particle cells, something is wrong
            }
        } else if (particle_number <this->apr_access->total_number_particles) {

            //iterating just move to next
            if(particle_number == (this->current_particle_cell.global_index+1)){
                bool success = move_to_next_particle_cell();
                this->set_neighbour_flag();
                return success;
            }

            this->current_particle_cell.level = level;
            //otherwise now we have to figure out where to look for the next particle cell;

            //then find the offset (zx row)
            this->current_particle_cell.pc_offset=0;

            while(particle_number > particles_offset_end(this->current_particle_cell.level,this->current_particle_cell.pc_offset)){
                this->current_particle_cell.pc_offset++;
            }

            //back out your xz from the offset
            this->current_particle_cell.z = (this->current_particle_cell.pc_offset)/spatial_index_x_max(this->current_particle_cell.level);
            this->current_particle_cell.x = (this->current_particle_cell.pc_offset) - this->current_particle_cell.z*(spatial_index_x_max(this->current_particle_cell.level));

            this->current_gap.iterator =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
            //then find the gap.
            while((particle_number >this->apr_access->global_index_end(this->current_gap))){
                this->current_gap.iterator++;
            }

            this->current_particle_cell.y = (this->current_gap.iterator->first) + (particle_number - this->current_gap.iterator->second.global_index_begin_offset); //#fixme
            this->current_particle_cell.global_index = particle_number;
            this->set_neighbour_flag();
            return true;

        } else {
            this->current_particle_cell.global_index = -1;
            return false; // requested particle number exceeds the number of particles
        }

    }

    uint64_t start_index(const uint16_t level, const uint64_t offset){

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

    uint64_t max_row_level_offset(const uint16_t x,const uint16_t z,const uint16_t num_parts){
        return ((x%2) + (z%2)*2)*((uint64_t)num_parts) ;//calculates the number of particles in the row
    }

    uint64_t set_new_lzx(const uint16_t level,const uint16_t z,const uint16_t x){
        this->current_particle_cell.level = level;
        //otherwise now we have to figure out where to look for the next particle cell;

        //back out your xz from the offset
        this->current_particle_cell.z = z;
        this->current_particle_cell.x = x;

        if(level == this->level_max()){
            this->current_particle_cell.pc_offset =this->apr_access->x_num[level-1]*(z/2) + (x/2);

            if(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size() > 0) {

                this->current_gap.iterator =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
                this->current_particle_cell.y = this->current_gap.iterator->first;

                uint64_t begin = start_index(level,this->current_particle_cell.pc_offset);

                this->current_particle_cell.global_index = begin;

                this->set_neighbour_flag();

                //requries now an offset depending on the child position odd/even
                auto it =(this->apr_access->gap_map.data[level][this->current_particle_cell.pc_offset][0].map.rbegin());
                uint16_t num_parts = ((it->second.global_index_begin_offset + (it->second.y_end-it->first))+1);

                this->end_index =  begin + num_parts;

                //calculates the offset for the xz position
                uint64_t index_offset = max_row_level_offset(x, z, num_parts);

                this->end_index += index_offset;
                this->current_particle_cell.global_index += index_offset;

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
                auto it =(this->apr_access->gap_map.data[level][this->current_particle_cell.pc_offset][0].map.rbegin());
                this->end_index = this->apr_access->global_index_by_level_and_zx_end[this->current_particle_cell.level][this->current_particle_cell.pc_offset];

                return this->current_particle_cell.global_index;
            } else {
                return UINT64_MAX;
            }

        }


    }

    uint64_t set_new_lzxy(const uint16_t level,const uint16_t z,const uint16_t x,const uint16_t y){

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

    bool set_iterator_to_particle_next_particle(){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if( (this->current_particle_cell.y+1) <= this->current_gap.iterator->second.y_end){
            //  Still in same y gap

            this->current_particle_cell.global_index++;
            this->current_particle_cell.y++;
            return true;

        } else {

            //not in the same gap
            this->current_gap.iterator++;//move the iterator forward.


            //I am in the next gap
            this->current_particle_cell.global_index++;
            this->current_particle_cell.y = this->current_gap.iterator->first; // the key is the first y value for the gap
            return true;
        }

    }


    bool find_neighbours_in_direction(const uint8_t& direction){

        //the three cases
        if(this->current_particle_cell.level ==this->apr_access->level_max){
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

        } else if(this->current_particle_cell.level ==this->apr_access->level_min){
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





    bool set_neighbour_iterator(APRIterator<ImageType> &original_iterator, const uint8_t& direction, const uint8_t& index){
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



    /////////////////////////
    /// Random access
    ///
    /////////////////////////

    bool set_iterator_by_particle_cell(ParticleCell& random_particle_cell){
        //
        //  Have to have set the particle cells x,y,z,level, and it will move the iterator to this location if it exists
        //

        random_particle_cell.pc_offset = this->apr_access->x_num[random_particle_cell.level] * random_particle_cell.z + random_particle_cell.x;

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

    bool set_iterator_by_global_coordinate(float x,float y,float z){
        //
        //  Finds the Particle Cell for which the point (x,y,z) belongs to its spatial domain and set the iterator to it
        //

        //check in bounds
        if(((uint16_t)(x)>(this->apr_access->org_dims[1]-1)) | ((uint16_t)(z)>(this->apr_access->org_dims[2]-1)) | ((uint16_t)(y)>(this->apr_access->org_dims[0]-1))){
            //out of bounds
            return false;
        }

        //Then check from the highest level to lowest.
        ParticleCell particle_cell;
        particle_cell.y = round(y);
        particle_cell.x = round(x);
        particle_cell.z = round(z);
        particle_cell.level = this->level_max();

        particle_cell.pc_offset = this->apr_access->x_num[particle_cell.level] * particle_cell.z + particle_cell.x;

        while( (particle_cell.level >= this->level_min()) && !(this->apr_access->find_particle_cell(particle_cell,this->current_gap)) ){
            particle_cell.y = particle_cell.y/2;
            particle_cell.x = particle_cell.x/2;
            particle_cell.z = particle_cell.z/2;
            particle_cell.level--;

            particle_cell.pc_offset = this->apr_access->x_num[particle_cell.level] * particle_cell.z + particle_cell.x;
        }

        this->current_particle_cell = particle_cell; //if its in bounds it will always have a particle cell responsible
        this->set_neighbour_flag();
        return true;
    }

    bool find_neighbours_same_level(const uint8_t& direction){

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


protected:
    //private methods

    bool find_next_child(const uint8_t& direction,const uint8_t& index){

        this->level_delta = _LEVEL_INCREASE;
       this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,this->level_delta,index);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,this->apr_access->get_local_iterator(this->local_iterators, this->level_delta, direction,index))){
                //found the neighbour! :D
                return true;
            }
        };
        return false;
    }


    bool move_iterator_to_next_non_empty_row(const uint64_t maximum_level){

        uint64_t offset_max =this->apr_access->x_num[this->current_particle_cell.level]*this->apr_access->z_num[this->current_particle_cell.level];

        //iterate until you find the next row or hit the end of the level
        while((this->current_particle_cell.pc_offset < offset_max) &&(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset].size()==0)){
            this->current_particle_cell.pc_offset++;
        }

        if(this->current_particle_cell.pc_offset == offset_max){
            //if within the level range, move to next level
            if(this->current_particle_cell.level < maximum_level){
                this->current_particle_cell.level++;
                this->current_particle_cell.pc_offset=0;
                return move_iterator_to_next_non_empty_row(maximum_level);
            } else {
                //reached last level
                return false;
            }
        } else {
            this->current_gap.iterator =this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.begin();
            this->current_particle_cell.global_index=this->current_gap.iterator->second.global_index_begin_offset; //#fixme
            this->current_particle_cell.y = this->current_gap.iterator->first;

            //compute x and z
            this->current_particle_cell.z = (this->current_particle_cell.pc_offset)/this->spatial_index_x_max(this->current_particle_cell.level);
            this->current_particle_cell.x = (this->current_particle_cell.pc_offset) - this->current_particle_cell.z*(this->spatial_index_x_max(this->current_particle_cell.level));

            return true;
        }

    }


    bool move_to_next_particle_cell(){
        //  Assumes all state variabels are valid for the current particle cell
        //
        //  moves particles cell in y direction if possible on same level
        //

        if( (this->current_particle_cell.y+1) <= this->current_gap.iterator->second.y_end){
            //  Still in same y gap

            this->current_particle_cell.global_index++;
            this->current_particle_cell.y++;
            return true;

        } else {

            //not in the same gap
            this->current_gap.iterator++;//move the iterator forward.

            if(this->current_gap.iterator!=(this->apr_access->gap_map.data[this->current_particle_cell.level][this->current_particle_cell.pc_offset][0].map.end())){
                //I am in the next gap
                this->current_particle_cell.global_index++;
                this->current_particle_cell.y = this->current_gap.iterator->first; // the key is the first y value for the gap
                return true;
            } else {
                this->current_particle_cell.pc_offset++;
                //reached the end of the row
                if(move_iterator_to_next_non_empty_row(this->level_max())){
                    //found the next row set the iterator to the begining and find the particle cell.

                    return true;
                } else {
                    //reached the end of the particle cells
                    this->current_particle_cell.global_index = UINT64_MAX;
                    return false;
                }
            }
        }
    }

};


#endif //PARTPLAY_APR_ITERATOR_NEW_HPP
