//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APR_ITERATOR_NEW_HPP
#define PARTPLAY_APR_ITERATOR_NEW_HPP

//#include "APR.hpp"
#include "APRAccess.hpp"
//#include "APRTree.hpp"

template<typename ImageType>
class APRIterator {

protected:

    LocalMapIterators local_iterators;

    const uint8_t level_check_max[2] = {_LEVEL_SAME,_LEVEL_DECREASE};

    const uint8_t level_check_min[2] = {_LEVEL_SAME,_LEVEL_INCREASE};

    const uint8_t level_check_middle[3] = {_LEVEL_SAME,_LEVEL_DECREASE,_LEVEL_INCREASE};

    ParticleCell neighbour_particle_cell{ 0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };

    ParticleCell current_particle_cell{0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX };

    APRAccess* apr_access;

    uint16_t level_delta{};

    MapIterator current_gap;

    uint8_t highest_resolution_type;

    bool check_neigh_flag = false;

    const uint16_t shift[6] = {YP_LEVEL_SHIFT,YM_LEVEL_SHIFT,XP_LEVEL_SHIFT,XM_LEVEL_SHIFT,ZP_LEVEL_SHIFT,ZM_LEVEL_SHIFT};
    const uint16_t mask[6] = {YP_LEVEL_MASK,YM_LEVEL_MASK,XP_LEVEL_MASK,XM_LEVEL_MASK,ZP_LEVEL_MASK,ZM_LEVEL_MASK};



public:

    void move_gap(unsigned long& gap){
        current_gap.iterator++;
        gap++;
    }

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
        return current_gap.iterator->second.global_index_begin;
    }

//    explicit APRIterator(APR<ImageType>& apr){
//        apr_access = &apr.apr_access;
//        current_particle_cell.global_index = UINT64_MAX;
//        highest_resolution_type = 1;
//    }

    explicit APRIterator(APRAccess& apr_access_, uint8_t aHighestResolutionType = 1){
        apr_access = &apr_access_;
        current_particle_cell.global_index = UINT64_MAX;
        highest_resolution_type = aHighestResolutionType;
    }

    uint64_t total_number_particles(){
        return (apr_access)->total_number_particles;
    }

    bool set_iterator_to_particle_by_number(const uint64_t particle_number){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if(particle_number==0){
            current_particle_cell.level = level_min();
            current_particle_cell.pc_offset=0;

            if(move_iterator_to_next_non_empty_row(level_max())){
                //found and set
                set_neighbour_flag();
                return true;
            } else{
                return false; //no particle cells, something is wrong
            }
        } else if (particle_number < apr_access->total_number_particles) {

            //iterating just move to next
            if(particle_number == (current_particle_cell.global_index+1)){
                bool success = move_to_next_particle_cell();
                set_neighbour_flag();
                return success;
            }

            current_particle_cell.level = level_min();
            //otherwise now we have to figure out where to look for the next particle cell;

            //first find the level
            while((current_particle_cell.level <= level_max()) && (particle_number > apr_access->global_index_by_level_end[current_particle_cell.level])  ){
                current_particle_cell.level++;
            }

            //then find the offset (zx row)
            current_particle_cell.pc_offset=0;

            while(particle_number > particles_offset_end(current_particle_cell.level,current_particle_cell.pc_offset)){
                current_particle_cell.pc_offset++;
            }

            //back out your xz from the offset
            current_particle_cell.z = (current_particle_cell.pc_offset)/spatial_index_x_max(current_particle_cell.level);
            current_particle_cell.x = (current_particle_cell.pc_offset) - current_particle_cell.z*(spatial_index_x_max(current_particle_cell.level));

            current_gap.iterator = apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.begin();
            //then find the gap.
            while((particle_number > apr_access->global_index_end(current_gap))){
                current_gap.iterator++;
            }

            current_particle_cell.y = (current_gap.iterator->first) + (particle_number - current_gap.iterator->second.global_index_begin);
            current_particle_cell.global_index = particle_number;
            set_neighbour_flag();
            return true;

        } else {
            current_particle_cell.global_index = -1;
            return false; // requested particle number exceeds the number of particles
        }

    }

    bool set_iterator_to_particle_by_number(const uint64_t particle_number,const uint16_t level){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if(particle_number==0){
            current_particle_cell.level = level;
            current_particle_cell.pc_offset=0;

            if(move_iterator_to_next_non_empty_row(level_max())){
                //found and set
                set_neighbour_flag();
                return true;
            } else{
                return false; //no particle cells, something is wrong
            }
        } else if (particle_number < apr_access->total_number_particles) {

            //iterating just move to next
            if(particle_number == (current_particle_cell.global_index+1)){
                bool success = move_to_next_particle_cell();
                set_neighbour_flag();
                return success;
            }

            current_particle_cell.level = level;
            //otherwise now we have to figure out where to look for the next particle cell;

            //then find the offset (zx row)
            current_particle_cell.pc_offset=0;

            while(particle_number > particles_offset_end(current_particle_cell.level,current_particle_cell.pc_offset)){
                current_particle_cell.pc_offset++;
            }

            //back out your xz from the offset
            current_particle_cell.z = (current_particle_cell.pc_offset)/spatial_index_x_max(current_particle_cell.level);
            current_particle_cell.x = (current_particle_cell.pc_offset) - current_particle_cell.z*(spatial_index_x_max(current_particle_cell.level));

            current_gap.iterator = apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.begin();
            //then find the gap.
            while((particle_number > apr_access->global_index_end(current_gap))){
                current_gap.iterator++;
            }

            current_particle_cell.y = (current_gap.iterator->first) + (particle_number - current_gap.iterator->second.global_index_begin);
            current_particle_cell.global_index = particle_number;
            set_neighbour_flag();
            return true;

        } else {
            current_particle_cell.global_index = -1;
            return false; // requested particle number exceeds the number of particles
        }

    }



    uint64_t set_new_lzx(const uint16_t level,const uint16_t z,const uint16_t x){
        current_particle_cell.level = level;
        //otherwise now we have to figure out where to look for the next particle cell;

        //back out your xz from the offset
        current_particle_cell.z = z;
        current_particle_cell.x = x;

        current_particle_cell.pc_offset = apr_access->x_num[level]*z + x;

        if(apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset].size() > 0) {

            current_gap.iterator = apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.begin();
            current_particle_cell.y = current_gap.iterator->first;
            current_particle_cell.global_index = current_gap.iterator->second.global_index_begin;

            set_neighbour_flag();

            // IN HERE PUT THE STARTING INDEX!
            return current_particle_cell.global_index;
        } else {
            return UINT64_MAX;
        }

    }

    bool set_iterator_to_particle_next_particle(){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if( (current_particle_cell.y+1) <= current_gap.iterator->second.y_end){
            //  Still in same y gap

            current_particle_cell.global_index++;
            current_particle_cell.y++;
            return true;

        } else {

            //not in the same gap
            current_gap.iterator++;//move the iterator forward.


            //I am in the next gap
            current_particle_cell.global_index++;
            current_particle_cell.y = current_gap.iterator->first; // the key is the first y value for the gap
            return true;
        }

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
            return (it->second.global_index_begin + (it->second.y_end-it->first));
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

        if(current_particle_cell.level==level_max()){
            return highest_resolution_type; //all highest resolution pcs are seed, when using the nieghborhood optimization/
        } else {
            return apr_access->particle_cell_type.data[current_particle_cell.global_index];
        }

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



    bool find_neighbours_in_direction(const uint8_t& direction){

        //the three cases
        if(current_particle_cell.level == apr_access->l_max){
            //for (int l = 0; l < 2; ++l) {

            apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,_LEVEL_SAME,0);

            if(check_neighbours_particle_cell_in_bounds()){
                if(apr_access->find_particle_cell(neighbour_particle_cell,local_iterators.same_level[direction])){
                    //found the neighbour! :D
                    level_delta = _LEVEL_SAME;
                    return true;
                }
            };

            apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,_LEVEL_DECREASE,0);

            if(check_neighbours_particle_cell_in_bounds()){
                if(apr_access->find_particle_cell(neighbour_particle_cell,local_iterators.parent_level[direction])){
                    level_delta = _LEVEL_DECREASE;

                    return true;

                }
            };

            //}

        } else if(current_particle_cell.level == apr_access->l_min){
            //for (int l = 0; l < 2; ++l) {

            apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,_LEVEL_SAME,0);

            if(check_neighbours_particle_cell_in_bounds()){
                if(apr_access->find_particle_cell(neighbour_particle_cell,local_iterators.same_level[direction])){
                    //found the neighbour! :D
                    level_delta = _LEVEL_SAME;
                    return true;
                }
            };

            apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,_LEVEL_INCREASE,0);

            if(check_neighbours_particle_cell_in_bounds()){
                if(apr_access->find_particle_cell(neighbour_particle_cell,local_iterators.child_level[direction][0])){
                    level_delta = _LEVEL_INCREASE;
                    return true;
                }
            };

            //}
        } else {
            //for (int l = 0; l < 3; ++l) {
            apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,_LEVEL_SAME,0);

            if(check_neighbours_particle_cell_in_bounds()){
                if(apr_access->find_particle_cell(neighbour_particle_cell,local_iterators.same_level[direction])){
                    //found the neighbour! :D
                    level_delta = _LEVEL_SAME;
                    return true;
                }
            };

            apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,_LEVEL_DECREASE,0);

            if(check_neighbours_particle_cell_in_bounds()){
                if(apr_access->find_particle_cell(neighbour_particle_cell,local_iterators.parent_level[direction])){
                    level_delta = _LEVEL_DECREASE;
                    return true;
                }
            };
            apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,_LEVEL_INCREASE,0);

            if(check_neighbours_particle_cell_in_bounds()){
                if(apr_access->find_particle_cell(neighbour_particle_cell,local_iterators.child_level[direction][0])){
                    level_delta = _LEVEL_INCREASE;
                    return true;
                }
            };



        }

        level_delta=_NO_NEIGHBOUR;

        return false;

    }





    bool set_neighbour_iterator(APRIterator<ImageType> &original_iterator, const uint8_t& direction, const uint8_t& index){
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
        return floor((current_particle_cell.x+0.5)*pow(2, apr_access->l_max - current_particle_cell.level));
    }

    inline float x_global(){
        //get x
        return (current_particle_cell.x+0.5)*pow(2, apr_access->l_max - current_particle_cell.level);
    }

    inline unsigned int y_nearest_pixel(){
        //get x
        return floor((current_particle_cell.y+0.5)*pow(2, apr_access->l_max - current_particle_cell.level));
    }

    inline float y_global(){
        //get x
        return (current_particle_cell.y+0.5)*pow(2, apr_access->l_max - current_particle_cell.level);
    }

    inline unsigned int z_nearest_pixel(){
        //get z nearest pixel
        return floor((current_particle_cell.z+0.5)*pow(2, apr_access->l_max - current_particle_cell.level));
    }

    inline float z_global(){
        //get z global coordinate
        return (current_particle_cell.z+0.5)*pow(2, apr_access->l_max - current_particle_cell.level);
    }

    inline uint16_t level_min(){
        return apr_access->l_min;
    }

    inline uint16_t level_max(){
        return apr_access->l_max;
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

    bool set_iterator_by_particle_cell(ParticleCell& random_particle_cell){
        //
        //  Have to have set the particle cells x,y,z,level, and it will move the iterator to this location if it exists
        //

        random_particle_cell.pc_offset =  apr_access->x_num[random_particle_cell.level] * random_particle_cell.z + random_particle_cell.x;

        if(apr_access->find_particle_cell(random_particle_cell,current_gap)){
            current_particle_cell = random_particle_cell;
            set_neighbour_flag();
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
        if(((uint16_t)(x)>(apr_access->org_dims[1]-1)) | ((uint16_t)(z)>(apr_access->org_dims[2]-1)) | ((uint16_t)(y)>(apr_access->org_dims[0]-1))){
            //out of bounds
            return false;
        }

        //Then check from the highest level to lowest.
        ParticleCell particle_cell;
        particle_cell.y = round(y);
        particle_cell.x = round(x);
        particle_cell.z = round(z);
        particle_cell.level = level_max();

        particle_cell.pc_offset =  apr_access->x_num[particle_cell.level] * particle_cell.z + particle_cell.x;

        while( (particle_cell.level >= level_min()) && !(apr_access->find_particle_cell(particle_cell,current_gap)) ){
            particle_cell.y = particle_cell.y/2;
            particle_cell.x = particle_cell.x/2;
            particle_cell.z = particle_cell.z/2;
            particle_cell.level--;

            particle_cell.pc_offset =  apr_access->x_num[particle_cell.level] * particle_cell.z + particle_cell.x;
        }

        current_particle_cell = particle_cell; //if its in bounds it will always have a particle cell responsible
        set_neighbour_flag();
        return true;
    }

    bool find_neighbours_same_level(const uint8_t& direction){

        bool found = false;

        this->apr_access->get_neighbour_coordinate(this->current_particle_cell,this->neighbour_particle_cell,direction,_LEVEL_SAME,0);

        if(this->check_neighbours_particle_cell_in_bounds()){
            if(this->apr_access->find_particle_cell(this->neighbour_particle_cell,local_iterators.same_level[direction])){
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

        level_delta = _LEVEL_INCREASE;
        apr_access->get_neighbour_coordinate(current_particle_cell,neighbour_particle_cell,direction,level_delta,index);

        if(check_neighbours_particle_cell_in_bounds()){
            if(apr_access->find_particle_cell(neighbour_particle_cell,apr_access->get_local_iterator(local_iterators, level_delta, direction,index))){
                //found the neighbour! :D
                return true;
            }
        };
        return false;
    }

    inline bool check_neighbours_particle_cell_in_bounds(){
        //uses the fact that the coordinates have unsigned type, and therefore if they are negative they will be above the bound
        if(check_neigh_flag) {
            return (neighbour_particle_cell.x < apr_access->x_num[neighbour_particle_cell.level]) &
                   (neighbour_particle_cell.z < apr_access->z_num[neighbour_particle_cell.level]);
        } else {
            return true;
        }
    }

    bool move_iterator_to_next_non_empty_row(const uint64_t maximum_level){

        uint64_t offset_max = apr_access->x_num[current_particle_cell.level]*apr_access->z_num[current_particle_cell.level];

        //iterate until you find the next row or hit the end of the level
        while((current_particle_cell.pc_offset < offset_max) && (apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset].size()==0)){
            current_particle_cell.pc_offset++;
        }

        if(current_particle_cell.pc_offset == offset_max){
            //if within the level range, move to next level
            if(current_particle_cell.level < maximum_level){
                current_particle_cell.level++;
                current_particle_cell.pc_offset=0;
                return move_iterator_to_next_non_empty_row(maximum_level);
            } else {
                //reached last level
                return false;
            }
        } else {
            current_gap.iterator = apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.begin();
            current_particle_cell.global_index=current_gap.iterator->second.global_index_begin;
            current_particle_cell.y = current_gap.iterator->first;

            //compute x and z
            current_particle_cell.z = (current_particle_cell.pc_offset)/spatial_index_x_max(current_particle_cell.level);
            current_particle_cell.x = (current_particle_cell.pc_offset) - current_particle_cell.z*(spatial_index_x_max(current_particle_cell.level));

            return true;
        }

    }


    bool move_to_next_particle_cell(){
        //  Assumes all state variabels are valid for the current particle cell
        //
        //  moves particles cell in y direction if possible on same level
        //

        if( (current_particle_cell.y+1) <= current_gap.iterator->second.y_end){
            //  Still in same y gap

            current_particle_cell.global_index++;
            current_particle_cell.y++;
            return true;

        } else {

            //not in the same gap
            current_gap.iterator++;//move the iterator forward.

            if(current_gap.iterator!=(apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.end())){
                //I am in the next gap
                current_particle_cell.global_index++;
                current_particle_cell.y = current_gap.iterator->first; // the key is the first y value for the gap
                return true;
            } else {
                current_particle_cell.pc_offset++;
                //reached the end of the row
                if(move_iterator_to_next_non_empty_row(level_max())){
                    //found the next row set the iterator to the begining and find the particle cell.

                    return true;
                } else {
                    //reached the end of the particle cells
                    current_particle_cell.global_index = UINT64_MAX;
                    return false;
                }
            }
        }
    }

    inline void set_neighbour_flag(){
        check_neigh_flag = apr_access->check_neighbours_flag(current_particle_cell.x,current_particle_cell.z,current_particle_cell.level);
    }

};


#endif //PARTPLAY_APR_ITERATOR_NEW_HPP
