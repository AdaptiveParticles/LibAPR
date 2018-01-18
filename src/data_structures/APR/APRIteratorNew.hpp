//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APR_ITERATOR_NEW_HPP
#define PARTPLAY_APR_ITERATOR_NEW_HPP

#include "src/data_structures/APR/APR.hpp"

#include "src/data_structures/APR/APRAccess.hpp"
template<typename ImageType>
class APRIteratorNew {

private:

    ParticleCell current_particle_cell;

    APRAccess* apr_access;

    uint16_t level_delta;

    MapIterator current_gap;

    const uint16_t shift[6] = {YP_LEVEL_SHIFT,YM_LEVEL_SHIFT,XP_LEVEL_SHIFT,XM_LEVEL_SHIFT,ZP_LEVEL_SHIFT,ZM_LEVEL_SHIFT};
    const uint16_t mask[6] = {YP_LEVEL_MASK,YM_LEVEL_MASK,XP_LEVEL_MASK,XM_LEVEL_MASK,ZP_LEVEL_MASK,ZM_LEVEL_MASK};

    bool move_iterator_to_next_non_empty_row(const uint64_t &maximum_level){

        uint64_t offset_max = apr_access->x_num[current_particle_cell.level]*apr_access->z_num[current_particle_cell.level];

        //iterate until you find the next row or hit the end of the level
        while((apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset].size()==0) & (current_particle_cell.pc_offset < offset_max)){
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
            current_gap.iterator= apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.begin();
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
                    current_particle_cell.global_index = -1;
                    return false;
                }
            }
        }
    }



public:

    APRIteratorNew(){
        //current_part = 0;
    }

    APRIteratorNew(APRAccess& apr_access_){
       apr_access = &apr_access_;
    }

    void initialize_from_apr(APR<ImageType>& apr){


    }

    uint64_t total_number_parts(){
        return (apr_access)->total_number_parts;
    }

    bool it_begin(){
        return set_iterator_to_particle_by_number(0);
    }

    bool it_forward(){
        return move_to_next_particle_cell();
    }

    bool it_end(){
        return (current_particle_cell.global_index != -1);
    }


    bool set_iterator_to_particle_by_number(const uint64_t &particle_number){
        //
        //  Moves the iterator to point to the particle number (global index of the particle)
        //

        if(particle_number==0){
            current_particle_cell.level = level_min();
            current_particle_cell.pc_offset=0;

            if(move_iterator_to_next_non_empty_row(level_max())){
                //found and set
                return true;
            } else{
                return false; //no particle cells, something is wrong
            }
        } else if (particle_number < apr_access->total_number_parts) {

            //iterating just move to next
            if(particle_number == (current_particle_cell.global_index+1)){
                return move_to_next_particle_cell();
            }

            current_particle_cell.level = level_min();

            //otherwise now we have to figure out where to look for the next particle cell;

            //first find the level
            while((particle_number > apr_access->global_index_by_level_end[current_particle_cell.level]) & (current_particle_cell.level <= level_max())){
                current_particle_cell.level++;
            }

            //then find the offset (zx row)
            current_particle_cell.pc_offset=0;

            while(particle_number > particles_offset_end(current_particle_cell.level,current_particle_cell.pc_offset)){
                current_particle_cell.pc_offset++;
                uint64_t temp = particles_offset_end(current_particle_cell.level,current_particle_cell.pc_offset);
                int stop = 1;
            }

            current_particle_cell.z = (current_particle_cell.pc_offset)/spatial_index_x_max(current_particle_cell.level);
            current_particle_cell.x = (current_particle_cell.pc_offset) - current_particle_cell.z*(spatial_index_x_max(current_particle_cell.level));

            current_gap.iterator = current_gap.iterator= apr_access->gap_map.data[current_particle_cell.level][current_particle_cell.pc_offset][0].map.begin();
            //then find the gap.
            while((particle_number > apr_access->global_index_end(current_gap))){
                current_gap.iterator++;
            }

            current_particle_cell.y = (current_gap.iterator->first) + (particle_number - current_gap.iterator->second.global_index_begin);
            current_particle_cell.global_index = particle_number;
            return true;

        } else {
            current_particle_cell.global_index = -1;
            return false; // requested particle number exceeds the number of particles
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


    bool set_iteartor_by_random_access(ParticleCell& input){

    }

//    bool set_iterator_to_particle_by_number(uint64_t part_num){
//        //
//        //  Moves the iterator to be at the set particle number (from depth_min to depth_max, iterating y, x, then z)
//        //
//
////        if(part_num == (current_part+1)){
////            curr_level.move_to_next_pc(*pc_data_pointer);
////            current_part++;
////        } else if(part_num != current_part) {
////            //
////            //  Find the part number
////            //
////
////            if(part_num < current_part){
////                current_part = 0;
////            }
////
////
////            if(current_part >= (num_parts_total)){
////                return false;
////            }
////
////            uint64_t depth = (*pc_data_pointer).depth_min;
////            //first depth search
////            while(((part_num) >= ((*num_parts)[depth])) | ((*num_parts)[depth] ==0) ){
////                depth++;
////            }
////
////            uint64_t offset_start = 0;
////
////            while(part_num >= (*num_parts_xz_pointer).data[depth][offset_start][0]){
////                offset_start++;
////            }
////
////            int total = (*num_parts_xz_pointer).data[depth][offset_start][0];
////
////            uint64_t z_ = (offset_start)/(*num_parts_xz_pointer).x_num[depth];
////            uint64_t x_ = (offset_start) - z_*(*num_parts_xz_pointer).x_num[depth];
////
////            //now it starts at begining and then must iterate forward
////            curr_level.set_new_depth(depth,*pc_data_pointer);
////            curr_level.set_new_z(z_,*pc_data_pointer);
////            curr_level.set_new_x(x_,*pc_data_pointer);
////            curr_level.update_j(*pc_data_pointer,0);
////
////            if(offset_start == 0){
////                current_part = (*num_parts)[depth-1];
////            } else {
////                current_part = (*num_parts_xz_pointer).data[depth][offset_start-1][0];
////            }
////
////            curr_level.move_to_next_pc(*pc_data_pointer);
////
////            while(current_part != part_num){
////
////                curr_level.move_to_next_pc(*pc_data_pointer);
////                current_part++;
////            }
////
////
////        }
////        else if(part_num ==0){
////            curr_level.set_new_depth((*pc_data_pointer).depth_min,*pc_data_pointer);
////            curr_level.set_new_z(0,*pc_data_pointer);
////            curr_level.set_new_x(0,*pc_data_pointer);
////            curr_level.update_j(*pc_data_pointer,0);
////
////            curr_level.move_to_next_pc(*pc_data_pointer);
////
////        }
//
//        return true;
//
//    }


    inline unsigned int x(){
        //get x
       return current_particle_cell.x;
    }

    inline unsigned int y(){
        //get x
        return current_particle_cell.y;
    }

    inline unsigned int z(){
        //get x
        return current_particle_cell.z;
    }



    inline unsigned int type(){
        //get x
        return current_particle_cell.type;
    }


    inline unsigned int level(){
        //get x
        return current_particle_cell.level;
    }

    bool set_neighbour_iterator(APRIteratorNew<ImageType> &original_iterator, const unsigned int dir, const unsigned int index){
        //
        //  This is sets the this iterator, to the neighbour of the particle cell that original_iterator is pointing to
        //

//        apr_access->get_neighbour_coordinate(original_iterator.current_particle_cell, current_particle_cell, dir, original_iterator.level_delta, index);
//
//        if(index > 0) {
//            //for children need to check boundary conditions
//            if (current_particle_cell.x < spatial_index_x_max(neigh.level)) {
//                if (current_particle_cell.z < spatial_index_z_max(neigh.level)) {
//                    return false;
//                }
//            }
//        }
//
//        return apr_access->find_particle_cell(current_particle_cell);

    }

    inline uint8_t number_neighbours_in_direction(unsigned int face){

        //level_delta =  (current_node & mask[face]) >> shift[face];

        switch (level_delta){
            case _LEVEL_INCREASE:
                return 4;
            case _NO_NEIGHBOUR:
                return 0;
        }
        return 1;

    }

    template<typename S>
    S& operator()(ExtraParticleData<S>& parts){
        //accesses the value of particle data when iterating
        return parts.data[current_particle_cell.global_index];
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

    inline unsigned int level_min(){
        return (*apr_access).level_min;
    }

    inline unsigned int level_max(){
        return (*apr_access).level_max;
    }

    inline unsigned int spatial_index_x_max(const unsigned int level){
        return (*apr_access).x_num[level];
    }

    inline unsigned int spatial_index_y_max(const unsigned int level){
        return (*apr_access).y_num[level];
    }

    inline unsigned int spatial_index_z_max(const unsigned int level){
        return (*apr_access).z_num[level];
    }


private:








};


#endif //PARTPLAY_APR_ITERATOR_NEW_HPP
