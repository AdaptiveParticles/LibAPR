//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APR_ITERATOR_NEW_HPP
#define PARTPLAY_APR_ITERATOR_NEW_HPP

#include "src/data_structures/APR/APR.hpp"

#include "src/data_structures/APR/APRAccess.hpp"

class APRIteratorNew {

private:

    ParticleCell current_particle_cell;

    APRAccess* apr_access;

    uint16_t current_node;

    uint16_t level_delta;

    constexpr uint16_t shift[6] = {YP_LEVEL_SHIFT,YM_LEVEL_SHIFT,XP_LEVEL_SHIFT,XM_LEVEL_SHIFT,ZP_LEVEL_SHIFT,ZM_LEVEL_SHIFT};
    constexpr uint16_t mask[6] = {YP_LEVEL_MASK,YM_LEVEL_MASK,XP_LEVEL_MASK,XM_LEVEL_MASK,ZP_LEVEL_MASK,ZM_LEVEL_MASK};

public:

    APR_iterator(){
        //current_part = 0;
    }

    APRIterator(APRAccess& apr_access_){
       apr_access = &apr_access_;
    }

    void initialize_from_apr(APR<ImageType>& apr){
//        current_part = -2;
//
//        if(apr.num_parts_xy.data.size() == 0) {
//            apr.get_part_numbers();
//            apr.set_part_numbers_xz();
//        }
//
//        this->num_parts = &(apr.num_parts);
//        this->num_parts_xz_pointer = &(apr.num_parts_xy);
//        this->num_parts_total = apr.num_parts_total;
//        this->pc_data_pointer = &(apr.pc_data);
//
//        this->curr_level.init(apr.pc_data);

    }



    uint64_t begin(){
        current_particle_cell.global_index = 0;
        return 0;
    }

    uint64_t begin(unsigned int depth){
//        return this->curr_level.init_iterate((*pc_data_pointer),depth);
    }

    uint64_t end(){
        return (current_particle_cell.global_index == APRAccess->num_parts_total);
    }

    uint64_t end(unsigned int depth){
//        return this->curr_level.counter > 0;
    }

    uint64_t it_forward(){

//        this->curr_level.move_to_next_pc((*pc_data_pointer));
//
//        return this->curr_level.counter;
    }

    uint64_t it_forward(unsigned int depth){

//        this->curr_level.move_to_next_pc((*pc_data_pointer),depth);
//
//        return this->curr_level.counter;
    }


    inline unsigned int particles_level_begin(unsigned int level_){
        //
        //  Used for finding the starting particle on a given level
        //
//        return (*num_parts)[level_-1];
    }

    inline unsigned int particles_level_end(unsigned int level_){
        //
        //  Find the last particle on a given level
        //
//        return (*num_parts)[level_];
    }

    inline unsigned int particles_z_begin(unsigned int level,unsigned int z){
        //
        //  Used for finding the starting particle on a given level
        //
//        if(z > 0) {
//            return ((*num_parts_xz_pointer).data[level][(*pc_data_pointer).x_num[level] * (z-1) + (*pc_data_pointer).x_num[level]-1][0]);
//        } else {
//            return (*num_parts)[level-1];
//        }
    }

    inline unsigned int particles_z_end(unsigned int level,unsigned int z){
        //
        //  Used for finding the starting particle on a given level
        //

//        return ((*num_parts_xz_pointer).data[level][(*pc_data_pointer).x_num[level] * z + (*pc_data_pointer).x_num[level]-1][0]);

    }

    inline unsigned int particles_zx_begin(unsigned int level,unsigned int z, unsigned int x){
        //
        //  Used for finding the starting particle on a given level
        //
//        if(x > 0) {
//            return ((*num_parts_xz_pointer).data[level][(*pc_data_pointer).x_num[level] * (z) + (x-1)][0]);
//        } else {
//            return particles_z_begin(level,z);
//        }
    }

    inline unsigned int particles_zx_end(unsigned int level,unsigned int z, unsigned int x){
        //
        //  Used for finding the starting particle on a given level
        //

//        return ((*num_parts_xz_pointer).data[level][(*pc_data_pointer).x_num[level] * (z) + (x)][0]);

    }


    bool set_iteartor_by_random_access(ParticleCell& input){

    }

    bool set_iterator_to_particle_by_number(uint64_t part_num){
        //
        //  Moves the iterator to be at the set particle number (from depth_min to depth_max, iterating y, x, then z)
        //

//        if(part_num == (current_part+1)){
//            curr_level.move_to_next_pc(*pc_data_pointer);
//            current_part++;
//        } else if(part_num != current_part) {
//            //
//            //  Find the part number
//            //
//
//            if(part_num < current_part){
//                current_part = 0;
//            }
//
//
//            if(current_part >= (num_parts_total)){
//                return false;
//            }
//
//            uint64_t depth = (*pc_data_pointer).depth_min;
//            //first depth search
//            while(((part_num) >= ((*num_parts)[depth])) | ((*num_parts)[depth] ==0) ){
//                depth++;
//            }
//
//            uint64_t offset_start = 0;
//
//            while(part_num >= (*num_parts_xz_pointer).data[depth][offset_start][0]){
//                offset_start++;
//            }
//
//            int total = (*num_parts_xz_pointer).data[depth][offset_start][0];
//
//            uint64_t z_ = (offset_start)/(*num_parts_xz_pointer).x_num[depth];
//            uint64_t x_ = (offset_start) - z_*(*num_parts_xz_pointer).x_num[depth];
//
//            //now it starts at begining and then must iterate forward
//            curr_level.set_new_depth(depth,*pc_data_pointer);
//            curr_level.set_new_z(z_,*pc_data_pointer);
//            curr_level.set_new_x(x_,*pc_data_pointer);
//            curr_level.update_j(*pc_data_pointer,0);
//
//            if(offset_start == 0){
//                current_part = (*num_parts)[depth-1];
//            } else {
//                current_part = (*num_parts_xz_pointer).data[depth][offset_start-1][0];
//            }
//
//            curr_level.move_to_next_pc(*pc_data_pointer);
//
//            while(current_part != part_num){
//
//                curr_level.move_to_next_pc(*pc_data_pointer);
//                current_part++;
//            }
//
//
//        }
//        else if(part_num ==0){
//            curr_level.set_new_depth((*pc_data_pointer).depth_min,*pc_data_pointer);
//            curr_level.set_new_z(0,*pc_data_pointer);
//            curr_level.set_new_x(0,*pc_data_pointer);
//            curr_level.update_j(*pc_data_pointer,0);
//
//            curr_level.move_to_next_pc(*pc_data_pointer);
//
//        }

        return true;

    }


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

        apr_access->get_neighbour_coordinate(org_it.current_particle_cell, current_particle_cell, dir, original_iterator.level_delta, index);

        if(index > 0) {
            //for children need to check boundary conditions
            if (current_particle_cell.x < spatial_index_x_max(neigh.level)) {
                if (current_particle_cell.z < spatial_index_z_max(neigh.level)) {
                    return false;
                }
            }
        }

        return apr_access->find_particle_cell(current_particle_cell);

    }

    inline uint8_t number_neighbours_in_direction(unsigned int dir){

        level_delta =  (current_node & mask[face]) >> shift[face];

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


};


#endif //PARTPLAY_APR_ITERATOR_NEW_HPP
