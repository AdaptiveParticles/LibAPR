//
// Created by cheesema on 03.01.18.
//

/////////////////////////////////////////////////////////
///
///
/// Bevan Cheeseman 2018
///
/// APR access iterator class, allowing various iteration methods through APR datastructures
///
/// relies on the SARI data-structure of V
///
/// Note:
///
//////////////////////////////////////////////////////////


#ifndef PARTPLAY_APR_ITERATOR_HPP
#define PARTPLAY_APR_ITERATOR_HPP

#include "src/numerics/filter_help/CurrLevel.hpp"

#include "src/data_structures/Tree/PartCellStructure.hpp"

#include "src/data_structures/APR/APR.hpp"

template<typename ImageType>
class APR_iterator {

protected:


public:

    PartCellData<uint64_t>* pc_data_pointer;

    ExtraPartCellData<uint64_t>* num_parts_xz_pointer;
    std::vector<float>* num_parts;

    CurrentLevel<ImageType,uint64_t> curr_level;

    uint64_t current_part;

    uint64_t num_parts_total;

    APR_iterator(){
        current_part = 0;
    }

    bool set_part(uint64_t part_num){
        //
        //  Moves the iterator to be at the set particle number (from depth_min to depth_max, iterating y, x, then z)
        //

        if(part_num == (current_part+1)){
            curr_level.move_to_next_pc(*pc_data_pointer);
            current_part++;
        } else if(part_num != current_part) {
            //
            //  Find the part number
            //

            if(part_num < current_part){
                current_part = 0;
            }


            if(current_part >= (num_parts_total)){
                return false;
            }

            uint64_t depth = (*pc_data_pointer).depth_min;
            //first depth search
            while((part_num > (*num_parts)[depth]) | ((*num_parts)[depth] ==0) ){
                depth++;
            }

            uint64_t offset_start = 0;

            while(part_num >= (*num_parts_xz_pointer).data[depth][offset_start][0]){
                offset_start++;
            }

            int total = (*num_parts_xz_pointer).data[depth][offset_start][0];

            uint64_t z_ = (offset_start)/(*num_parts_xz_pointer).x_num[depth];
            uint64_t x_ = (offset_start) - z_*(*num_parts_xz_pointer).x_num[depth];

            //now it starts at begining and then must iterate forward
            curr_level.set_new_depth(depth,*pc_data_pointer);
            curr_level.set_new_z(z_,*pc_data_pointer);
            curr_level.set_new_x(x_,*pc_data_pointer);
            curr_level.update_j(*pc_data_pointer,0);

            if(offset_start == 0){
                current_part = (*num_parts)[depth-1];
            } else {
                current_part = (*num_parts_xz_pointer).data[depth][offset_start-1][0];
            }

            curr_level.move_to_next_pc(*pc_data_pointer);

            while(current_part != part_num){

                curr_level.move_to_next_pc(*pc_data_pointer);
                current_part++;
            }


        } else if(part_num ==0){
            curr_level.set_new_depth((*pc_data_pointer).depth_min,*pc_data_pointer);
            curr_level.set_new_z(0,*pc_data_pointer);
            curr_level.set_new_x(0,*pc_data_pointer);
            curr_level.update_j(*pc_data_pointer,0);

            curr_level.move_to_next_pc(*pc_data_pointer);

        }

        return true;

    }


    inline unsigned int x(){
        //get x
        return curr_level.x;
    }

    inline unsigned int y(){
        //get x
        return curr_level.y;
    }

    inline unsigned int z(){
        //get x
        return curr_level.z;
    }

    inline unsigned int j(){
        //get x
        return curr_level.j;
    }

    inline unsigned int type(){
        //get x
        return curr_level.status;
    }

    inline unsigned int depth(){
        //get x
        return curr_level.depth;
    }

    bool set_neigh_it(APR_iterator<ImageType>& org_it,unsigned int dir,unsigned int index){
        //
        //  Update the iterator to the neighbour
        //

        uint64_t neigh_key = org_it.curr_level.neigh_part_keys.neigh_face[dir][index];

        if(neigh_key > 0) {

            //updates all information except the y info.
            this->curr_level.init(neigh_key, *pc_data_pointer);

            //update the y info using the current position
            if(this->depth() == org_it.depth()){
                //neigh is on same layer
                this->curr_level.y = org_it.y() + (*pc_data_pointer).von_neumann_y_cells[dir];
            }
            else if (this->depth() < org_it.depth()){
                //neigh is on parent layer
                this->curr_level.y = (org_it.y() + (*pc_data_pointer).von_neumann_y_cells[dir])/2;
            }
            else{
                //neigh is on child layer
                this->curr_level.y = (org_it.y() + (*pc_data_pointer).von_neumann_y_cells[dir])*2 +  ((*pc_data_pointer).von_neumann_y_cells[dir] < 0) + (*pc_data_pointer).neigh_child_y_offsets[dir][index];
            }

            return true;
        } else{
            //no particle
            return false;
        }

    }

    inline unsigned int number_neigh(unsigned int dir){
        return this->curr_level.neigh_part_keys.neigh_face[dir].size();
    }


    void update_neigh_all(){
        //updates the internal neighbour structures with the keys to all the neighbours
        this->curr_level.update_neigh_all(*pc_data_pointer);
    }

    void update_neigh_dir(unsigned int dir){
        //updates the internal neighbour structures with the keys to all the neighbours
        this->curr_level.update_neigh_dir(*pc_data_pointer,dir);
    }

    template<typename S>
    void get_neigh_all(ExtraPartCellData<S>& parts,std::vector<std::vector<S>>& neigh_vec){
        //
        // gets all particle neighbours and returns them in a vector with 6 vectors of the particles on each face of the Particle Cell
        //

        this->curr_level.update_and_get_neigh_all(parts,*pc_data_pointer,neigh_vec);

    }

    template<typename S>
    void get_neigh_all_avg(ExtraPartCellData<S>& parts,std::vector<std::vector<S>>& neigh_vec){
        //
        // gets all particle neighbours and returns them in a vector with 6 vectors, if exists provides the average of the neighbours
        //

        this->curr_level.update_and_get_neigh_all_avg(parts,*pc_data_pointer,neigh_vec);

    }

    template<typename S>
    void get_neigh_dir(ExtraPartCellData<S>& parts,std::vector<S>& neigh_vec,unsigned int dir){
        //
        // gets all particle neighbours and returns them in a vector with 6 vectors, if exists provides the average of the neighbours
        //

        this->curr_level.update_get_neigh_dir(parts,*pc_data_pointer,neigh_vec,dir);

    }

    template<typename S>
    S& operator()(ExtraPartCellData<S>& parts){
        //accesses the value of particle data when iterating
        return this->curr_level.get_val(parts);

    }

    inline unsigned int x_nearest_pixel(){
        //get x
        return floor((this->curr_level.x+0.5)*pow(2, (*pc_data_pointer).depth_max - this->curr_level.depth));
    }

    inline float x_global(){
        //get x
        return (this->curr_level.x+0.5)*pow(2, (*pc_data_pointer).depth_max - this->curr_level.depth);
    }

    inline unsigned int y_nearest_pixel(){
        //get x
        return floor((this->curr_level.y+0.5)*pow(2, (*pc_data_pointer).depth_max - this->curr_level.depth));
    }

    inline float y_global(){
        //get x
        return (this->curr_level.y+0.5)*pow(2, (*pc_data_pointer).depth_max - this->curr_level.depth);
    }

    inline unsigned int z_nearest_pixel(){
        //get x
        return floor((this->curr_level.z+0.5)*pow(2, (*pc_data_pointer).depth_max - this->curr_level.depth));
    }

    inline float z_global(){
        //get x
        return (this->curr_level.z+0.5)*pow(2, (*pc_data_pointer).depth_max - this->curr_level.depth);
    }

    inline unsigned int depth_max(){
        return (*pc_data_pointer).depth_max;
    }

    inline unsigned int depth_min(){
        return (*pc_data_pointer).depth_min;
    }


};


#endif //PARTPLAY_APR_ITERATOR_HPP