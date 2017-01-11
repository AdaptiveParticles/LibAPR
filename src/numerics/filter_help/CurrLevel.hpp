///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_CURRENTLEVEL_HPP
#define PARTPLAY_CURRENTLEVEL_HPP
// type T data structure base type

#include "../../data_structures/Tree/PartCellStructure.hpp"

template<typename V,typename T>
class CurrentLevel {
    
public:
    
    
    CurrentLevel(){
        depth = 0;
        x = 0;
        z = 0;
        j = 0;
        pc_offset = 0;
        y = 0;
        j_num = 0;
        x_num = 0;
        z_num = 0;
        
    };
    
    
    template<typename U>
    void set_new_depth(T depth_,ParticleDataNew<U, T>& part_data){
        
        depth = depth_;
        x_num = part_data.access_data.x_num[depth];
        z_num = part_data.access_data.x_num[depth];
        
    }
    
    
    template<typename U>
    void set_new_xz(T x_,T z_,ParticleDataNew<U, T>& part_data){
        
        x = x_;
        z = z_;
        
        pc_offset = x_num*z + x;
        j_num = part_data.access_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;
        
        
    }
    
    template<typename U>
    bool new_j(T j_,const ParticleDataNew<U, T>& part_data){
        j = j_;
        
        node_val = part_data.access_data.data[depth][pc_offset][j_];
        
        //returns if it is a cell or not
        return !(node_val&1);
        
    }
    
    template<typename U>
    U& get_part(ParticleDataNew<U, T>& part_data){
        return part_data.particle_data.data[depth][pc_offset][part_offset];
    }
    
    
    template<typename U>
    void update_cell(ParticleDataNew<U, T>& part_data){
        
        status = part_data.access_node_get_status(node_val);
        
        y++;
        
        //seed offset accoutns for which (x,z) you are doing
        part_offset = part_data.access_node_get_part_offset(node_val);
        
    }
    
    
    void update_gap(){
        
        y += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
        y--;
    }
    
    

    T status;
    T part_offset;
    T pc_offset;
    
    T j_num;
    
    T depth;
    
    T x;
    T z;
    T j;
    
    int y;
        
    T x_num;
    T z_num;
    
    T node_val;
    T y_num;

private:
    
    
    //offsets
    
    
    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP