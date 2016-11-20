///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_PARTCELLOFFSET_HPP
#define PARTPLAY_PARTCELLOFFSET_HPP
// type T data structure base type

#include "PartCellStructure.hpp"

template<typename T>
class PartCellOffset {
    
public:
    
    
    PartCellOffset(){
        depth = 0;
        x = 0;
        z = 0;
        j = 0
        pc_offset = 0;
        y = 0;
        j_num = 0;
        x_num = 0;
        z_num = 0;
    };
    
    void set_offsets(T offset_x_,T offset_z_,T offset_y_,T offset_depth_){
        //these are the offsets
        offset_x = offset_x_;
        offset_z = offset_z_;
        offset_y = offset_y_;
        offset_depth = offset_depth_;
    }
    
    template<typename U>
    void set_new_depth(T depth_,PartCellStructure<U,T>& pc_struct){
        
        depth = depth_ + offset_depth_;
        x_num = pc_struct.x_num[depth];
        z_num = pc_struct.x_num[depth];
    
    }
    
    template<typename U>
    void set_new_xz(T x_,T z_,PartCellStructure<U,T>& pc_struct){
        
        x = x_ + x_offset;
        z = z_ + z_offset;
        
        pc_offset = x_num*z + x;
        j_num = pc_struct.pc_data.data[depth][pc_offset].size();
        
        if(j_num > 1){
            y = (pc_struct.pc_data.data[depth][pc_offset][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
        } else {
            y = 64000;
        }

    }
    
    template<typename U>
    bool incriment_y(const T y_input,PartCellStructure<U,T>& pc_struct){
        
        
        //iterate forward
        while ((y < y_input) & (j < (j_num-1))){
            
            j++;
            node_val = data[i][offset_pc_data_neigh][j_neigh];
            
            if (node_val&1){
                //get the index gap node
                y = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                j++;
                
            } else {
                //normal node
                y++;
            }
            
        }
        
        return true //need to update this
        
    }
    
    
private:
    
    T depth;
    T x;
    T z;
    T j;
    T pc_offset;
    T y;
    T j_num;
    T x_num;
    T z_num;
    
    //offsets
    T x_offset;
    T z_offset;
    T y_offset;
    T depth_offset
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP