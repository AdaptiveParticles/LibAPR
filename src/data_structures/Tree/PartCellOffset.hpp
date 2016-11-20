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
        j = 0;
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
        y_factor = pow(2.0,offset_depth_);
    }
    
    template<typename U>
    void set_new_depth(T depth_,PartCellStructure<U,T>& pc_struct){
        
        depth = depth_ + offset_depth;
        x_num = pc_struct.x_num[depth];
        z_num = pc_struct.x_num[depth];
    
    }
    
    template<typename U>
    void set_new_xz(T x_,T z_,PartCellStructure<U,T>& pc_struct){
        
        x = x_ + offset_x;
        z = z_ + offset_z;
        
        pc_offset = x_num*z + x;
        j_num = pc_struct.pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        
        if(j_num > 1){
            y = (pc_struct.pc_data.data[depth][pc_offset][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
        } else {
            y = 64000;
        }

    }
    
    template<typename U>
    bool incriment_y(T y_input,PartCellStructure<U,T>& pc_struct){
        
        //need to deal with offset and
        y_input = floor((y_input+offset_y)*y_factor);
        
        //iterate forward
        while ((y < y_input) & (j < (j_num-1))){
            
            j++;
            node_val = pc_struct.pc_data.data[depth][pc_offset][j];
            
            if (node_val&1){
                //get the index gap node
                y = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                j++;
                
            } else {
                //normal node
                y++;
                part_offset += 1 + (((node_val & STATUS_MASK) >> STATUS_SHIFT) == SEED)*7;
            }
            
        }
        
        if (y == y_input){
            return true;
        }else {
            return false;
        }
        
    }
    
    template<typename U>
    void incriment_y_and_update(T y_input,PartCellStructure<U,T>& pc_struct,std::vector<float>& temp_vec,std::vector<float>& temp_vec_depth){
        
        bool active;
        //update the values
        active = incriment_y(y_input,pc_struct);
        
        if(active == true){
            temp_vec.back() = pc_struct.part_data.particle_data[depth][pc_offset][part_offset];
            temp_vec_depth.back() = depth;
        }
        
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
    T part_offset;
    T node_val;
    
    //offsets
    T offset_x;
    T offset_z;
    T offset_y;
    T offset_depth;
    
    float y_factor;
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP