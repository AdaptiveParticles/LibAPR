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
        depth_factor = pow(2.0,offset_depth_-1);
    }
    
    template<typename U>
    void set_new_depth(T depth_,PartCellStructure<U,T>& pc_struct){
        
        depth = depth_ + offset_depth;
        x_num = pc_struct.x_num[depth];
        z_num = pc_struct.x_num[depth];
    
    }
    
    
    template<typename U>
    void set_new_xz(T x_,T z_,PartCellStructure<U,T>& pc_struct){
        
        x = (x_ + offset_x)*depth_factor;
        z = (z_ + offset_z)*depth_factor;
        
        pc_offset = x_num*z + x;
        j_num = pc_struct.pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        seed_offset = (((uint64_t)((x_+offset_x)*depth_factor*2))&1)*2 + (((uint64_t)((z_+offset_z)*depth_factor*2))&1)*4;
        
        if(j_num > 1){
            y = (pc_struct.pc_data.data[depth][pc_offset][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
            y--;
        } else {
            y = 64000;
        }

    }
    
    template<typename U>
    bool incriment_y(T y_real,PartCellStructure<U,T>& pc_struct){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        //need to deal with offset and
        T y_input = floor((y_real+offset_y)*depth_factor);
        
        
        if (y_input != y){
            //iterate forward
            while ((y < y_input) & (j < (j_num-1))){
                
                j++;
                node_val = pc_struct.part_data.access_data.data[depth][pc_offset][j];
                
                if (node_val&1){
                    //get the index gap node
                    y += (node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                    j++;
                    
                } else {
                    //normal node
                    y++;
                    status = ((node_val & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                   
                    part_offset = ((node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE) + (status == SEED)*seed_offset;
                }
                
            }
            
            if (y == y_input){
                return true;
            }else {
                return false;
            }
            
        } else {
            
            if(status == SEED){
                
                //check if moved to next particle
                if( floor(y_real*(depth_factor/2)) == (y/2 + 1)  ){
                    part_offset++; //moved to next seed particle (update)
                    return true;
                } else {
                    return false;
                }
                
            }
            
            return false;
        }
        
        
        
        
    }
    
    template<typename U>
    void incriment_y_and_update(T y_input,PartCellStructure<U,T>& pc_struct,std::vector<float>& temp_vec,std::vector<float>& temp_vec_depth){
        //
        //  Updates the array location for the move in particle, if the shift, results in the new cell hittin ghte end of the array, you update the array
        //
        
        bool active;
        //update the values
        active = incriment_y(y_input,pc_struct);
        
        if(active == true){
            temp_vec.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset];
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
    T status;
    T seed_offset;
    
    //offsets
    T offset_x;
    T offset_z;
    T offset_y;
    T offset_depth;
    
    float depth_factor;
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP