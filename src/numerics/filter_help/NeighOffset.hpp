///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_NEIGHOFFSET_HPP
#define PARTPLAY_NEIGHOFFSET_HPP
// type T data structure base type




#include "../../data_structures/Tree/PartCellStructure.hpp"
#include "FilterLevel.hpp"

template<typename T,typename V>
class NeighOffset {
    
    friend class FilterLevel<T,V>;
public:
    
    
    
    NeighOffset(int hi_res_flag,bool active_flag): hi_res_flag(hi_res_flag), active_flag(active_flag){
        depth = 0;
        x = 0;
        z = 0;
        j = 0;
        pc_offset = 0;
        y = 0;
        j_num = 0;
        x_num = 0;
        z_num = 0;
        status = 0;
        
    };
    
    
    void set_offsets(int offset_x_,int offset_z_,int offset_y_,int offset_depth_){
        //these are the offsets
        offset_x = offset_x_;
        offset_z = offset_z_;
        offset_y = offset_y_;
       
        high_res_index = offset_x_ + offset_z_*2;
        
        depth_factor_local = powf(2.0,offset_depth_);
        
    }
    
    template<typename U>
    void set_new_depth(T depth_,PartCellStructure<U,T>& pc_struct){
        
        depth = depth_ + offset_depth;
        x_num = pc_struct.x_num[depth];
        z_num = pc_struct.z_num[depth];
        
        depth_factor = powf(2.0,pc_struct.depth_max - depth + 1);
        
    }
    
    
    template<typename U>
    void set_new_xz(T x_,T z_,PartCellStructure<U,T>& pc_struct){
        
        if(active_flag){
            
            x = floor((x_ )*depth_factor_local) + offset_x;
            z = floor((z_ )*depth_factor_local) + offset_z;
            
            x = std::min(x,x_num-1);
            z = std::min(z,z_num-1);
            
            pc_offset = x_num*z + x;
            j_num = pc_struct.pc_data.data[depth][pc_offset].size();
            part_offset = 0;
            seed_offset = (((uint64_t)((x_+offset_x)*depth_factor*2))&1)*2 + (((uint64_t)((z_+offset_z)*depth_factor*2))&1)*4;
        } else {
            j_num = 0;
            y = 64000;
        }
        
        if(j_num > 1){
            y = (pc_struct.pc_data.data[depth][pc_offset][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
            y--;
        } else {
            y = 64000;
        }
        
    }
    
    template<typename U>
    bool incriment_y(uint64_t y_real,PartCellStructure<U,T>& pc_struct){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        //need to deal with offset and
        T y_input = floor((y_real+offset_y)/depth_factor);
        update_flag = false;
        
        if (y_input != y){
            //iterate forward
            while ((y < y_input) & (j < (j_num-1))){
                
                j++;
                node_val = pc_struct.part_data.access_data.data[depth][pc_offset][j];
                
                if (node_val&1){
                    //get the index gap node
                    y += (node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                    y--;
                    
                } else {
                    //normal node
                    y++;
                    status = ((node_val & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                    
                    part_offset = ((node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
                }
                
            }
            
            if (y == y_input){
                update_flag = true;
                return true;
            }else {
                update_flag = false;
                return false;
            }
            
        } else {
            
            if(status == SEED){
                
                //check if moved to next particle
                if( floor(y_real/(depth_factor*2)) == (y*2 + 1)  ){
                    part_offset++; //moved to next seed particle (update)
                    update_flag = true;
                    return true;
                } else {
                    update_flag = false;
                    return false;
                }
                
            }
            update_flag = false;
            return false;
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
    
    T high_res_index;
    
    //offsets
    int offset_x;
    int offset_z;
    int offset_y;
    int offset_depth;
    
    float depth_factor;
    float depth_factor_local;
    
    bool update_flag;
    
    const int hi_res_flag;
    
    const bool active_flag;
    
    std::vector<V> temp_vec;
    
    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP