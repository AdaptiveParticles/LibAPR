///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_FILTEROFFSET_HPP
#define PARTPLAY_FILTEROFFSET_HPP
// type T data structure base type

#include "../../data_structures/Tree/PartCellStructure.hpp"
#include "FilterLevel.hpp"

template<typename T,typename V>
class FilterOffset {
    
    friend class FilterLevel<T,V>;
public:
    
    
    
    FilterOffset(bool hi_res_flag,bool active_flag): hi_res_flag(hi_res_flag), active_flag(active_flag){
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
        
        if(active_flag){
        
            x = (x_ + offset_x)*depth_factor;
            z = (z_ + offset_z)*depth_factor;
        
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
        T y_input = floor((y_real+offset_y)*depth_factor);
        update_flag = false;
        
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
                update_flag = true;
                return true;
            }else {
                update_flag = false;
                return false;
            }
            
        } else {
            
            if(status == SEED){
                
                //check if moved to next particle
                if( floor(y_real*(depth_factor/2)) == (y/2 + 1)  ){
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
    
    
    template<typename U>
    void incriment_y_and_update(PartCellStructure<U,uint64_t>& pc_struct,FilterLevel<T,V>& curr_level){
        //
        //  Updates the array location for the move in particle, if the shift, results in the new cell hittin ghte end of the array, you update the array
        //
        
        //update the values
        incriment_y(curr_level.y_global,pc_struct);
        
        update_all_temp_vecs(pc_struct,curr_level);
        
    }
    
    template<typename U>
    void incriment_y_and_update(T y_input,PartCellStructure<U,T>& pc_struct,FilterLevel<T,U>& curr_level){
        //
        //  Updates the array location for the move in particle, if the shift, results in the new cell hittin ghte end of the array, you update the array
        //
        
        //update the values
        incriment_y(y_input,pc_struct);
        
        update_all_temp_vecs(pc_struct,curr_level);
        
    }
    
    template<typename U>
    void update_temp_vec(PartCellStructure<U,T>& pc_struct,std::vector<float>& temp_vec,uint64_t seed_offset){
        
        if(update_flag){
            temp_vec.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + seed_offset];
        }
        
    }
    
    template<typename U>
    void update_all_temp_vecs(PartCellStructure<U,T>& pc_struct,FilterLevel<T,V>& curr_level){
        //update all the current vectors if required (update_flag is set as to whether this is the current new cell)
        
        if(update_flag){
            if(hi_res_flag){
                // Higher resolution level, need to account for the different particles in the cell
                
                U temp_sum = 0;
            
                curr_level.temp_vec_s0.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
            
                curr_level.temp_vec_s1.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
                temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
            
                curr_level.temp_vec_s2.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
                temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
            
                curr_level.temp_vec_s3.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
                temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
            
                curr_level.temp_vec_ns.back() = temp_sum/4.0f;
            } else {
                //Lower resolution, they all get the same values
                
                U temp = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                
                curr_level.temp_vec_s0.back() = temp;
                curr_level.temp_vec_s1.back() = temp;
                curr_level.temp_vec_s2.back() = temp;
                curr_level.temp_vec_s3.back() = temp;
                
                curr_level.temp_vec_ns.back() = temp;
                
                
            }
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
    
    bool update_flag;
    
    const bool hi_res_flag;
    
    const bool active_flag;
    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP