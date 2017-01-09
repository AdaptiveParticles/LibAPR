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

#define LOWER_RESOLUTION -1
#define SAME_RESOLUTION 0
#define HIGHER_RESOLUTION 1

#define ACTIVE 1
#define INACTIVE 0


#include "../../data_structures/Tree/PartCellStructure.hpp"
#include "FilterLevel.hpp"

template<typename T,typename V>
class FilterOffset {
    
    friend class FilterLevel<T,V>;
public:
    
    
    
    FilterOffset(int hi_res_flag,bool active_flag): hi_res_flag(hi_res_flag), active_flag(active_flag){
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
        offset_depth = offset_depth_;
        
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
            y = -1;
            y_input = -1;
            
        } else {
            y = 64000;
        }
        j = -1;
    }
    
    template<typename U>
    bool incriment_y(uint64_t y_real,PartCellStructure<U,T>& pc_struct){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        //need to deal with offset and
        y_prev = y_input;
        y_input = floor((y_real+offset_y)/depth_factor);
        update_flag = false;
        
        
        //iterate forward
        while ((y < y_input) & (j < (j_num-1))){
            
            j++;
            node_val = pc_struct.part_data.access_data.data[depth][pc_offset][j];
            
            if (node_val&1){
                //get the index gap node
                y += (node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                j++;
                
                node_val = pc_struct.part_data.access_data.data[depth][pc_offset][j];
                status = ((node_val & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                
                part_offset = ((node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
                
            } else {
                //normal node
                y++;
                status = ((node_val & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                
                part_offset = ((node_val & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
            }
            
        }
        
        if (y == y_input){
            
            if(y_input == y_prev){
                
                if(status == SEED){
                    
                    //check if moved to next particle
                    if( floor(y_real/(depth_factor*2)) == (y*2 + 1)  ){
                        part_offset++; //moved to next seed particle (update)
                        update_flag = true;
                        return true;
                    } else {
                        update_flag = true;
                        return true;
                    }
                }
                else {
                    update_flag = true;
                    return true;
                }
                
                
            } else {
                
                update_flag = true;
                return true;
            }
        } else {
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
    void incriment_y_and_update_new(PartCellStructure<U,uint64_t>& pc_struct,FilterLevel<T,V>& curr_level){
        //
        //  Updates the array location for the move in particle, if the shift, results in the new cell hittin ghte end of the array, you update the array
        //
        
        //update the values
        incriment_y(curr_level.y_global,pc_struct);
        
        update_all_temp_vecs_new(curr_level.y_global,pc_struct,curr_level);
        
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
    void incriment_y_and_update_new(T y_input,PartCellStructure<U,T>& pc_struct,FilterLevel<T,U>& curr_level){
        //
        //  Updates the array location for the move in particle, if the shift, results in the new cell hittin ghte end of the array, you update the array
        //
        
        //update the values
        incriment_y(y_input,pc_struct);
        
        update_all_temp_vecs_new(y_input,pc_struct,curr_level);
        
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
            if((hi_res_flag == SAME_RESOLUTION) & (status == SEED)){
                // Higher resolution level, need to account for the different particles in the cell
                
                //here need to account for status right?
                
                U temp_sum = 0;
                
                curr_level.temp_vec_s0.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                
                curr_level.temp_vec_s1.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
                temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
                
                curr_level.temp_vec_s2.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
                temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
                
                curr_level.temp_vec_s3.back() = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
                temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
                
                curr_level.temp_vec_ns.back() = temp_sum/4.0f;
            } else if (hi_res_flag == HIGHER_RESOLUTION) {
                //Higher resolution, they all get the same values
                
                V temp_sum;
                
                if(status ==SEED){
                    //Seed status need to average the values
                    
                    temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                    temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
                    temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
                    temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
                    
                    temp_sum = temp_sum/4.0f;
                    
                    
                } else {
                    temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                }
                
                switch(high_res_index){
                    case 0:{
                        curr_level.temp_vec_s0.back() = temp_sum;
                        break;
                    }
                    case 1:{
                        curr_level.temp_vec_s1.back() = temp_sum;
                        break;
                    }
                    case 2:{
                        curr_level.temp_vec_s2.back() = temp_sum;
                        break;
                    }
                    case 3:{
                        curr_level.temp_vec_s3.back() = temp_sum;
                        break;
                    }
                        
                        curr_level.temp_vec_ns.back() = temp_sum/4.0f;
                        
                }
                
                
            } else {
                //Lower resolution, they all get the same values
                
                U temp = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + (status == SEED)*seed_offset];
                
                curr_level.temp_vec_s0.back() = temp;
                curr_level.temp_vec_s1.back() = temp;
                curr_level.temp_vec_s2.back() = temp;
                curr_level.temp_vec_s3.back() = temp;
                
                curr_level.temp_vec_ns.back() = temp;
                
            }
        }
        
    }
    template<typename U>
    void update_all_temp_vecs_new(uint64_t index,PartCellStructure<U,T>& pc_struct,FilterLevel<T,V>& curr_level){
        //update all the current vectors if required (update_flag is set as to whether this is the current new cell)
        
        if(update_flag){
            
            index = std::min((uint64_t)(index +curr_level.filter_offset),(uint64_t)(pc_struct.org_dims[0]-1));
            
            
            if(hi_res_flag== SAME_RESOLUTION){
                int stop = 1;
            }
            
            if((hi_res_flag == SAME_RESOLUTION) & (status == SEED)){
                // Higher resolution level, need to account for the different particles in the cell
                
                //here need to account for status right?
                
                U temp_sum = 0;
                
                curr_level.temp_vec_s0[index] = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                
                curr_level.temp_vec_s1[index] = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
                temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
                
                curr_level.temp_vec_s2[index] = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
                temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
                
                curr_level.temp_vec_s3[index] = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
                temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
                
                curr_level.temp_vec_ns[index] = (temp_sum)/4.0f;
                
                
            } else if (hi_res_flag == HIGHER_RESOLUTION) {
                //Higher resolution, they all get the same values
                
                V temp_sum;
                
                if(status ==SEED){
                    //Seed status need to average the values
                    
                    temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                    temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 2];
                    temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 4];
                    temp_sum += pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 6];
                    
                    temp_sum = temp_sum/4.0f;
                    
                    
                } else {
                    temp_sum = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + 0];
                }
                
                switch(high_res_index){
                    case 0:{
                        curr_level.temp_vec_s0[index] = temp_sum;
                        break;
                    }
                    case 1:{
                        curr_level.temp_vec_s1[index] = temp_sum;
                        break;
                    }
                    case 2:{
                        curr_level.temp_vec_s2[index] = temp_sum;
                        break;
                    }
                    case 3:{
                        curr_level.temp_vec_s3[index] = temp_sum;
                        break;
                    }
                }
                curr_level.temp_vec_ns[index] += temp_sum/4.0f;
                
            } else {
                //Lower resolution, they all get the same values
                
                U temp = pc_struct.part_data.particle_data.data[depth][pc_offset][part_offset + (status == SEED)*seed_offset];
                
                //int index_max = std::min((int)curr_level.temp_vec_s0.size()-1,(int)(index + depth_factor));
                
                //std::fill(curr_level.temp_vec_s0.begin() + index,curr_level.temp_vec_s0.begin() + index_max,temp);
                //std::fill(curr_level.temp_vec_s1.begin() + index,curr_level.temp_vec_s1.begin() + index_max,temp);
                //std::fill(curr_level.temp_vec_s2.begin() + index,curr_level.temp_vec_s2.begin() + index_max,temp);
                //std::fill(curr_level.temp_vec_s3.begin() + index,curr_level.temp_vec_s3.begin() + index_max,temp);
                
                //std::fill(curr_level.temp_vec_ns.begin() + index,curr_level.temp_vec_ns.begin() + index_max,temp);
                
                
                curr_level.temp_vec_s0[index] = temp;
                curr_level.temp_vec_s1[index] = temp;
                curr_level.temp_vec_s2[index] = temp;
                curr_level.temp_vec_s3[index] = temp;
                
                curr_level.temp_vec_ns[index] = temp;
                
            }
        }
        
    }
    
    
    
private:
    
    T depth;
    T x;
    T z;
    int j;
    T pc_offset;
    int y;
    int j_num;
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

    int y_input;
    int y_prev;


};

#endif //PARTPLAY_PARTCELLOFFSET_HPP