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
#include "CurrLevel.hpp"

template<typename V,typename T>
class NeighOffset {
    
public:
    
    
    
    NeighOffset(){
        
    };
    
    
    void set_offsets(int offset_x_,int offset_z_,int offset_y_,int offset_depth_){
        //these are the offsets
        offset_x = offset_x_;
        offset_z = offset_z_;
        offset_y = offset_y_;
        offset_depth = offset_depth_;
        
        
    }
    
    template<typename U>
    void set_new_depth(T depth_,ParticleDataNew<U, T>& part_data){
        
        depth_same = depth_;
        x_num_same = part_data.access_data.x_num[depth_same];
        z_num_same = part_data.access_data.z_num[depth_same];
        
    
        
    }
    
    
    template<typename U>
    void reset_j(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        
        part_xz = curr_level.part_xz;
        
        if (curr_level.part_xz > 1){
            
            switch(curr_level.part_xz){
                case 1 :{
                    x_same = (2*curr_level.x + offset_x);
                    z_same = (2*curr_level.z + offset_z);
                    
                    part_xz_same = (x_same&1) + 2*(z_same&1);
                    
                    x_same = x_same/2;
                    z_same = z_same/2;
                    
                    break;
                    
                }
                case 2 :{
                    x_same = (2*curr_level.x + offset_x + 1);
                    z_same = (2*curr_level.z + offset_z);
                    
                    part_xz_same = (x_same&1) + 2*(z_same&1);
                    
                    x_same = x_same/2;
                    z_same = z_same/2;
                    
                    break;
                }
                case 3 :{
                    x_same = (2*curr_level.x + offset_x);
                    z_same = (2*curr_level.z + offset_z + 1);
                    
                    part_xz_same = (x_same&1) + 2*(z_same&1);
                    
                    x_same = x_same/2;
                    z_same = z_same/2;
                    
                    break;
                    
                }
                case 4 :{
                    x_same = (2*curr_level.x + offset_x + 1);
                    z_same = (2*curr_level.z + offset_z + 1);
                    
                    part_xz_same = (x_same&1) + 2*(z_same&1);
                    
                    x_same = x_same/2;
                    z_same = z_same/2;
                    
                    break;
                    
                }
                    
            }
            
        } else {
            
            x_same = curr_level.x + offset_x;
            z_same = curr_level.z + offset_z;
            
            if(offset_x == 1){
            
                part_xz_1_same = 1;
                part_xz_2_same = 1;
                part_xz_3_same = 3;
                part_xz_4_same = 3;
            
                part_offset_1_same = 0;
                part_offset_2_same = 1;
                part_offset_3_same = 0;
                part_offset_4_same = 1;
            } else if (offset_x == -1){
                part_xz_1_same = 2;
                part_xz_2_same = 2;
                part_xz_3_same = 4;
                part_xz_4_same = 4;
                
                part_offset_1_same = 0;
                part_offset_2_same = 1;
                part_offset_3_same = 0;
                part_offset_4_same = 1;
                
            } else if(offset_z == 1){
                part_xz_1_same = 1;
                part_xz_2_same = 1;
                part_xz_3_same = 2;
                part_xz_4_same = 2;
                
                part_offset_1_same = 0;
                part_offset_2_same = 1;
                part_offset_3_same = 0;
                part_offset_4_same = 1;
                
                
            } else if(offset_z == -1){
                part_xz_1_same = 3;
                part_xz_2_same = 3;
                part_xz_3_same = 4;
                part_xz_4_same = 4;
                
                part_offset_1_same = 0;
                part_offset_2_same = 1;
                part_offset_3_same = 0;
                part_offset_4_same = 1;
            } else if (offset_y == 1){
                part_xz_1_same = 1;
                part_xz_2_same = 2;
                part_xz_3_same = 3;
                part_xz_4_same = 4;
                
                part_offset_1_same = 0;
                part_offset_2_same = 0;
                part_offset_3_same = 0;
                part_offset_4_same = 0;
                
                
                
            } else if (offset_y == -1){
                part_xz_1_same = 1;
                part_xz_2_same = 2;
                part_xz_3_same = 3;
                part_xz_4_same = 4;
                
                part_offset_1_same = 1;
                part_offset_2_same = 1;
                part_offset_3_same = 1;
                part_offset_4_same = 1;
                
                
                
            }
            
        }
        
        x_same = std::min(x_same,x_num_same-1);
        z_same = std::min(z_same,z_num_same-1);
        
        pc_offset_same = x_num_same*z_same + x_same;
        j_num_same = part_data.access_data.data[depth_same][pc_offset_same].size();
        part_offset_same = 0;
        
        
        if(pc_offset_same == 2715){
            int stop = 1;
        }
        
        j_same = 0;
        
        if((x_same < 0) | (z_same < 0)){
            j_num_same = 0;
        }
        
        if(j_num_same > 1){
            y_same = -1;
            
        } else {
            y_same = 64000;
        }
        
    }
    
    template<typename U>
    void incriment_y_same_depth(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        current_flag = 0;
        //need to deal with offset and
        T y_input = curr_level.y + offset_y;
        
        //iterate forward
        while ((y_same < y_input) & (j_same < (j_num_same-1))){
            
            j_same++;
            node_val_same = part_data.access_data.data[depth_same][pc_offset_same][j_same];
            
            if (node_val_same&1){
                //get the index gap node
                y_same += (node_val_same & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                j_same++;
                
                node_val_same = part_data.access_data.data[depth_same][pc_offset_same][j_same];
                status_same = ((node_val_same & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                part_offset_same = ((node_val_same & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
                
                
            } else {
                //normal node
                y_same++;
                status_same = ((node_val_same & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                
                part_offset_same = ((node_val_same & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
            }
            
        }
        
        if (y_same == y_input){
            current_flag = 1;
        }
        
        
        
        
    }
    
    template<typename U>
    void incriment_y_part_same_depth(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        
        if(current_flag == 1){
            if(status_same == SEED){
                part_offset_same++;
            }
            
        }
        
    }
    
    template<typename S>
    S get_part(ExtraPartCellData<std::vector<S>>& p_data){
        
        if(current_flag ==1){
            
            if(part_xz == 0){
                //iterating non seed particles
                if (status_same == SEED){
                    return seed_reduce_operator(p_data);
                } else {
                    return p_data.data[depth_same][pc_offset_same][0][part_offset_same];
                }
                
                
            } else {
                //iterating seed particles
                
                if (status_same == SEED){
                    return p_data.data[depth_same][pc_offset_same][part_xz_same][part_offset_same];
                } else {
                    return p_data.data[depth_same][pc_offset_same][0][part_offset_same];
                }
                
            }
            
            
        } else {
            return 0;
        }
        
        
        
    }
    
    template<typename U>
    U seed_reduce_operator(ExtraPartCellData<std::vector<U>>& p_data){
        
        U temp =p_data.data[depth_same][pc_offset_same][part_xz_1_same][part_offset_same + part_offset_1_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_2_same][part_offset_same + part_offset_2_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_3_same][part_offset_same + part_offset_3_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_4_same][part_offset_same + part_offset_4_same];
        return temp;
        
        
    }
    
private:
    
    T depth_same;
    T x_same;
    T z_same;
    T j_same;
    T pc_offset_same;
    T y_same;
    T j_num_same;
    T x_num_same;
    T z_num_same;
    T part_offset_same;
    T node_val_same;
    T status_same;
    T seed_offset_same;
    T part_xz_same;
    
    T part_xz_1_same;
    T part_xz_2_same;
    T part_xz_3_same;
    T part_xz_4_same;
    
    T part_offset_1_same;
    T part_offset_2_same;
    T part_offset_3_same;
    T part_offset_4_same;
    
    T high_res_index;
    
    T part_xz;
    bool seed;
    
    
    //offsets
    int offset_x;
    int offset_z;
    int offset_y;
    int offset_depth;
    
    
    
    float depth_factor;
    float depth_factor_local;
    
    int current_flag;
    
    uint64_t not_exist;
    
    std::vector<V> temp_vec;
    
    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP