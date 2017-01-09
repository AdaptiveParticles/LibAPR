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

#define SAME_LEVEL 1
#define PARENT_LEVEL 2
#define CHILD_LEVEL 3


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
        
        depth_parent = depth_ - 1;
        x_num_parent = part_data.access_data.x_num[depth_parent];
        z_num_parent = part_data.access_data.z_num[depth_parent];
        
        depth_child = depth_ + 1;
        x_num_child = part_data.access_data.x_num[depth_child];
        z_num_child = part_data.access_data.z_num[depth_child];
        
    }
    
    
    template<typename U>
    void reset_j(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        
        
        
        
        if(depth_child < part_data.access_data.depth_max){
            
            x_child = curr_level.x*2 + offset_x;
            z_child = curr_level.z*2 + offset_z;
            
            x_child = std::min(x_child,x_num_child-1);
            z_child = std::min(z_child,z_num_child-1);
            
            
            pc_offset_child = x_num_child*z_child + x_child;
            j_num_child = part_data.access_data.data[depth_child][pc_offset_child].size();
            part_offset_child = 0;
            
            
            j_child = 0;
            
            if((x_child < 0) | (z_child < 0)){
                j_num_child = 0;
            }
            
            
            
            if(j_num_child > 1){
                y_child = -1;
                
            } else {
                y_child = 64000;
            }
            
            int dir = (offset_y == -1) + (offset_x == 1)*2 + (offset_x == -1)*3 + (offset_z == 1)*4 + (offset_z == -1)*5;
            
            pc_offset_child_1 = x_num_child*z_child + x_child;
            pc_offset_child_2 = x_num_child*(z_child + child_z[neigh_child_dir[dir][0]]) + (x_child+child_x[neigh_child_dir[dir][0]]);
            pc_offset_child_3 = x_num_child*(z_child + child_z[neigh_child_dir[dir][1]]) + (x_child+child_x[neigh_child_dir[dir][1]]);
            pc_offset_child_4 = x_num_child*(z_child + child_z[neigh_child_dir[dir][2]]) + (x_child+child_x[neigh_child_dir[dir][2]]);
            
            
            j_offset_child_1 = 0;
            j_offset_child_2 = child_y[neigh_child_dir[dir][0]];
            j_offset_child_3 = child_y[neigh_child_dir[dir][1]];
            j_offset_child_4 = child_y[neigh_child_dir[dir][2]];
            
            
            

            
            
        } else {
            y_child = 64000;
        }
        
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
            
            y_parent = 64000;
            
            
        } else {
            
            //parent level
            //same level
            
            if(depth_parent > part_data.access_data.depth_min){
                x_parent = curr_level.x/2 + offset_x;
                z_parent = curr_level.z/2 + offset_z;
                
                x_parent = std::min(x_parent,x_num_parent-1);
                z_parent = std::min(z_parent,z_num_parent-1);
                
                
                pc_offset_parent = x_num_parent*z_parent + x_parent;
                j_num_parent = part_data.access_data.data[depth_parent][pc_offset_parent].size();
                part_offset_parent = 0;
                
                
                j_parent = 0;
                
                if((x_parent < 0) | (z_parent < 0)){
                    j_num_parent = 0;
                }
                
                
                
                if(j_num_parent > 1){
                    y_parent = -1;
                    
                } else {
                    y_parent = 64000;
                }
            } else {
                y_parent = 64000;
            }
            
            
            //same level
            x_same = curr_level.x + offset_x;
            z_same = curr_level.z + offset_z;
            
            if(offset_x == 1){
                
                if(curr_level.z&1){
                    part_xz_1_parent = 4;
                    
                } else {
                    part_xz_1_parent = 2;
                }
                
                part_offset_1_parent = 0;
                
                part_xz_1_same = 1;
                part_xz_2_same = 1;
                part_xz_3_same = 3;
                part_xz_4_same = 3;
                
                part_offset_1_same = 0;
                part_offset_2_same = 1;
                part_offset_3_same = 0;
                part_offset_4_same = 1;
                
                
                
                
            } else if (offset_x == -1){
                
                if(curr_level.z&1){
                    part_xz_1_parent = 3;
                    
                } else {
                    part_xz_1_parent = 1;
                }
                
                part_offset_1_parent = 0;
                
                part_xz_1_same = 2;
                part_xz_2_same = 2;
                part_xz_3_same = 4;
                part_xz_4_same = 4;
                
                part_offset_1_same = 0;
                part_offset_2_same = 1;
                part_offset_3_same = 0;
                part_offset_4_same = 1;
                
            } else if(offset_z == 1){
                
                if(curr_level.x&1){
                    part_xz_1_parent = 2;
                    
                } else {
                    part_xz_1_parent = 1;
                }
                
                part_offset_1_parent = 0;
                
                part_xz_1_same = 1;
                part_xz_2_same = 1;
                part_xz_3_same = 2;
                part_xz_4_same = 2;
                
                part_offset_1_same = 0;
                part_offset_2_same = 1;
                part_offset_3_same = 0;
                part_offset_4_same = 1;
                
                
            } else if(offset_z == -1){
                
                if(curr_level.x&1){
                    part_xz_1_parent = 3;
                    
                } else {
                    part_xz_1_parent = 4;
                }
                part_offset_1_parent = 0;
                
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
                
                part_offset_1_parent = 0;
                if(curr_level.x&1){
                    if(curr_level.z&1){
                        part_xz_1_parent = 4;
                        
                    } else {
                        part_xz_1_parent = 2;
                    }
                    
                } else {
                    if(curr_level.z&1){
                        part_xz_1_parent = 3;
                        
                    } else {
                        part_xz_1_parent = 1;
                    }
                }
                
                
                
            } else if (offset_y == -1){
                part_xz_1_same = 1;
                part_xz_2_same = 2;
                part_xz_3_same = 3;
                part_xz_4_same = 4;
                
                part_offset_1_same = 1;
                part_offset_2_same = 1;
                part_offset_3_same = 1;
                part_offset_4_same = 1;
                
                part_offset_1_parent = 1;
                if(curr_level.x&1){
                    if(curr_level.z&1){
                        part_xz_1_parent = 4;
                        
                    } else {
                        part_xz_1_parent = 2;
                    }
                    
                } else {
                    if(curr_level.z&1){
                        part_xz_1_parent = 3;
                        
                    } else {
                        part_xz_1_parent = 1;
                    }
                }
                
            }
            
        }
        
        
        
        
        x_same = std::min(x_same,x_num_same-1);
        z_same = std::min(z_same,z_num_same-1);
        
        pc_offset_same = x_num_same*z_same + x_same;
        j_num_same = part_data.access_data.data[depth_same][pc_offset_same].size();
        part_offset_same = 0;
        
        
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
    void incriment_y_same_depth(const CurrentLevel<U,T>& curr_level,const ParticleDataNew<U, T>& part_data){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        current_flag = 0;
        //need to deal with offset and
        int y_input = curr_level.y + offset_y;
        
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
            current_flag = SAME_LEVEL;
        }
        
        
        
        
    }
    template<typename U>
    void incriment_y_child_depth(const CurrentLevel<U,T>& curr_level,const ParticleDataNew<U, T>& part_data){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        
        //need to deal with offset and
        int y_input = 2*curr_level.y + offset_y;
        
        //iterate forward
        while ((y_child < y_input) & (j_child < (j_num_child-1))){
            
            j_child++;
            node_val_child = part_data.access_data.data[depth_child][pc_offset_child][j_child];
            
            if (node_val_child&1){
                //get the index gap node
                y_child += (node_val_same & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                j_child++;
                
                node_val_child = part_data.access_data.data[depth_child][pc_offset_child][j_child];
                status_child = ((node_val_child & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                part_offset_child = ((node_val_child & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
                
                
            } else {
                //normal node
                y_child++;
                status_child = ((node_val_child & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                
                part_offset_child = ((node_val_child & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
            }
            
        }
        
        if (y_child == y_input){
            current_flag = CHILD_LEVEL;
        }
        
        
        
        
    }

    
    template<typename U>
    void incriment_y_parent_depth(const CurrentLevel<U,T>& curr_level,const ParticleDataNew<U, T>& part_data){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        //need to deal with offset and
        int y_input = curr_level.y/2 + offset_y;
        
        //iterate forward
        while ((y_parent < y_input) & (j_parent < (j_num_same-1))){
            
            j_parent++;
            node_val_parent = part_data.access_data.data[depth_parent][pc_offset_parent][j_parent];
            
            if (node_val_parent&1){
                //get the index gap node
                y_parent += (node_val_parent & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                j_parent++;
                
                node_val_parent = part_data.access_data.data[depth_parent][pc_offset_parent][j_parent];
                status_parent = ((node_val_parent & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                part_offset_parent = ((node_val_parent & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
                
                
            } else {
                //normal node
                y_parent++;
                status_parent = ((node_val_parent & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                
                part_offset_parent = ((node_val_parent & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
            }
            
        }
        
        if (y_parent == y_input){
            current_flag = PARENT_LEVEL;
            
            if(offset_y == 0){
                part_offset_1_parent = (curr_level.y&1);
            }
            
        }
        
        
        
        
    }
    
    
    
    template<typename U>
    void incriment_y_part_same_depth(const CurrentLevel<U,T>& curr_level,const ParticleDataNew<U, T>& part_data){
        
        if(current_flag == SAME_LEVEL){
            if(status_same == SEED){
                part_offset_same++;
            }
            
        }
        
    }
    
    template<typename S>
    S get_part(const ExtraPartCellData<std::vector<S>>& p_data){
        
        if(current_flag ==SAME_LEVEL){
            
            if(part_xz == 0){
                //iterating non seed particles
                if (status_same == SEED){
                    return seed_reduce_operator_same(p_data);
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
            
            
        } else if (current_flag == PARENT_LEVEL){
            
            if(status_parent == SEED){
                
                
                return p_data.data[depth_parent][pc_offset_parent][part_xz_parent][part_offset_parent + part_offset_1_parent];
                
                
                
            } else {
                return p_data.data[depth_parent][pc_offset_parent][0][part_offset_parent];
                
                
            }
            
            
        } else if (current_flag == CHILD_LEVEL){
            return p_data.data[depth_child][pc_offset_child][0][part_offset_child];
        }
        else {
            return 0;
        }
        
        
        
    }


    template<typename U>
    U seed_reduce_operator_same(const ExtraPartCellData<std::vector<U>>& p_data){
        
        U temp =p_data.data[depth_same][pc_offset_same][part_xz_1_same][part_offset_same + part_offset_1_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_2_same][part_offset_same + part_offset_2_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_3_same][part_offset_same + part_offset_3_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_4_same][part_offset_same + part_offset_4_same];
        return temp;
        
        
    }
    
    template<typename U>
    U seed_reduce_operator_child(const ExtraPartCellData<std::vector<U>>& p_data){
        
        U temp =p_data.data[depth_same][pc_offset_same][part_xz_1_same][part_offset_same + part_offset_1_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_2_same][part_offset_same + part_offset_2_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_3_same][part_offset_same + part_offset_3_same];
        temp +=p_data.data[depth_same][pc_offset_same][part_xz_4_same][part_offset_same + part_offset_4_same];
        return temp;
        
        
    }
    
private:
    
    const int8_t child_y[6] = { 1, 0, 0, 1, 0, 1};
    const int8_t child_x[6] = { 0, 1, 1, 0, 0, 1};
    const int8_t child_z[6] = { 0, 1, 0, 1, 1, 0};
    
    //the ordering of retrieval of four neighbour cells
    const uint8_t neigh_child_dir[6][3] = {{2,4,1},{2,4,1},{0,4,3},{0,4,3},{0,2,5},{0,2,5}};
    
    
    
    T depth_same;
    T x_same;
    T z_same;
    T j_same;
    T pc_offset_same;
    int y_same;
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
    
    
    T depth_parent;
    T x_parent;
    T z_parent;
    T j_parent;
    T pc_offset_parent;
    int y_parent;
    T j_num_parent;
    T x_num_parent;
    T z_num_parent;
    T part_offset_parent;
    T node_val_parent;
    T status_parent;
    T seed_offset_parent;
    T part_xz_parent;
    
    T part_xz_1_parent;
    
    
    T part_offset_1_parent;
    
    T depth_child;
    T x_child;
    T z_child;
    T j_child;
    T pc_offset_child;
    int y_child;
    T j_num_child;
    T x_num_child;
    T z_num_child;
    T part_offset_child;
    T node_val_child;
    T status_child;
    
    T part_xz_child;
    
    
    T pc_offset_child_1;
    T pc_offset_child_2;
    T pc_offset_child_3;
    T pc_offset_child_4;
    
    T part_offset_child_1;
    T part_offset_child_2;
    T part_offset_child_3;
    T part_offset_child_4;

    
    T j_offset_child_1;
    T j_offset_child_2;
    T j_offset_child_3;
    T j_offset_child_4;
    
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