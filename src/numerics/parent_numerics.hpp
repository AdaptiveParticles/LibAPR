////////////////////////////////////////////////////////////
//
//
//  Bevan Cheeseman 2016
//
//
//  Numerical Methods on the APR using the tree structure
//
//
///////////////////////////////////////////////////////////

#ifndef _segment_h
#define _segment_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"


template <typename T,typename S,typename U>
void calc_cell_min_max(PartCellStructure<T,S>& pc_struct,PartCellParent<S>& pc_parent,ExtraPartCellData<U>& particle_data,ExtraPartCellData<U>& min_data,ExtraPartCellData<U>& max_data){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Computes min and max up tree (min and max will be of parent cell size)
    //
    //  Input: pc_struct and particle data, (typical case just input the intensities)
    //
    
    min_data.initialize_structure_cells(pc_parent.neigh_info);
    max_data.initialize_structure_cells(pc_parent.neigh_info);
    
    
    //loop over parent cells children, if it is a real node, then if it is SEED avg particles, if it is not take them, then compute min and max (done)
    
    /////////////////////////
    //
    //  Parent Loop
    //
    ////////////////////////////
    
    Part_timer timer;
    timer.verbose_flag = true;
    
    uint64_t x_;
    uint64_t j_;
    uint64_t z_;
    uint64_t curr_key;
    uint64_t status;
    uint64_t part_offset;
    
    uint64_t node_val_parent;
    uint64_t node_val_part;
    std::vector<uint64_t> children_keys;
    std::vector<uint64_t> children_ind;
    
    children_keys.resize(8,0);
    children_ind.resize(8,0);
    
    timer.start_timer("PARENT LOOP");
    
    //reverse loop direction
    for(uint64_t i = pc_parent.neigh_info.depth_max;i >= pc_parent.neigh_info.depth_min;i--){
        //loop over the resolutions of the structure
        const unsigned int x_num_ =  pc_parent.neigh_info.x_num[i];
        const unsigned int z_num_ =  pc_parent.neigh_info.z_num[i];
        
        
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,part_offset,node_val_part) firstprivate(children_keys,children_ind) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_parent.neigh_info.pc_key_set_z(curr_key,z_);
            pc_parent.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_parent.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_parent.neigh_info.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_parent = pc_parent.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_parent&1)){
                        //Indicates this is a particle cell node
                        
                        
                        pc_parent.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        status = pc_parent.neigh_info.get_status(node_val_parent);
                        
                        //parent has real siblings
                        if(status > 0){
                            
                            //get the children
                            
                            pc_parent.get_children_keys(curr_key,children_keys,children_ind);
                            
                            U min_temp = 99999999;
                            U max_temp = 0;
                            
                            for(int c = 0;c < children_keys.size();c++){
                                uint64_t child = children_keys[c];
                                
                                if(child > 0){
                                    
                                    if(children_ind[c] == 1){
                                        //loop over the particles
                                        node_val_part = pc_struct.part_data.access_data.get_val(child);
                                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                                        
                                        float mean = 0;
                                        
                                        //loop over the particles
                                        for(int p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                                            pc_struct.part_data.access_data.pc_key_set_index(child,part_offset+p);
                                            
                                            mean += pc_struct.part_data.particle_data.get_part(child);
                                        }
                                        
                                        mean = mean/pc_struct.part_data.get_num_parts(status);
                                        
                                        min_temp = std::min(mean,min_temp);
                                        max_temp = std::max(mean,max_temp);
                                        
                                    } else {
                                        min_temp = std::min(min_data.get_val(child),min_temp);
                                        max_temp = std::max(max_data.get_val(child),max_temp);
                                    }
                                    
                                    
                                }
                            }
                            
                            //now set the values
                            min_data.get_val(curr_key) = min_temp;
                            max_data.get_val(curr_key) = max_temp;
                            

                            
                        }
                    }
                }
            }
        }
    }
    
    timer.stop_timer();
    
    
    timer.start_timer("check");
    
    
    //
    
    //reverse loop direction
    for(uint64_t i = pc_parent.neigh_info.depth_max;i >= pc_parent.neigh_info.depth_min;i--){
        //loop over the resolutions of the structure
        const unsigned int x_num_ =  pc_parent.neigh_info.x_num[i];
        const unsigned int z_num_ =  pc_parent.neigh_info.z_num[i];
        
        
        //#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,part_offset,node_val_part) firstprivate(children_keys,children_ind) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_parent.neigh_info.pc_key_set_z(curr_key,z_);
            pc_parent.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_parent.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_parent.neigh_info.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_parent = pc_parent.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_parent&1)){
                        //Indicates this is a particle cell node
                        
                        
                        pc_parent.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        
                        float check_min = min_data.get_val(curr_key);
                        float check_max = max_data.get_val(curr_key);
                        
                        if(check_min < 500){
                            int stop = 1;
                        }
                        
                        
                        
                    }
                }
            }
        }
    }


    timer.stop_timer();
    
}
template<typename T,typename U>
T compute_parent_cell_neigh_mean(const U& parent_node,const U& parent_key,ExtraPartCellData<T>& parent_data,PartCellParent<U>& parent_cells){
    //
    //  Bevan Cheeseman 2016
    //
    //  Given a data set defined on parent cells, compute the mean over the parent neighbours on the same level
    //
    //
    
    PartCellNeigh<U> neigh_keys;
    
    parent_cells.get_neighs_parent_all(parent_key,parent_node,neigh_keys);
    
    T temp = parent_data.get_val(parent_key);
    float counter = 1;
    T val=0;
    
    for(uint64_t face = 0; face < neigh_keys.neigh_face.size();face++){
        
        for(uint64_t n = 0; n < neigh_keys.neigh_face[face].size();n++){
            uint64_t neigh_key = neigh_keys.neigh_face[face][n];
            
            if(neigh_key > 0){
                val= parent_data.get_val(neigh_key);
                if (val > 0){
                    counter++;
                    temp+=val;
                }
            }
            
        }
    }
    
    return (temp/counter);
    
}
template<typename T,typename U>
void smooth_parent_result(PartCellParent<U>& pc_parent,ExtraPartCellData<T>& parent_data){
    //
    //
    //  Calculates an average on every part level
    //
    //
    
    ExtraPartCellData<T> output;
    output.initialize_structure_cells(pc_parent.neigh_info);
    
    Part_timer timer;
    timer.verbose_flag = true;
    
    uint64_t x_;
    uint64_t j_;
    uint64_t z_;
    uint64_t curr_key;
    uint64_t status;
    
    uint64_t node_val_parent;
    uint64_t node_val_part;

    timer.start_timer("calc mean");
    
    //reverse loop direction
    for(uint64_t i = pc_parent.neigh_info.depth_max;i >= pc_parent.neigh_info.depth_min;i--){
        //loop over the resolutions of the structure
        const unsigned int x_num_ =  pc_parent.neigh_info.x_num[i];
        const unsigned int z_num_ =  pc_parent.neigh_info.z_num[i];
        
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,node_val_part)  if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_parent.neigh_info.pc_key_set_z(curr_key,z_);
            pc_parent.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_parent.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_parent.neigh_info.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_parent = pc_parent.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_parent&1)){
                        //Indicates this is a particle cell node
                    
                        pc_parent.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        output.get_val(curr_key) = compute_parent_cell_neigh_mean(node_val_parent,curr_key,parent_data,pc_parent);
                        
                    }
                }
            }
        }
    }
    
    //set the output
    std::swap(output,parent_data);
    
    timer.stop_timer();

}


template<typename T,typename V>
void loop_up_return_vals(std::vector<V>& vals,T curr_key,PartCellParent<T>& pc_parent,ExtraPartCellData<V>& parent_data,const std::vector<unsigned int>& status_offsets){
    //
    //  Loops up structure
    //
    
    V last = parent_data.get_val(curr_key);
    
    T curr_node = pc_parent.parent_info.get_val(curr_key);
    
    unsigned int offset = 0;
    
    //
    unsigned int counter = 0;
    
    for(int i = 0;i < status_offsets.size();i++){
        if(status_offsets[i] == offset){
            vals[i] = parent_data.get_val(curr_key);
            counter ++;
        }
    }
    
    if(counter < 3){
    
        //get the children
        curr_key = pc_parent.get_parent_key(curr_node,curr_key);
        float depth = pc_parent.neigh_info.pc_key_get_depth(curr_key);
        float x = pc_parent.neigh_info.pc_key_get_x(curr_key);
        float z = pc_parent.neigh_info.pc_key_get_z(curr_key);
        float j = pc_parent.neigh_info.pc_key_get_j(curr_key);
        
        
        while ((curr_key > 0) & (counter < 3)){
            offset++;
            
            last = parent_data.get_val(curr_key);
            
            for(int i = 0;i < status_offsets.size();i++){
                if(status_offsets[i] == offset){
                    vals[i] = last;
                    counter ++;
                    
                }
            }
            curr_node = pc_parent.parent_info.get_val(curr_key);
            curr_key = pc_parent.get_parent_key(curr_node,curr_key);
        }
        
        if(counter <3){
            for(int i = 0;i < status_offsets.size();i++){
                if(vals[i] == 0){
                    vals[i] = last;
                }
            }
        }
    
    }
    
    if(last == 0){
        int stop = 1;
    }

    
    
}


template<typename U,typename T,typename V>
void get_value_up_tree_offset(PartCellStructure<U,T>& pc_struct,PartCellParent<T>& pc_parent,ExtraPartCellData<V>& parent_data,ExtraPartCellData<V>& partcell_data,const std::vector<unsigned int> status_offsets,bool min_max){
    //
    //  Bevan Cheeseman 2016
    //
    //  Loops up through a structure and gets value in the parent structure that are a certain offset from a particular cell based of the status of the cell
    //  min_max flag, 1 for max, 0 for min
    //
    
    //initialize
    partcell_data.initialize_structure_cells(pc_struct.pc_data);
    
    ////////////////////////////
    //
    // Parent Loop
    //
    ////////////////////////////
    
    Part_timer timer;
    timer.verbose_flag = true;
    
    uint64_t x_;
    uint64_t j_;
    uint64_t z_;
    uint64_t curr_key;
    uint64_t status;
    
    uint64_t node_val_parent;
    uint64_t node_val_part;
    std::vector<uint64_t> children_keys;
    std::vector<uint64_t> children_ind;
    
    children_keys.resize(8,0);
    children_ind.resize(8,0);
    
    timer.start_timer("Push down tree");
    
    //reverse loop direction
    for(uint64_t i = pc_parent.neigh_info.depth_max;i >= pc_parent.neigh_info.depth_min;i--){
        //loop over the resolutions of the structure
        const unsigned int x_num_ =  pc_parent.neigh_info.x_num[i];
        const unsigned int z_num_ =  pc_parent.neigh_info.z_num[i];
        
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,node_val_part) firstprivate(children_keys,children_ind) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_parent.neigh_info.pc_key_set_z(curr_key,z_);
            pc_parent.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_parent.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_parent.neigh_info.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_parent = pc_parent.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_parent&1)){
                        //Indicates this is a particle cell node
                        
                        pc_parent.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        status = pc_parent.neigh_info.get_status(node_val_parent);
                        
                        //parent has real siblings
                        if(status > 0){
                            
                            //first loop up the structure and get the required values
                            std::vector<V> vals;
                            vals.resize(3,0);
                            
                            loop_up_return_vals(vals,curr_key,pc_parent,parent_data,status_offsets);
                            
                            pc_parent.get_children_keys(curr_key,children_keys,children_ind);
                            
                            T part_status = 0;
                            
                            for(int c = 0;c < children_keys.size();c++){
                                uint64_t child = children_keys[c];
                                
                                if(child > 0){
                                    
                                    if(children_ind[c] == 1){
                                        // get the childs status and then give it the correct value
                                        node_val_part = pc_struct.pc_data.get_val(child);
                                        part_status = pc_struct.pc_data.get_status(node_val_part);
                                        partcell_data.get_val(child) = vals[part_status-1];
                                        
                                        
                                    }
                                    
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    timer.stop_timer();
    
    uint64_t i = pc_struct.depth_min;
    
    const unsigned int x_num_ =  pc_struct.pc_data.x_num[i];
    const unsigned int z_num_ =  pc_struct.pc_data.z_num[i];
    
    V min_val = 0;
    V max_val = 64000;
    
    //Need to account for those areas that are on the lowest resolution and therefore have no min or max
    for(z_ = 0;z_ < z_num_;z_++){
        //both z and x are explicitly accessed in the structure
        curr_key = 0;
        
        pc_struct.pc_data.pc_key_set_z(curr_key,z_);
        pc_struct.pc_data.pc_key_set_depth(curr_key,i);
        
        for(x_ = 0;x_ < x_num_;x_++){
            
            pc_struct.pc_data.pc_key_set_x(curr_key,x_);
            
            const size_t offset_pc_data = x_num_*z_ + x_;
            
            const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
            
            //the y direction loop however is sparse, and must be accessed accordinagly
            for(j_ = 0;j_ < j_num;j_++){
                
                //particle cell node value, used here as it is requried for getting the particle neighbours
                node_val_parent = pc_struct.pc_data.data[i][offset_pc_data][j_];
                
                if (!(node_val_parent&1)){
                    //Indicates this is a particle cell node
                    
                    pc_struct.pc_data.pc_key_set_j(curr_key,j_);
                    
                    if (min_max){
                        partcell_data.get_val(curr_key) = max_val;
                    } else {
                        partcell_data.get_val(curr_key) = min_val;
                    }
                }
            }
        }
    }



}

template<typename U,typename T,typename V>
void get_adaptive_min_max(PartCellStructure<U,T>& pc_struct,ExtraPartCellData<V>& partcell_min,ExtraPartCellData<V>& partcell_max,const std::vector<unsigned int> status_offset){
    //
    //  Bevan Cheeseman 2016
    //
    //  Computes a locally adapted min and max using the tree structure of the representaion and using resolution offsets set in status offset= {seed offset, boundary offset, filler offset}
    //
    
    PartCellParent<uint64_t> pc_parent(pc_struct);
    
    ExtraPartCellData<float> min_data;
    ExtraPartCellData<float> max_data;
    
    calc_cell_min_max<float,uint64_t,float>(pc_struct,pc_parent,pc_struct.part_data.particle_data,min_data,max_data);
    
    //need to do the smoothing loop (min and max)
    smooth_parent_result(pc_parent,min_data);
    smooth_parent_result(pc_parent,max_data);
    
    //get the value according to the status_offsets
    get_value_up_tree_offset(pc_struct,pc_parent,min_data,partcell_min,status_offset,0);
    get_value_up_tree_offset(pc_struct,pc_parent,max_data,partcell_max,status_offset,1);
    
    
    
    
}




#endif