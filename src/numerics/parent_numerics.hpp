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
        
        
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,part_offset,node_val_part) firstprivate(children_keys,children_ind) if(z_num_*x_num_ > 100)
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
                        if(status == 2){
                            
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
    
    for(uint64_t face = 0; face < neigh_keys.neigh_face.size();face++){
        
        for(uint64_t n = 0; n < neigh_keys.neigh_face[face].size();n++){
            uint64_t neigh_key = neigh_keys.neigh_face[face][n];
            
            if(neigh_key > 0){
                temp+= parent_data.get_val(neigh_key);
                counter++;
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
        
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,node_val_part)  if(z_num_*x_num_ > 100)
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
void loop_up_return_vals(std::vector<V>& vals,T curr_key,T curr_node,PartCellParent<T>& pc_parent,ExtraPartCellData<V>& parent_data,const std::vector<unsigned int>& status_offsets){
    //
    //  Loops up structure
    //
    
    V last = parent_data.get_val(curr_key);
    
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
    
        while ((curr_key > 0) & (counter < 3)){
            offset++;
            
            for(int i = 0;i < status_offsets.size();i++){
                if(status_offsets[i] == offset){
                    vals[i] = parent_data.get_val(curr_key);
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
    
    
}


template<typename U,typename T,typename V>
void get_value_up_tree_offset(PartCellStructure<U,T>& pc_struct,PartCellParent<T>& pc_parent,ExtraPartCellData<V>& parent_data,ExtraPartCellData<V>& partcell_data,const std::vector<unsigned int> status_offsets){
    //
    //  Bevan Cheeseman 2016
    //
    //  Loops up through a structure and gets value in the parent structure that are a certain offset from a particular cell based of the status of the cell
    //
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
        
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,node_val_part) firstprivate(children_keys,children_ind) if(z_num_*x_num_ > 100)
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
                        if(status == 2){
                            
                            //first loop up the structure and get the required values
                            std::vector<V> vals;
                            vals.resize(3,0);
                            
                            loop_up_return_vals(vals,curr_key,node_val_parent,pc_parent,parent_data,status_offsets);
                            
                            pc_parent.get_children_keys(curr_key,children_keys,children_ind);
                            
                            T part_status = 0;
                            
                            for(int c = 0;c < children_keys.size();c++){
                                uint64_t child = children_keys[c];
                                
                                if(child > 0){
                                    
                                    if(children_ind[c] == 1){
                                        // get the childs status and then give it the correct value
                                        node_val_part = pc_struct.pc_data.get_val(child);
                                        part_status = pc_struct.pc_data.get_status(node_val_part);
                                        partcell_data.get_val(child) = vals[part_status];
                                        
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
    get_value_up_tree_offset(pc_struct,pc_parent,min_data,partcell_min,status_offset);
    get_value_up_tree_offset(pc_struct,pc_parent,max_data,partcell_max,status_offset);
    
    
}


//template <typename T,typename S>
//void calc_cell_min_max(Part_rep& p_rep,Part_data<T>& part_level_data,std::vector<S>& min_data,std::vector<S>& max_data){
//    //
//    //  Bevan Cheeseman 2016
//    //
//    //  calculates min_max of the particle cell structure.
//    //
//    //
//    
//    
//    fill_particle_cell_tree(p_rep);
//    
//    float temp_max;
//    float temp_min;
//    unsigned int curr_index;
//    
//    
//    min_data.resize(p_rep.get_cell_num(),64000);
//    max_data.resize(p_rep.get_cell_num(),0);
//    
//    Cell_id parent_cell;
//    
//    
//    //loop over all levels of k
//    for (int k_ = p_rep.pl_map.k_max; k_ > 0; k_--) {
//        for(auto cell_ref : p_rep.pl_map.pl[k_]) {
//            
//            curr_index = cell_ref.second;
//            
//            if (p_rep.status.data[curr_index] > 0) {
//                
//                temp_min = 0;
//                temp_max = 0;
//                
//                float count = 0;
//                //average over the current area for the first step then use min max
//                
//                for (int j = p_rep.pl_map.cell_indices[curr_index].first; j < p_rep.pl_map.cell_indices[curr_index].last; j++) {
//                    
//                    count++;
//                    
//                    temp_min = temp_min + part_level_data.data[j];
//                    
//                    //temp_min = std::min((float)part_level_data.data[j],temp_min);
//                    //temp_max = std::max((float)part_level_data.data[j],temp_max);
//                    
//                }
//                
//                min_data[curr_index] = temp_min/count;
//                max_data[curr_index] = temp_min/count;
//                
//            }
//            
//            
//            
//            get_parent(p_rep.pl_map.cells[curr_index], parent_cell);
//            
//            auto parent_cell_ref = p_rep.pl_map.pl[parent_cell.k].find(parent_cell);
//            
//            if (parent_cell_ref != p_rep.pl_map.pl[parent_cell.k].end()){
//                
//                min_data[parent_cell_ref->second] = std::min(min_data[curr_index],min_data[parent_cell_ref->second]);
//                
//                max_data[parent_cell_ref->second] = std::max(max_data[curr_index],max_data[parent_cell_ref->second]);
//                
//            }
//            
//        }
//    }
//    
//    
//}
//template <typename T>
//T compute_over_neigh(Part_rep& p_rep,unsigned int curr_id,int type,std::vector<T>& cell_vec){
//    //
//    //  Bevan Cheeseman 2016
//    //
//    //  Compute some operation over neighborhood
//    //
//    //
//    
//    Cell_id curr_cell = p_rep.pl_map.cells[curr_id];
//    std::vector<Cell_id> neigh_cell;
//    int neigh_type = 0;
//    
//    get_neighbours(curr_cell,neigh_cell,neigh_type,p_rep.dim);
//    
//    T temp;
//    
//    temp = cell_vec[curr_id];
//    
//    float count = 1;
//    
//    //neigh_cell.resize(0);
//    
//    for(int i = 0; i < neigh_cell.size();i++){
//        
//        auto neigh_cell_ref = p_rep.pl_map.pl[neigh_cell[i].k].find(neigh_cell[i]);
//        
//        if (neigh_cell_ref != p_rep.pl_map.pl[neigh_cell[i].k].end()){
//            
//            count++;
//            
//            int neigh_id = neigh_cell_ref->second;
//            
//            if (type == 0){
//                temp = std::max(temp,cell_vec[neigh_id]);
//            } else if (type == 1){
//                temp = std::min(temp,cell_vec[neigh_id]);
//            } else if(type == 3){
//                temp += cell_vec[neigh_id];
//            }
//            
//            
//        }
//    }
//    
//    temp = temp/count;
//    
//    return temp;
//    
//}
//template <typename T>
//void go_down_tree(Part_rep& p_rep,unsigned int curr_id,std::vector<T>& min_data,std::vector<T>& push_min_data,int k_diff,std::vector<T>& temp_vec,int type){
//    //
//    //  Bevan Cheeseman 2016
//    //
//    //  Recusively go down the branches of the tree
//    //
//    
//    
//    Cell_id curr_cell;
//    std::vector<Cell_id> child_cells;
//    
//    curr_cell = p_rep.pl_map.cells[curr_id];
//    
//    get_children(curr_cell,child_cells,p_rep.dim);
//    
//    for(int i = 0;i < child_cells.size();i++){
//        auto child_cell_ref = p_rep.pl_map.pl[child_cells[i].k].find(child_cells[i]);
//        
//        if (child_cell_ref != p_rep.pl_map.pl[child_cells[i].k].end()){
//            
//            unsigned int child_id = child_cell_ref->second;
//            
//            if(p_rep.status.data[child_id] == 0){
//                
//                //need to add the neighborhood search and the operation over them all here
//                
//                //temp_vec[child_cells[i].k] = min_data[child_id];
//                temp_vec[child_cells[i].k] = compute_over_neigh(p_rep,curr_id,type,min_data);
//                go_down_tree(p_rep,child_id,min_data,push_min_data,k_diff,temp_vec,type);
//                
//            } else {
//                
//                float temp = temp_vec[std::max(p_rep.pl_map.k_min,child_cells[i].k - k_diff)];
//                
//                
//                if (p_rep.status.data[child_id] == 2) {
//                    push_min_data[child_id] = temp_vec[std::max(p_rep.pl_map.k_min,child_cells[i].k - k_diff)];
//                } else if (p_rep.status.data[child_id] ==4) {
//                    
//                    push_min_data[child_id] = temp_vec[std::max(p_rep.pl_map.k_min,child_cells[i].k - k_diff-1)];
//                } else {
//                    push_min_data[child_id] = temp_vec[std::max(p_rep.pl_map.k_min,child_cells[i].k - k_diff-2)];
//                }
//            }
//            
//            
//        }
//    }
//    
//}
//template <typename T>
//void push_down_tree(Part_rep& p_rep,std::vector<T>& min_data,std::vector<T>& push_min_data,int k_diff,int type){
//    //
//    //  Bevan Cheeseman 2016
//    //
//    //  Pushes a variable down the tree by k_diff
//    //
//    //  Tree must be filled
//    //
//    
//    std::vector<T> temp_vec;
//    
//    push_min_data.resize(min_data.size());
//    
//    temp_vec.resize(p_rep.pl_map.k_max);
//    
//    int curr_k;
//    
//    for(auto cell_ref : p_rep.pl_map.pl[p_rep.pl_map.k_min]) {
//        
//        curr_k = p_rep.pl_map.k_min;
//        
//        Cell_id curr_cell = cell_ref.first;
//        std::vector<Cell_id> child_cells;
//        
//        int curr_cell_id = cell_ref.second;
//        
//        temp_vec[curr_k] = compute_over_neigh(p_rep,curr_cell_id,type,min_data);
//        push_min_data[curr_cell_id] = compute_over_neigh(p_rep,curr_cell_id,type,min_data);
//        
//        get_children(curr_cell,child_cells,p_rep.dim);
//        
//        for(int i = 0;i < child_cells.size();i++){
//            auto child_cell_ref = p_rep.pl_map.pl[child_cells[i].k].find(child_cells[i]);
//            
//            if (child_cell_ref != p_rep.pl_map.pl[child_cells[i].k].end()){
//                
//                unsigned int child_id = child_cell_ref->second;
//                
//                if(p_rep.status.data[child_id] == 0){
//                    compute_over_neigh(p_rep,child_id,type,min_data);
//                    //temp_vec[child_cells[i].k] = min_data[child_id];
//                    temp_vec[child_cells[i].k] = compute_over_neigh(p_rep,child_id,type,min_data);
//                    
//                    go_down_tree(p_rep,child_id,min_data,push_min_data,k_diff,temp_vec,type);
//                } else {
//                    
//                    push_min_data[child_id] = compute_over_neigh(p_rep,curr_cell_id,type,min_data);
//                    
//                }
//                
//                
//            }
//        }
//        
//    }
//    
//    
//    
//    
//}

#endif