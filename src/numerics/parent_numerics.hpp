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

#ifndef _compression_h
#define _compression_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"


template <typename T,typename S,typename U>
void calc_cell_min_max(PartCellStructure<T,S>& pc_struct,ExtraPartCellData<U>& particle_data,ExtraPartCellData<U>& min_data,ExtraPartCellData<U>& max_data){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Computes min and max up tree (min and max will be of parent cell size)
    //
    //  Input: pc_struct and particle data, (typical case just input the intensities)
    //


    
    
    
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