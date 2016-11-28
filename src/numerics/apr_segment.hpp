////////////////////////
//
//  Bevan Cheeseman 2016
//
/////////////////////////

#ifndef _segment_rc_h
#define _segment_rc_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"


int uf_find(int x,std::vector<uint16_t>& labels) {
    int y = x;
    while (labels[y] != y)
        y = labels[y];
    
    while (labels[x] != x) {
        int z = labels[x];
        labels[x] = y;
        x = z;
    }
    return y;
}

/*  uf_union joins two equivalence classes and returns the canonical label of the resulting class. */

int uf_union(int x, int y,std::vector<uint16_t>& labels) {
    return labels[uf_find(x,labels)] = uf_find(y,labels);
}

/*  uf_make_set creates a new equivalence class and returns its label */

int uf_make_set(std::vector<uint16_t>& labels) {
    labels[0] ++;
    labels.push_back(labels[0]);
    //labels[labels[0]] = labels[0];
    return labels[0];
}



template<typename S>
void calc_connected_component(PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<uint8_t>& binary_mask,ExtraPartCellData<uint16_t>& component_label){
    //
    //  Calculate connected component from a binary mask
    //
    
    component_label.initialize_structure_parts(pc_struct.part_data.particle_data);
    
    Part_timer timer;
    
    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_part_keys; // data structure for holding particle or cell neighbours
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //
    
    uint64_t status=0;
    uint64_t part_offset=0;
    uint64_t p;
    
    std::vector<uint16_t> labels;
    
    timer.verbose_flag = true;
    
    timer.start_timer("connected comp first loop");
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset) firstprivate(neigh_part_keys,neigh_cell_keys) if(z_num_*x_num_ > 100)
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
                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            
                            if( binary_mask.get_part(curr_key) ==1){
                                
                                uint64_t neigh_part;
                                std::vector<uint16_t> neigh_labels;
                                
                                //first get -y neighbour
                                pc_struct.part_data.get_part_neighs_face(1,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                
                                for(int n = 0; n < neigh_part_keys.neigh_face[0].size();n++){
                                    neigh_part = neigh_part_keys.neigh_face[0][n];
                                    
                                    if(neigh_part > 0){
                                        if(component_label.get_part(neigh_part) > 0){
                                            //do something
                                            neigh_labels.push_back(component_label.get_part(neigh_part));
                                        }
                                    }
                                    
                                }
                                
                                //first get -y neighbour
                                pc_struct.part_data.get_part_neighs_face(3,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                
                                for(int n = 0; n < neigh_part_keys.neigh_face[3].size();n++){
                                    neigh_part = neigh_part_keys.neigh_face[3][n];
                                    
                                    if(neigh_part > 0){
                                        if(component_label.get_part(neigh_part) > 0){
                                            //do something
                                            neigh_labels.push_back(component_label.get_part(neigh_part));
                                        }
                                    }
                                    
                                }
                                
                                //first get -y neighbour
                                pc_struct.part_data.get_part_neighs_face(5,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                
                                for(int n = 0; n < neigh_part_keys.neigh_face[5].size();n++){
                                    neigh_part = neigh_part_keys.neigh_face[5][n];
                                    
                                    if(neigh_part > 0){
                                        if(component_label.get_part(neigh_part) > 0){
                                            //do something
                                            neigh_labels.push_back(component_label.get_part(neigh_part));
                                        }
                                    }
                                    
                                }
                                
                                if(neigh_labels.size() == 0){
                                    //
                                    // no neighbour labels new region
                                    //
                                    
                                    component_label.get_part(curr_key) = uf_make_set();
                                    
                                    
                                } else if (neigh_labels.size() = 1){
                                    //
                                    // one neighbour label, set yourslef to that label
                                    //
                                    
                                    component_label.get_part(curr_key) = neigh_labels[0];
                                    
                                    
                                } else {
                                    //
                                    // multiple neighbour regions, resolve
                                    //
                                    
                                    for(int n = 0; n < (neigh_labels.size()-1);n++){
                                        
                                        
                                        
                                    }
                                    
                                    
                                }
                                
                                
                                
                                
                                
                            
                            }
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    
    timer.stop_timer();

    
    
    
    
    
    
    
    
}








#endif