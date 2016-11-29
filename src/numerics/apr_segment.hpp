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

#include "filter_help/FilterOffset.hpp"
#include "filter_help/FilterLevel.hpp"

#include "filter_help/CurrLevel.hpp"
#include "filter_help/NeighOffset.hpp"

int uf_find(int x,std::vector<int>& labels) {
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

int uf_union(int x, int y,std::vector<int>& labels) {
    return labels[uf_find(x,labels)] = uf_find(y,labels);
}

/*  uf_make_set creates a new equivalence class and returns its label */

int uf_make_set(std::vector<int>& labels) {
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
    //  Should be written with the neighbour iterators instead.
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
    
    std::vector<int> labels;
    labels.resize(1,0);
    
    std::vector<int> neigh_labels;
    neigh_labels.reserve(10);
    
    neigh_part_keys.neigh_face[1].reserve(4);
    neigh_part_keys.neigh_face[3].reserve(4);
    neigh_part_keys.neigh_face[5].reserve(4);
    
    timer.verbose_flag = true;
    
    timer.start_timer("connected comp first loop");
    
    for(uint64_t i = pc_struct.pc_data.depth_max;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset) firstprivate(neigh_part_keys,neigh_cell_keys,neigh_labels) if(z_num_*x_num_ > 100)
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
                                
                                neigh_labels.resize(0);
                                //first get -y neighbour
                                pc_struct.part_data.get_part_neighs_face(1,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                
                                //first get -x neighbour
                                pc_struct.part_data.get_part_neighs_face(3,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                
                                //first get -z neighbour
                                pc_struct.part_data.get_part_neighs_face(5,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                
                                for(int n = 0; n < neigh_part_keys.neigh_face[1].size();n++){
                                    neigh_part = neigh_part_keys.neigh_face[1][n];
                                    
                                    if(neigh_part > 0){
                                        if(component_label.get_part(neigh_part) > 0){
                                            //do something
                                            neigh_labels.push_back(component_label.get_part(neigh_part));
                                        }
                                    }
                                    
                                }
                                
                                for(int n = 0; n < neigh_part_keys.neigh_face[3].size();n++){
                                    neigh_part = neigh_part_keys.neigh_face[3][n];
                                    
                                    if(neigh_part > 0){
                                        if(component_label.get_part(neigh_part) > 0){
                                            //do something
                                            neigh_labels.push_back(component_label.get_part(neigh_part));
                                        }
                                    }
                                    
                                }
                                
                                
                                
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
                                    
                                    component_label.get_part(curr_key) = uf_make_set(labels);
                                    
                                    
                                } else if (neigh_labels.size() == 1){
                                    //
                                    // one neighbour label, set yourslef to that label
                                    //
                                    
                                    component_label.get_part(curr_key) = neigh_labels[0];
                                    
                                    
                                } else {
                                    //
                                    // multiple neighbour regions, resolve
                                    //
                                    
                                    uint16_t curr_label = neigh_labels[0];
                                    
                                    //resolve labels
                                    for(int n = 0; n < (neigh_labels.size()-1);n++){
                                        curr_label = uf_union(curr_label,neigh_labels[n+1],labels);
                        
                                    }
                                    
                                    component_label.get_part(curr_key) = curr_label;
                                    
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


    std::vector<int> new_labels;
    new_labels.resize(labels.size(),0);
    
    timer.start_timer("connected comp second loop");

    
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
                            
                            if( component_label.get_part(curr_key) > 0){
                                
                                int x = uf_find(component_label.get_part(curr_key),labels);
                                if (new_labels[x] == 0) {
                                    new_labels[0]++;
                                    new_labels[x] = new_labels[0];
                                }
                                
                                component_label.get_part(curr_key) = new_labels[x];
                                
                            }
                            
                        }
                    }
                }
            }
        }
    }
    
    
    timer.stop_timer();
    
    
    
}

template<typename S>
void calc_connected_component_alt(PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<uint8_t>& binary_mask,ExtraPartCellData<uint16_t>& component_label){
    //
    //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct.pc_data);
    part_new.transfer_intensities(pc_struct.part_data);
    
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
    
    std::vector<int> labels;
    labels.resize(1,0);
    
    std::vector<int> neigh_labels;
    neigh_labels.reserve(10);
    
    float neigh;
    
    timer.verbose_flag = false;
    
    timer.start_timer("iterate parts");
    
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
    
    
    for(uint64_t depth = (pc_struct.pc_data.depth_min + 1);depth <= pc_struct.pc_data.depth_max;depth++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
        
        CurrentLevel<float,uint64_t> curr_level;
        curr_level.set_new_depth(depth,part_new);
        
        NeighOffset<float,uint64_t> neigh_y;
        neigh_y.set_new_depth(depth,part_new);
        neigh_y.set_offsets(0,0,1,0);
        
        NeighOffset<float,uint64_t> neigh_z;
        neigh_z.set_new_depth(depth,part_new);
        neigh_z.set_offsets(0,1,0,0);
        
        NeighOffset<float,uint64_t> neigh_x;
        neigh_x.set_new_depth(depth,part_new);
        neigh_x.set_offsets(1,0,0,0);
        
#pragma omp parallel for default(shared) private(p,z_,x_,j_,neigh) firstprivate(curr_level,neigh_x,neigh_z,neigh_y) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                for(uint64_t l = 0;l < 5;l++){
                    
                    curr_level.set_new_xz(x_,z_,l,part_new);
                    neigh_x.reset_j(curr_level,part_new);
                    neigh_z.reset_j(curr_level,part_new);
                    neigh_y.reset_j(curr_level,part_new);
                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < curr_level.j_num;j_++){
                        
                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        bool iscell = curr_level.new_j(j_,part_new);
                        
                        if (iscell){
                            //Indicates this is a particle cell node
                            curr_level.update_cell(part_new);
                            
                            neigh_x.incriment_y_same_depth(curr_level,part_new);
                            neigh_z.incriment_y_same_depth(curr_level,part_new);
                            neigh_y.incriment_y_same_depth(curr_level,part_new);
                            
                            


                            
                            if(curr_level.status==SEED){
                                //seed loop (first particle)
                                
                                if(l > 0){
                                    
                                    
                                    neigh =  neigh_x.get_part(part_new.particle_data);
                                    neigh +=  neigh_z.get_part(part_new.particle_data);
                                    neigh +=  neigh_y.get_part(part_new.particle_data);
                                    curr_level.get_part(part_new) = neigh;
                                
                                    curr_level.iterate_y_seed();
                                    //second particle
                                    neigh_x.incriment_y_part_same_depth(curr_level,part_new);
                                    neigh_z.incriment_y_part_same_depth(curr_level,part_new);
                                    neigh_y.incriment_y_part_same_depth(curr_level,part_new);
                                    
                                    neigh = neigh_x.get_part(part_new.particle_data);
                                    neigh += neigh_z.get_part(part_new.particle_data);
                                    neigh += neigh_y.get_part(part_new.particle_data);
                                    curr_level.get_part(part_new) = neigh;
                                    
                                    
                                } else {
                                    
                                    neigh_x.incriment_y_parent_depth(curr_level,part_new);
                                    neigh_z.incriment_y_parent_depth(curr_level,part_new);
                                    neigh_y.incriment_y_parent_depth(curr_level,part_new);
                                }
                                
                            } else {
                                //non seed loop
                                if( l == 0){
                                    
                                    neigh_x.incriment_y_parent_depth(curr_level,part_new);
                                    neigh_z.incriment_y_parent_depth(curr_level,part_new);
                                    neigh_y.incriment_y_parent_depth(curr_level,part_new);
                                    
                                    neigh = neigh_x.get_part(part_new.particle_data);
                                    neigh += neigh_z.get_part(part_new.particle_data);
                                    neigh += neigh_y.get_part(part_new.particle_data);
                                   
                                    curr_level.get_part(part_new) = neigh;
                                }
                            }
                            
                            
                        } else {
                            // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                            
                            curr_level.update_gap();
                            
                        }
                        
                        
                    }
                    
                }
            }
        }
    }
    
    }
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Neigh Regime New took: " << time << std::endl;
    
    timer.start_timer("iterate parts old");
    
    for(int r = 0;r < num_repeats;r++){
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        #pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset)  firstprivate(neigh_part_keys,neigh_cell_keys) if(z_num_*x_num_ > 100)
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
                            
                            pc_struct.part_data.get_part_neighs_face(2,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                            pc_struct.part_data.get_part_neighs_face(0,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                            pc_struct.part_data.get_part_neighs_face(4,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                            
                            float temp = 0;
                            
                            for(int n = 0; n < neigh_part_keys.neigh_face[2].size();n++){
                                uint64_t neigh_part = neigh_part_keys.neigh_face[2][n];
                                
                                
                                if(neigh_part > 0){
                                    
                                        //do something
                                    temp += pc_struct.part_data.particle_data.get_part(neigh_part);
                                    
                                }
                                
                            }
                            
                            for(int n = 0; n < neigh_part_keys.neigh_face[0].size();n++){
                                uint64_t neigh_part = neigh_part_keys.neigh_face[0][n];
                                
                                
                                if(neigh_part > 0){
                                    
                                    //do something
                                    temp += pc_struct.part_data.particle_data.get_part(neigh_part);
                                    
                                }
                                
                            }
                            
                            for(int n = 0; n < neigh_part_keys.neigh_face[4].size();n++){
                                uint64_t neigh_part = neigh_part_keys.neigh_face[4][n];
                                
                                
                                if(neigh_part > 0){
                                    
                                    //do something
                                    temp += pc_struct.part_data.particle_data.get_part(neigh_part);
                                    
                                }
                                
                            }


                            
                            pc_struct.part_data.particle_data.get_part(curr_key) = temp;
                                
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    }
    
    timer.stop_timer();

    time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Neigh Regime Old took: " << time << std::endl;
    
}

//void calc_boundary_parts(PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<uint16_t>& component_label){
//    //
//    //  Calculates boundary part locations
//    //
//    //
//    //
//    
//    
//    
//}






#endif