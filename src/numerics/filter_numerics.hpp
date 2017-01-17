#ifndef _filter_num_h
#define _filter_num_h
//////////////////////////////////////////////////
//
//
//  Bevan Cheeseman 2016
//
//  Filter operations (seperable)
//
//
//////////////////////////////////////////////////

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

#include "../../test/utils.h"

template<typename T>
void iterate_temp_vec(std::vector<T>& temp_vec,std::vector<T>& temp_vec_depth){
    //
    //  Iterates forward these arrays
    //
    //  Copying the last value in
    //
    
    //shuffle values to the left, and then place end value, copied to end. Repeat for both
    std::rotate(temp_vec.begin(),temp_vec.begin() + 1,temp_vec.end());
    
    temp_vec.back() = temp_vec[0];
    
    std::rotate(temp_vec_depth.begin(),temp_vec_depth.begin() + 1,temp_vec_depth.end());
    
    temp_vec_depth.back() = temp_vec_depth[0];
}
template<typename T>
void iterate_temp_vec(std::vector<T>& temp_vec){
    //
    //  Iterates forward these arrays
    //
    //  Copying the last value in
    //
    
    //shuffle values to the left, and then place end value, copied to end. Repeat for both
    std::rotate(temp_vec.begin(),temp_vec.begin() + 1,temp_vec.end());
    
    temp_vec.back() = temp_vec[0];
    
}

template<typename S>
void compute_gradient_new(PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<S>& filter_output){
    //
    //  Calculate Neighbours Using Iterators
    //
    //
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct);
    
    //filter_output.initialize_structure_parts(part_new);
    //
    Part_timer timer;
    
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    
    float int_neigh_p;
    float int_neigh_m;
    
    timer.verbose_flag = false;
    float num_repeats = 50;
    
    timer.start_timer("compute gradient y");
    
    for(int r = 0;r < num_repeats;r++){
        
        
        for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = part_new.access_data.x_num[depth];
            const unsigned int z_num_ = part_new.access_data.z_num[depth];
            
            CurrentLevel<float,uint64_t> curr_level;
            curr_level.set_new_depth(depth,part_new);
            
            NeighOffset<float,uint64_t> neigh_y(0);
            NeighOffset<float,uint64_t> neigh_ym(1);
            
            neigh_y.set_new_depth(curr_level,part_new);
            neigh_ym.set_new_depth(curr_level,part_new);
            
#pragma omp parallel for default(shared) private(z_,x_,j_,int_neigh_p,int_neigh_m) firstprivate(curr_level,neigh_y,neigh_ym) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    curr_level.set_new_xz(x_,z_,part_new);
                    neigh_y.set_new_row(curr_level,part_new);
                    //neigh_ym.set_new_row(curr_level,part_new);
                    
                    for(j_ = 0;j_ < curr_level.j_num;j_++){
                        
                        bool iscell = curr_level.new_j(j_,part_new);
                        
                        if (iscell){
                            //Indicates this is a particle cell node
                            curr_level.update_cell(part_new);
                            
                            neigh_y.iterate(curr_level,part_new);
                            neigh_ym.iterate(curr_level,part_new);
                            
                            int_neigh_p = neigh_y.get_part(part_new.particle_data);
                            int_neigh_m = neigh_ym.get_part(part_new.particle_data);
                            
                            (void) int_neigh_p;
                            (void) int_neigh_m;
                            
                        } else {
                            
                            curr_level.update_gap();
                            
                        }
                        
                        
                    }
                }
            }
        }
    }
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1);
    
    std::cout << " Gradient New took: " << time/num_repeats << std::endl;
    
    
}

template<typename S>
void compute_gradient(PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<S>& filter_output){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    ParticleDataNew<float, uint64_t> part_new;
    
    //part_new.initialize_from_structure(pc_struct.pc_data);
    //part_new.transfer_intensities(pc_struct.part_data);
    
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
    
    
    float num_repeats = 50;
    

    timer.start_timer("compute gradient old");
    
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
                                
                                pc_struct.part_data.get_part_neighs_face(0,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                pc_struct.part_data.get_part_neighs_face(1,p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                            
                                
                                float val_1 = 0;
                                float val_2 = 0;
                                
                                for(int n = 0; n < neigh_part_keys.neigh_face[0].size();n++){
                                    uint64_t neigh_part = neigh_part_keys.neigh_face[0][n];
                                    
                                    
                                    if(neigh_part > 0){
                                        
                                        //do something
                                        val_1 += pc_struct.part_data.particle_data.get_part(neigh_part);
                                        
                                    }
                                    
                                }
                                
                                (void) val_1;
                                
                                if(val_1 > 0){
                                    val_1 = val_1/neigh_part_keys.neigh_face[0].size();
                                }
                                
                                
                                for(int n = 0; n < neigh_part_keys.neigh_face[1].size();n++){
                                    uint64_t neigh_part = neigh_part_keys.neigh_face[1][n];
                                    
                                    
                                    if(neigh_part > 0){
                                        
                                        //do something
                                        val_2 += pc_struct.part_data.particle_data.get_part(neigh_part);
                                        
                                    }
                                    
                                }
                                
                                if(val_2 > 0){
                                    val_2 = val_2/neigh_part_keys.neigh_face[1].size();
                                }
                                
                                (void) val_2;
//
//                                
//                                float grad = (val_1 - val_2)/2;
                                
                                //filter_output.data[i][offset_pc_data][part_offset+p] = abs(grad);
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
    
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << "Compute Gradient Old: " << time << std::endl;
    
}

void neigh_cells(PartCellData<uint64_t>& pc_data){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    
    Part_timer timer;
    
    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information

    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //

    
    timer.verbose_flag = false;
    

    
    const int direction = 0;
    
    float num_repeats = 50;
    
    timer.start_timer("neigh_cell_comp");
    
    for(int r = 0;r < num_repeats;r++){
        

        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                curr_key = 0;
                
                pc_data.pc_key_set_z(curr_key,z_);
                pc_data.pc_key_set_depth(curr_key,i);
                
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    pc_data.pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        
                        
                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        node_val_pc = pc_data.data[i][offset_pc_data][j_];
                        
                        if (!(node_val_pc&1)){
                            //Indicates this is a particle cell node
                            //y_coord++;
                            
                            pc_data.pc_key_set_j(curr_key,j_);
                            
                            pc_data.get_neighs_face(curr_key,node_val_pc,0,neigh_cell_keys);
                            pc_data.get_neighs_face(curr_key,node_val_pc,1,neigh_cell_keys);
                            
                            //loop over the nieghbours
                            for(int n = 0; n < neigh_cell_keys.neigh_face[0].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[0][n];
                                
                                if(neigh_key > 0){
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                    
                                }
                                
                                
                            }
                            
                            //loop over the nieghbours
                            for(int n = 0; n < neigh_cell_keys.neigh_face[1].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[1][n];
                                
                                if(neigh_key > 0){
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                    (void) neigh_key;
                                }
                                
                                
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
    
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << "Get neigh: " << time << std::endl;
    
   
    
}
void neigh_cells_new(PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t> part_new){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    
    Part_timer timer;
    
    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information
    
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //
    
    
    timer.verbose_flag = false;
    
    
    
    const int direction = 0;
    
    float num_repeats = 50;
    
    timer.start_timer("neigh_cell_comp");
    
    for(int r = 0;r < num_repeats;r++){
        
        
        
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                curr_key = 0;
                
                pc_data.pc_key_set_z(curr_key,z_);
                pc_data.pc_key_set_depth(curr_key,i);
                
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    pc_data.pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();
                    
                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        
                        float part_int= 0;
                        
                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        node_val_pc = pc_data.data[i][offset_pc_data][j_];
                        
                        if (!(node_val_pc&1)){
                            //Indicates this is a particle cell node
                            //y_coord++;
                            
                            pc_data.pc_key_set_j(curr_key,j_);
                            
                            pc_data.get_neighs_face(curr_key,node_val_pc,0,neigh_cell_keys);
                            pc_data.get_neighs_face(curr_key,node_val_pc,1,neigh_cell_keys);
                            
                            //loop over the nieghbours
                            for(int n = 0; n < neigh_cell_keys.neigh_face[0].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[0][n];
                                
                                if(neigh_key > 0){
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                    uint64_t part_node = part_new.access_data.get_val(neigh_key);
                                    uint64_t part_offset = part_new.access_node_get_part_offset(part_node);
                                    part_new.access_data.pc_key_set_index(neigh_key,part_offset);
                                    part_int += part_new.particle_data.get_part(neigh_key);
                                    
                                }
                                
                                
                            }
                            
                            //loop over the nieghbours
                            for(int n = 0; n < neigh_cell_keys.neigh_face[1].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[1][n];
                                
                                if(neigh_key > 0){
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                    uint64_t part_node = part_new.access_data.get_val(neigh_key);
                                    uint64_t part_offset = part_new.access_node_get_part_offset(part_node);
                                    part_new.access_data.pc_key_set_index(neigh_key,part_offset);
                                    part_int += part_new.particle_data.get_part(neigh_key);
                                    
                                    
                                }
                                
                                
                            }
                            
                            (void) part_int;
                            
                            
                        } else {
                            // Inidicates this is not a particle cell node, and is a gap node
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
    }
    
    
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << "Get neigh: " << time << std::endl;
    
    
    
}

void neigh_cells_new_random(PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t> part_new,float num_repeats){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    
    Part_timer timer;
    
    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information
    

    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //
    
    
    
    size_t j_num;
    timer.verbose_flag = false;
    
    
    
    const int direction = 0;
    
    timer.start_timer("neigh_cell_comp");
    
    
    int x_,z_,j_,depth;
    size_t offset_pc_data;
    
    for(int r = 0;r < num_repeats;r++){
        
        depth = std::rand()%(pc_data.depth_max-pc_data.depth_min) + pc_data.depth_min;
        
        int x_num_ = pc_data.x_num[depth];
        int z_num_ = pc_data.z_num[depth];
        
        x_ = std::rand()%x_num_;
        z_ = std::rand()%z_num_;
        
        offset_pc_data = x_num_*z_ + x_;

        j_num = pc_data.data[depth][offset_pc_data].size();
        
        j_ = std::rand()%j_num;
        
        pc_data.pc_key_set_x(curr_key,x_);
        
       
        //both z and x are explicitly accessed in the structure
        curr_key = 0;
        
        pc_data.pc_key_set_z(curr_key,z_);
        pc_data.pc_key_set_depth(curr_key,depth);
        
        
        float part_int= 0;
        
        //particle cell node value, used here as it is requried for getting the particle neighbours
        node_val_pc = pc_data.data[depth][offset_pc_data][j_];
        
        if (!(node_val_pc&1)){
            //Indicates this is a particle cell node
            //y_coord++;
            
            pc_data.pc_key_set_j(curr_key,j_);
            
            pc_data.get_neighs_face(curr_key,node_val_pc,0,neigh_cell_keys);
            pc_data.get_neighs_face(curr_key,node_val_pc,1,neigh_cell_keys);
            
            //loop over the nieghbours
            for(int n = 0; n < neigh_cell_keys.neigh_face[0].size();n++){
                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                uint64_t neigh_key = neigh_cell_keys.neigh_face[0][n];
                
                if(neigh_key > 0){
                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                    uint64_t part_node = part_new.access_data.get_val(neigh_key);
                    uint64_t part_offset = part_new.access_node_get_part_offset(part_node);
                    part_new.access_data.pc_key_set_index(neigh_key,part_offset);
                    part_int += part_new.particle_data.get_part(neigh_key);
                    
                }
                
                
            }
            
            //loop over the nieghbours
            for(int n = 0; n < neigh_cell_keys.neigh_face[1].size();n++){
                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                uint64_t neigh_key = neigh_cell_keys.neigh_face[1][n];
                
                if(neigh_key > 0){
                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                    uint64_t part_node = part_new.access_data.get_val(neigh_key);
                    uint64_t part_offset = part_new.access_node_get_part_offset(part_node);
                    part_new.access_data.pc_key_set_index(neigh_key,part_offset);
                    part_int += part_new.particle_data.get_part(neigh_key);
                    
                }
                
                
            }
            
            (void) part_int;
        } else {
            r--;
        }
        
            
            
    }



    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1);
    
    timer.start_timer("neigh_cell_comp");
    
    
    for(int r = 0;r < num_repeats;r++){
        
        depth = std::rand()%(pc_data.depth_max-pc_data.depth_min) + pc_data.depth_min;
        
        int x_num_ = pc_data.x_num[depth];
        int z_num_ = pc_data.z_num[depth];
        
        x_ = std::rand()%x_num_;
        z_ = std::rand()%z_num_;
        
        offset_pc_data = x_num_*z_ + x_;
        
        j_num = pc_data.data[depth][offset_pc_data].size();
        
        j_ = std::rand()%j_num;
        
        pc_data.pc_key_set_x(curr_key,x_);
        
        
        //both z and x are explicitly accessed in the structure
        curr_key = 0;
        
        pc_data.pc_key_set_z(curr_key,z_);
        pc_data.pc_key_set_depth(curr_key,depth);
        
        
        float part_int= 0;
        
        //particle cell node value, used here as it is requried for getting the particle neighbours
        node_val_pc = pc_data.data[depth][offset_pc_data][j_];
        
        (void) node_val_pc;
        
    }
    
    
    
    timer.stop_timer();
    
    float time2 = (timer.t2 - timer.t1);
    
    std::cout << "Get neigh r : " << (time - time2) << std::endl;

    std::cout << "Get neigh:  " << (time2) << std::endl;
    
    std::cout << "Get neigh: " << (time) << std::endl;

}


template<typename U>
void convolution_filter_pixels(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 1;
    filter.resize(filter_offset*2 +1,1);
    
    std::vector<U> temp_vec;
    temp_vec.resize(filter.size());
    
    uint64_t offset_;
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,offset_) firstprivate(temp_vec)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                
                for(k = 0;k < y_num;k++){
                    
                    std::rotate(temp_vec.begin(),temp_vec.begin() + 1,temp_vec.end());
                    
                    offset_ = std::min(k + filter_offset,y_num);
                    
                    temp_vec.back() = input_data.mesh[j*x_num*y_num + i*y_num + k + offset_];
                    
                    for(uint64_t f = 0;f < filter.size();f++){
                        
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[f]*filter[f];
                    }
                    
                }
            }
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}
template<typename U>
void convolution_filter_pixels_temp(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 8;
    filter.resize(filter_offset*2 +1,1);
    
    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max) firstprivate(temp_vec)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                
                for(k = 0;k < y_num;k++){
                    
                    offset_max = std::min((uint64_t)(k + filter_offset),(uint64_t)(y_num-1));
                    offset_min = std::max((uint64_t)(k - filter_offset),(uint64_t)0);
                    
                    uint64_t f = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){
                        
                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + i*y_num + c]*filter[f];
                        f++;
                    }
                    
                }
            }
        }
        
    }
    
    (void) output_data.mesh;
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}
template<typename U>
void convolution_filter_pixels_off(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 1;
    filter.resize(filter_offset*2 +1,1);
    
    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max) firstprivate(temp_vec)
        for(i = 0; i < x_num;i++){
            for(k = 0;k < y_num;k++){
            
                for(j = 0; j < z_num;j++){
                
                
                    
                    offset_max = std::min((uint64_t)(k + filter_offset),(uint64_t)(y_num-1));
                    offset_min = std::max((uint64_t)(k - filter_offset),(uint64_t)0);
                    
                    uint64_t f = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){
                        
                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + i*y_num + c]*filter[f];
                        f++;
                    }
                    
                }
            }
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Pixel Filter Size Off: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}
template<typename U>
void convolution_filter_pixels_random(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 1;
    filter.resize(filter_offset*2 +1,1);
    
    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = input_data.mesh.size();
    
    for(int r = 0;r < num_repeats;r++){
        
        i = std::rand()%x_num;
        j = std::rand()%z_num;
        k = std::rand()%y_num;
                
                    
        offset_max = std::min((uint64_t)(k + filter_offset),(uint64_t)(y_num-1));
        offset_min = std::max((uint64_t)(k - filter_offset),(uint64_t)0);
                    
        uint64_t f = 0;
        for(uint64_t c = offset_min;c <= offset_max;c++){
                        
            //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
            output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + i*y_num + c]*filter[f];
            f++;
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1);
    
    std::cout << " Pixel Filter Size Random: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}
template<typename S>
void apr_filter_full(PartCellStructure<S,uint64_t>& pc_struct){
    //
    //  Calculate Neighbours Using Iterators
    //
    //
    
    Mesh_data<uint16_t> filter_img;
    
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct);
    
    ExtraPartCellData<float> filter_output;
    filter_output.initialize_structure_parts(part_new.particle_data);
    //
    Part_timer timer;
    
    timer.start_timer("interp");
    
    pc_struct.interp_parts_to_pc(filter_img,pc_struct.part_data.particle_data);
    
    timer.stop_timer();
    
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    int y_;
    
    timer.verbose_flag = false;
    float num_repeats = 50;
    
    std::vector<float> filter;

    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 5;
    filter.resize(filter_offset*2 +1,1);
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    const int x_num_m = filter_img.x_num;
    const int y_num_m = filter_img.y_num;
    const int z_num_m = filter_img.z_num;
    
    timer.start_timer("compute gradient y");
    
    for(int r = 0;r < num_repeats;r++){
        
        
        for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = part_new.access_data.x_num[depth];
            const unsigned int z_num_ = part_new.access_data.z_num[depth];
            
            CurrentLevel<float,uint64_t> curr_level;
            curr_level.set_new_depth(depth,part_new);
            
#pragma omp parallel for default(shared) private(z_,x_,j_,y_,offset_min,offset_max) firstprivate(curr_level) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    curr_level.set_new_xz(x_,z_,part_new);
        
                    for(j_ = 0;j_ < curr_level.j_num;j_++){
                        
                        bool iscell = curr_level.new_j(j_,part_new);
                        
                        if (iscell){
                            //Indicates this is a particle cell node
                            curr_level.update_cell(part_new);
                            
                            y_ = curr_level.y;
                            
                            offset_max = std::min((uint64_t)(y_ + filter_offset),(uint64_t)(y_num_m-1));
                            offset_min = std::max((uint64_t)(y_ - filter_offset),(uint64_t)0);
                            
                            uint64_t f = 0;
                            S temp = 0;
                            for(uint64_t c = offset_min;c <= offset_max;c++){
                                
                                //need to change the below to the vector
                                temp += filter_img.mesh[z_*x_num_m*y_num_m + x_*y_num_m + c]*filter[f];
                                f++;
                            }
                            
                            curr_level.get_part(filter_output) = temp;

                            
                            
                        } else {
                            
                            curr_level.update_gap();
                            
                        }
                        
                        
                    }
                }
            }
        }
    }
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1);
    
    std::cout << " Adaptive Filter took: " << time/num_repeats << std::endl;
    
    
}


#endif