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
#include "../../benchmarks/analysis/AnalysisData.hpp"


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
void particle_linear_neigh_access_alt_1(PartCellStructure<S,uint64_t>& pc_struct){
    //
    //  Calculate Neighbours Using Iterators
    //
    //
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct);
    
    ExtraPartCellData<float> filter_output;
    filter_output.initialize_structure_cells(part_new.access_data);
    
    ExtraPartCellData<float> particle_data;
    particle_data.initialize_structure_cells(part_new.access_data);
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
            NeighOffset<float,uint64_t> neigh_y1(1);
            NeighOffset<float,uint64_t> neigh_y2(2);
            NeighOffset<float,uint64_t> neigh_y3(3);
            NeighOffset<float,uint64_t> neigh_y4(4);
            NeighOffset<float,uint64_t> neigh_y5(5);
            
            neigh_y.set_new_depth(curr_level,part_new);
            neigh_y1.set_new_depth(curr_level,part_new);
            neigh_y2.set_new_depth(curr_level,part_new);
            neigh_y3.set_new_depth(curr_level,part_new);
            neigh_y4.set_new_depth(curr_level,part_new);
            neigh_y5.set_new_depth(curr_level,part_new);
            
#pragma omp parallel for default(shared) private(z_,x_,j_,int_neigh_p) firstprivate(curr_level,neigh_y,neigh_y1,neigh_y2,neigh_y3,neigh_y4,neigh_y5) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    curr_level.set_new_xz(x_,z_,part_new);
                    
                    neigh_y.set_new_row(curr_level,part_new);
                    neigh_y1.set_new_row(curr_level,part_new);
                    neigh_y2.set_new_row(curr_level,part_new);
                    neigh_y3.set_new_row(curr_level,part_new);
                    neigh_y4.set_new_row(curr_level,part_new);
                    neigh_y5.set_new_row(curr_level,part_new);
                    
                    
                    for(j_ = 0;j_ < curr_level.j_num;j_++){
                        
                        bool iscell = curr_level.new_j(j_,part_new);
                        
                        if (iscell){
                            //Indicates this is a particle cell node
                            curr_level.update_cell(part_new);
                            
                            neigh_y.iterate(curr_level,part_new);
                            neigh_y1.iterate(curr_level,part_new);
                            neigh_y2.iterate(curr_level,part_new);
                            neigh_y3.iterate(curr_level,part_new);
                            neigh_y4.iterate(curr_level,part_new);
                            neigh_y5.iterate(curr_level,part_new);
                            
                            int_neigh_p = neigh_y.get_part(part_new.particle_data);
                            int_neigh_p += neigh_y1.get_part(part_new.particle_data);
                            int_neigh_p += neigh_y2.get_part(part_new.particle_data);
                            int_neigh_p += neigh_y3.get_part(part_new.particle_data);
                            int_neigh_p += neigh_y4.get_part(part_new.particle_data);
                            int_neigh_p += neigh_y5.get_part(part_new.particle_data);
                            
                            curr_level.get_val(filter_output) = int_neigh_p;
                            
                        } else {
                            
                            curr_level.update_gap();
                            
                        }
                        
                        
                    }
                }
            }
        }
    }
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << "Get neigh particle linear alt: " << time << std::endl;
    std::cout << "per 1000000 particles took: " << time/(1.0*pc_struct.get_number_parts()/1000000.0) << std::endl;
    
    
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

void neigh_cells(PartCellData<uint64_t>& pc_data){
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
    

    
    const int num_dir = 6;
    
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
void particle_linear_neigh_access(PartCellStructure<float,uint64_t>& pc_struct,float num_repeats,AnalysisData& analysis_data){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct);
    
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);
    
    
    ExtraPartCellData<float> filter_output;
    filter_output.initialize_structure_cells(pc_data);
    
    ExtraPartCellData<float> particle_data;
    particle_data.initialize_structure_cells(pc_data);
    
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
    
    const int num_dir = 6;
    
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
                            
                            //for(int dir = 0;dir < num_dir;dir++){
                               // pc_data.get_neighs_face(curr_key,node_val_pc,dir,neigh_cell_keys);
                           // }

                            pc_data.get_neighs_all(curr_key,node_val_pc,neigh_cell_keys);

                            
                            for(int dir = 0;dir < num_dir;dir++){
                                //loop over the nieghbours
                                for(int n = 0; n < neigh_cell_keys.neigh_face[dir].size();n++){
                                    // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                    uint64_t neigh_key = neigh_cell_keys.neigh_face[dir][n];
                                    
                                    if(neigh_key > 0){
                                        //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                        part_int+= particle_data.get_val(neigh_key);
                                    }
                                    
                                }
                            }
                            
                            filter_output.data[i][offset_pc_data][j_] = part_int;
                            
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

    analysis_data.add_float_data("neigh_part_linear_total",time);
    analysis_data.add_float_data("neigh_part_linear_perm",time/(1.0*pc_struct.get_number_parts()/1000000.0));

    std::cout << "Get neigh particle linear: " << time << std::endl;
    std::cout << "per 1000000 particles took: " << time/(1.0*pc_struct.get_number_parts()/1000000.0) << std::endl;
    
}
void move_cells_random(PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t> part_new){
    //
    //  Bevan Cheeseman 2017
    //
    //  Non-cache friendly neighbour access
    //
    //
    
    Part_timer timer;
    
    ExtraPartCellData<float> filter_output;
    filter_output.initialize_structure_cells(part_new.access_data);
    
    //
    // Extra variables required
    //
    
    timer.verbose_flag = false;
    
    float num_repeats = 50000000;
    
    CurrentLevel<float,uint64_t> curr_level;
    
    //
    //  Initialize Randomly
    //
    
    int j_i = 0;
    
    curr_level.type = 0;
    
    while((curr_level.type == 0) || (j_i <= 1)){
        
        int depth_i = (int) std::rand()%(pc_data.depth_max-pc_data.depth_min) + (int) pc_data.depth_min;
        
        int x_num_ = pc_data.x_num[depth_i];
        int z_num_ = pc_data.z_num[depth_i];
        
        int x_i = std::rand()%x_num_;
        int z_i = std::rand()%z_num_;
        
        int offset_pc_data = x_num_*z_i + x_i;
        
        int j_num = (int) pc_data.data[depth_i][offset_pc_data].size();
        
        j_i = (std::rand()%(j_num-2)) + 1;
        
        curr_level.init(x_i,z_i,j_i,depth_i,part_new);
        
    }
    
    
    //iterate loop;
    
    timer.start_timer("neigh_cell_comp");
    
    unsigned int dir = 0;
    unsigned int index = 0;
    float neigh_int= 0;
    
    for(int r = 0;r < num_repeats;r++){
        //choose one of the 6 directions (+y,-y,+x..
        dir = std::rand()%6;
        //if there are children which one
        index = std::rand()%4;
        
        //move randomly
        curr_level.move_cell(dir,index,part_new,pc_data);
        
        //get all
        curr_level.update_all_neigh(pc_data);
        
        neigh_int = 0;
        
        for(int i = 0;i < 6;i++){
            neigh_int += curr_level.get_neigh_int(i,part_new,pc_data);
            
        }
        
        curr_level.get_val(filter_output) = neigh_int;
    }
    
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1)/(num_repeats/1000000.0);
    
    std::cout << "Particle Move random 1000000* : " << time << std::endl;
    
    
    
}
template<typename U>
void pixels_move_random(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};
    
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    
    int j = 0;
    int k = 0;
    int i = 0;
    
    int j_n = 0;
    int k_n = 0;
    int i_n = 0;
    
    i = std::rand()%x_num;
    j = std::rand()%z_num;
    k = std::rand()%y_num;
    
    unsigned int dir = 0;
    unsigned int index = 0;
    float neigh_sum = 0;
    
    float num_repeats = 5000000;
    
    for(int r = 0;r < num_repeats;r++){
        
       //get new random direction
        //choose one of the 6 directions (+y,-y,+x..
        dir = std::rand()%6;
        //if there are children which one
        index = std::rand()%4;
        
        i = std::min(std::max((int)0,i+dir_x[dir]),(int)x_num-1);
        j = std::min(std::max((int)0,j+dir_z[dir]),(int)z_num-1);
        k = std::min(std::max((int)0,k+dir_y[dir]),(int)y_num-1);
        
        neigh_sum = 0;
        
        for(int  d  = 0;d < 6;d++){
            
            i_n = i + dir_x[d];
            k_n = k + dir_y[d];
            j_n = j + dir_z[d];
            
            //check boundary conditions
            if((i_n >= 0) & (i_n < x_num) ){
                if((j_n >=0) & (j_n < z_num) ){
                    if((k_n >=0) & (k_n < y_num) ){
                        neigh_sum += input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                    }
                }
            }
            
        }
        
        output_data.mesh[j*x_num*y_num + i*y_num + k] = neigh_sum;
    }
    
    
    (void) output_data.mesh;
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1);
    
    std::cout << " Pixel Move random 1000000* : " << (x_num*y_num*z_num) << " took: " << time/(num_repeats/1000000.0) << std::endl;
    
}


void particle_random_access(PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct);
    
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);
    
    int num_repeats = pc_struct.get_number_parts();
    
    Part_timer timer;
    
    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information
    

    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //
    
    ExtraPartCellData<float> filter_output;
    filter_output.initialize_structure_cells(part_new.access_data);
    
    ExtraPartCellData<float> particle_data;
    particle_data.initialize_structure_cells(part_new.access_data);
    
    size_t j_num;
    timer.verbose_flag = false;
    
    
    const int num_dir = 6;
    
    timer.start_timer("neigh_cell_comp");
    
    
    int x_,z_,j_,depth;
    size_t offset_pc_data;
    
    int counter = 0;
    
    for(int r = 0;r < num_repeats;r++){
        
        depth = std::rand()%(pc_data.depth_max-pc_data.depth_min) + pc_data.depth_min;
        
        int x_num_ = pc_data.x_num[depth];
        int z_num_ = pc_data.z_num[depth];
        
        x_ = std::rand()%x_num_;
        z_ = std::rand()%z_num_;
        
        offset_pc_data = x_num_*z_ + x_;

        j_num = pc_data.data[depth][offset_pc_data].size();
        
        j_ = std::rand()%j_num;
        
        
        //both z and x are explicitly accessed in the structure
        curr_key = 0;
        
        //particle cell node value, used here as it is requried for getting the particle neighbours
        node_val_pc = pc_data.data[depth][offset_pc_data][j_];
        
        if (!(node_val_pc&1)){
            //Indicates this is a particle cell node
            //y_coord++;
            float part_int= 0;
            
            pc_data.pc_key_set_z(curr_key,z_);
            pc_data.pc_key_set_x(curr_key,x_);
            pc_data.pc_key_set_depth(curr_key,depth);
            
            pc_data.pc_key_set_j(curr_key,j_);
            
            
            for(int dir = 0;dir < num_dir;dir++){
                pc_data.get_neighs_face(curr_key,node_val_pc,dir,neigh_cell_keys);
            }
            
            for(int dir = 0;dir < num_dir;dir++){
                //loop over the nieghbours
                for(int n = 0; n < neigh_cell_keys.neigh_face[dir].size();n++){
                    // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                    uint64_t neigh_key = neigh_cell_keys.neigh_face[dir][n];
                    
                    if(neigh_key > 0){
                        //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                        part_int+= particle_data.get_val(neigh_key);
                    }
                    
                }
            }
            
            filter_output.data[depth][offset_pc_data][j_] = part_int;
            counter++;
            
        } else {
            r--;
            counter++;
        }
        
    }



    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1);
    
    timer.start_timer("neigh_cell_comp");
    
    
    for(int r = 0;r < counter;r++){
        
        depth = std::rand()%(pc_data.depth_max-pc_data.depth_min) + pc_data.depth_min;
        
        int x_num_ = pc_data.x_num[depth];
        int z_num_ = pc_data.z_num[depth];
        
        x_ = std::rand()%x_num_;
        z_ = std::rand()%z_num_;
        
        offset_pc_data = x_num_*z_ + x_;
        
        j_num = pc_data.data[depth][offset_pc_data].size();
        
        j_ = std::rand()%j_num;
        
        (void) j_;
        (void) x_;
        (void) z_;
        (void) j_num;
        (void) offset_pc_data;
        (void) depth;
        
    }
    
    timer.stop_timer();
    
    float time2 = (timer.t2 - timer.t1);
    //std::cout << "Overhead " << time2 << std::endl;
    
    //std::cout << "Random Access Neigh Particles: " << ((time-time2)) << std::endl;
    //std::cout << "per 1000000 particles: " << (time/(num_repeats/1000000)) << std::endl;

    analysis_data.add_float_data("random_access_parts_neigh_total",time-time2);
    analysis_data.add_float_data("random_access_parts_neigh_perm",(time-time2)/(1.0*num_repeats/1000000));

}


template<typename U>
void pixel_filter_full(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num,uint64_t filter_offset,float num_repeats,AnalysisData& analysis_data){
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
    
    
    filter.resize(filter_offset*2 +1,1);
    
    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
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
    
    //std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " y took: " << time << std::endl;
    
    // x loop
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max) firstprivate(temp_vec)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                
                for(k = 0;k < y_num;k++){
                    
                    offset_max = std::min((uint64_t)(i + filter_offset),(uint64_t)(x_num-1));
                    offset_min = std::max((uint64_t)(i - filter_offset),(uint64_t)0);
                    
                    uint64_t f = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){
                        
                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + c*y_num + k]*filter[f];
                        f++;
                    }
                    
                }
            }
        }
        
    }
    
    (void) output_data.mesh;
    
    timer.stop_timer();
    float time2 = (timer.t2 - timer.t1)/num_repeats;
    
    //std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " x took: " << time2 << std::endl;
    
    // z loop
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max) firstprivate(temp_vec)
         for(j = 0; j < z_num;j++){
             for(i = 0; i < x_num;i++){
               
                
                for(k = 0;k < y_num;k++){
                    
                    offset_max = std::min((uint64_t)(j + filter_offset),(uint64_t)(z_num-1));
                    offset_min = std::max((uint64_t)(j - filter_offset),(uint64_t)0);
                    
                    uint64_t f = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){
                        
                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[c*x_num*y_num + i*y_num + k]*filter[f];
                        f++;
                    }
                    
                }
            }
        }
        
    }
    
    (void) output_data.mesh;
    
    timer.stop_timer();
    float time3 = (timer.t2 - timer.t1)/num_repeats;
    
   // std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " z took: " << time3 << std::endl;
    
   // std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " all took: " << (time+time2+time3) << std::endl;

    analysis_data.add_float_data("pixel_filter_y",time);
    analysis_data.add_float_data("pixel_filter_x",time2);
    analysis_data.add_float_data("pixel_filter_z",time3);

    analysis_data.add_float_data("pixel_filter_all",time + time2 + time3);
    
}
template<typename U>
void pixels_linear_neigh_access(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats,AnalysisData& analysis_data){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    
    int j = 0;
    int k = 0;
    int i = 0;
    
    int j_n = 0;
    int k_n = 0;
    int i_n = 0;
    
    //float neigh_sum = 0;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                for(k = 0;k < y_num;k++){
                    float neigh_sum = 0;
                    
                    for(int  d  = 0;d < 6;d++){
                        
                        i_n = i + dir_x[d];
                        k_n = k + dir_y[d];
                        j_n = j + dir_z[d];
                        
                        //check boundary conditions
                        if((i_n >=0) & (i_n < x_num) ){
                            if((j_n >=0) & (j_n < z_num) ){
                                if((k_n >=0) & (k_n < y_num) ){
                                    neigh_sum += input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                                }
                            }
                        }
                        
                    }
                    
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = neigh_sum;
                    
                }
            }
        }
        
    }
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    //std::cout << "Pixel Linear Neigh: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    //std::cout << "per 1000000 pixel took: " << (time)/((1.0*x_num*y_num*z_num)/1000000.0) << std::endl;

    analysis_data.add_float_data("neigh_pixel_linear_total",time);
    analysis_data.add_float_data("neigh_pixel_linear_perm",(time)/((1.0*x_num*y_num*z_num)/1000000.0));

}
template<typename U>
void pixel_neigh_random(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num,AnalysisData& analysis_data){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    int j = 0;
    int k = 0;
    int i = 0;
    
    int j_n = 0;
    int k_n = 0;
    int i_n = 0;
    
    float num_repeats = input_data.mesh.size();
    float neigh_sum = 0;
    
    for(int r = 0;r < num_repeats;r++){
        
        i = std::rand()%x_num;
        j = std::rand()%z_num;
        k = std::rand()%y_num;
                
                    
//        offset_max = std::min((uint64_t)(k + filter_offset),(uint64_t)(y_num-1));
//        offset_min = std::max((uint64_t)(k - filter_offset),(uint64_t)0);
//                    
//        uint64_t f = 0;
//        for(uint64_t c = offset_min;c <= offset_max;c++){
//                        
//            //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
//            output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + i*y_num + c]*filter[f];
//            f++;
//        }
        
        neigh_sum = 0;
        
        for(int  d  = 0;d < 6;d++){
            
            i_n = i + dir_x[d];
            k_n = k + dir_y[d];
            j_n = j + dir_z[d];
            
            //check boundary conditions
            if((i_n >=0) & (i_n < x_num) ){
                if((j_n >=0) & (j_n < z_num) ){
                    if((k_n >=0) & (k_n < y_num) ){
                        neigh_sum += input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                    }
                }
            }
            
        }
        
       output_data.mesh[j*x_num*y_num + i*y_num + k] = neigh_sum;

        
    }
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1);
    
    timer.start_timer("full previous filter");
    
    for(int r = 0;r < num_repeats;r++){
        
        i = std::rand()%x_num;
        j = std::rand()%z_num;
        k = std::rand()%y_num;
        
        
        (void) i;
        (void) j;
        (void) k;
        
    }
    
    
    timer.stop_timer();
    float time2 = (timer.t2 - timer.t1);
    
    //std::cout << "Random Access Pixel: Size: " << (x_num*y_num*z_num) << " took: " << (time-time2) << std::endl;
    //std::cout << "per 1000000 pixel took: " << (time-time2)/((1.0*x_num*y_num*z_num)/1000000.0) << std::endl;

    analysis_data.add_float_data("random_access_pixel_neigh_total",time-time2);
    analysis_data.add_float_data("random_access_pixel_neigh_perm",(time-time2)/((1.0*x_num*y_num*z_num)/1000000.0));
    
}
template<typename S>
void apr_filter_full(PartCellStructure<S,uint64_t>& pc_struct,uint64_t filter_offset,float num_repeats,AnalysisData& analysis_data){
    //
    //  Calculate Neighbours Using Iterators
    //
    //
    
    
    ParticleDataNew<S, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct);
    
    ExtraPartCellData<S> filter_output;
    filter_output.initialize_structure_cells(part_new.access_data);
    //
    Part_timer timer;
    
    Mesh_data<S> filter_img;
    Mesh_data<S> temp_array;
    
    int x_dim = ceil(pc_struct.org_dims[0]/2.0)*2;
    int z_dim = ceil(pc_struct.org_dims[1]/2.0)*2;
    int y_dim = ceil(pc_struct.org_dims[2]/2.0)*2;
    
    filter_img.mesh.resize(x_dim*z_dim*y_dim);
    temp_array.mesh.resize(x_dim*z_dim*y_dim);
    
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    int y_;
    
    timer.verbose_flag = false;
    
    std::vector<float> filter;

    timer.verbose_flag = false;
    //timer.start_timer("full previous filter");
    
    filter.resize(filter_offset*2 +1,1);
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    const int x_num_m = filter_img.x_num;
    const int y_num_m = filter_img.y_num;
    const int z_num_m = filter_img.z_num;



    timer.start_timer("compute gradient y no interp temp_vec");

    for(int r = 0;r < num_repeats;r++){


        std::vector<float> temp_vec;
        temp_vec.resize(y_dim,0);

        for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = part_new.access_data.x_num[depth];
            const unsigned int z_num_ = part_new.access_data.z_num[depth];

            CurrentLevel<float,uint64_t> curr_level;
            curr_level.set_new_depth(depth,part_new);

#pragma omp parallel for default(shared) private(z_,x_,j_,y_,offset_min,offset_max) firstprivate(curr_level,temp_vec) if(z_num_*x_num_ > 100)
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
                                temp += temp_vec[c]*filter[f];
                                f++;
                            }

                            curr_level.get_val(filter_output) = temp;


                        } else {

                            curr_level.update_gap();

                        }


                    }
                }
            }
        }
    }

    timer.stop_timer();

    float time_vec = (timer.t2 - timer.t1);


    timer.start_timer("compute gradient y");
    
    for(int r = 0;r < num_repeats;r++){
        
        pc_struct.interp_parts_to_pc(pc_struct.part_data.particle_data,filter_img,temp_array);
        
        
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
                            
                            curr_level.get_val(filter_output) = temp;
                            
                            
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
    
    //std::cout << " Adaptive Filter y took: " << time/num_repeats << std::endl;
    
    for(int r = 0;r < num_repeats;r++){
        
        pc_struct.interp_parts_to_pc(pc_struct.part_data.particle_data,filter_img,temp_array);
        
        
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

                            
                            offset_max = std::min((uint64_t)(x_ + filter_offset),(uint64_t)(x_num_m-1));
                            offset_min = std::max((uint64_t)(x_ - filter_offset),(uint64_t)0);
                            
                            uint64_t f = 0;
                            S temp = 0;
                            for(uint64_t c = offset_min;c <= offset_max;c++){
                                //NEED TO CHANGE THE COORDINATES ARE WRONG!!!!!

                                //need to change the below to the vector
                                temp += filter_img.mesh[z_*x_num_m*y_num_m + c*y_num_m + y_]*filter[f];
                                f++;
                            }
                            
                            curr_level.get_val(filter_output) = temp;

                            
                        } else {
                            
                            curr_level.update_gap();
                            
                        }
                        
                        
                    }
                }
            }
        }
    }
    
    timer.stop_timer();
    
    float time2 = (timer.t2 - timer.t1);
    
    //std::cout << " Adaptive Filter x took: " << time2/num_repeats << std::endl;
    
    for(int r = 0;r < num_repeats;r++){
        
        pc_struct.interp_parts_to_pc(pc_struct.part_data.particle_data,filter_img,temp_array);
        
        
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
                            
                            offset_max = std::min((uint64_t)(z_ + filter_offset),(uint64_t)(z_num_m-1));
                            offset_min = std::max((uint64_t)(z_ - filter_offset),(uint64_t)0);
                            
                            uint64_t f = 0;
                            S temp = 0;
                            for(uint64_t c = offset_min;c <= offset_max;c++){
                                
                                //need to change the below to the vector
                                temp += filter_img.mesh[c*x_num_m*y_num_m + x_*y_num_m + y_]*filter[f];
                                f++;
                            }
                            
                            curr_level.get_val(filter_output) = temp;
                            
                            
                            
                        } else {
                            
                            curr_level.update_gap();
                            
                        }
                        
                        
                    }
                }
            }
        }
    }
    
    timer.stop_timer();
    
    float time3 = (timer.t2 - timer.t1);


    timer.start_timer("interp");
    for(int r = 0;r < num_repeats;r++){

        pc_struct.interp_parts_to_pc(pc_struct.part_data.particle_data,filter_img,temp_array);

    }
    timer.stop_timer();

    float time_interp = (timer.t2 - timer.t1);

    //std::cout << "Particle pc Filter z took: " << time3/num_repeats << std::endl;
    
    //std::cout << "Particle pc Filter all took: " << (time + time2 + time3)/num_repeats << std::endl;

    analysis_data.add_float_data("particle_filter_y",time/num_repeats);
    analysis_data.add_float_data("particle_filter_x",time2/num_repeats);
    analysis_data.add_float_data("particle_filter_z",time3/num_repeats);

    analysis_data.add_float_data("particle_filter_y_vec",time_vec/num_repeats);

    analysis_data.add_float_data("particle_filter_all",(time + time2 + time3)/num_repeats);

    analysis_data.add_float_data("particle_filter_y_no_interp",(time-time_interp)/num_repeats);
    analysis_data.add_float_data("particle_filter_x_no_interp",(time2-time_interp)/num_repeats);
    analysis_data.add_float_data("particle_filter_z_no_interp",(time3-time_interp)/num_repeats);

    analysis_data.add_float_data("particle_filter_all_no_interp",(time + time2 + time3 - time_interp*3)/num_repeats);

    std::cout << (time+time2 + time3)/num_repeats << std::endl;
    //std::cout << time_vec/num_repeats << std::endl;

}
template<typename U>
void sep_neigh_filter(PartCellData<uint64_t>& pc_data,ExtraPartCellData<U>& input_data,std::vector<U>& filter){
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    //copy across
    ExtraPartCellData<U> filter_output;
    filter_output.initialize_structure_cells(pc_data);
    
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
    
    if(filter.size() != 3){
        std::cout << " Wrong Sized Filter" << std::endl;
        return;
    }
    
    
    timer.verbose_flag = false;

    
    timer.start_timer("neigh_cell_comp");
    
    for(int dir = 0;dir < 3;dir++){
        
        
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
                            
                            pc_data.get_neighs_axis(curr_key,node_val_pc,neigh_cell_keys,dir);
                            
                            U temp_int = 0;
                            int counter= 0;
                           
                            U curr_int = input_data.get_val(curr_key);
                            
                             U accum_int = filter[1]*curr_int;
                            
                            if(curr_int > 0){
                                //(+direction)
                                //loop over the nieghbours
                                for(int n = 0; n < neigh_cell_keys.neigh_face[2*dir].size();n++){
                                    // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                    uint64_t neigh_key = neigh_cell_keys.neigh_face[2*dir][n];
                                    
                                    if(neigh_key > 0){
                                        //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                        temp_int += input_data.get_val(neigh_key);
                                        counter++;
                                    }
                                    
                                    
                                }
                                
                                if(temp_int==0){
                                    temp_int = curr_int;
                                } else {
                                    temp_int = temp_int/counter;
                                }
                                counter = 0;
                                
                                accum_int += temp_int*filter[2];
                                
                                temp_int = 0;
                                //(-direction)
                                //loop over the nieghbours
                                for(int n = 0; n < neigh_cell_keys.neigh_face[2*dir+1].size();n++){
                                    // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                    uint64_t neigh_key = neigh_cell_keys.neigh_face[2*dir+1][n];
                                    
                                    if(neigh_key > 0){
                                        //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                        temp_int += input_data.get_val(neigh_key);
                                        counter++;
                                    }
                                }
                                
                                if(temp_int==0){
                                    temp_int = curr_int;
                                } else {
                                    temp_int = temp_int/counter;
                                }
                                
                                accum_int += temp_int*filter[0];
                                
                            }
                            
                            filter_output.get_val(curr_key) = accum_int;
                            
                            
                            
                        } else {
                            // Inidicates this is not a particle cell node, and is a gap node
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        //running them consecutavely
        std::swap(filter_output,input_data);

    }
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1);
    
    //std::cout << "Seperable Smoothing Filter: " << time << std::endl;
    

    
}
template<typename S>
void new_filter_part(PartCellStructure<S,uint64_t>& pc_struct,uint64_t filter_offset,float num_repeats,AnalysisData& analysis_data){
//
//     Bevan Cheeseman 2017
//


    ParticleDataNew<S, uint64_t> part_new;

    part_new.initialize_from_structure(pc_struct);


    ExtraPartCellData<S> filter_output;
    filter_output.initialize_structure_cells(part_new.access_data);

    ExtraPartCellData<S> part_data;

    part_new.create_particles_at_cell_structure(part_data);



    //
    Part_timer timer;

    Mesh_data<S> filter_img;
    Mesh_data<S> temp_array;

    int y_dim = ceil(pc_struct.org_dims[0]/2.0)*2;
    int x_dim = ceil(pc_struct.org_dims[1]/2.0)*2;
    int z_dim = ceil(pc_struct.org_dims[2]/2.0)*2;

    filter_img.mesh.resize(x_dim*z_dim*y_dim);
    temp_array.mesh.resize(x_dim*z_dim*y_dim);

    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    int y_;

    timer.verbose_flag = false;

    std::vector<float> filter;

    timer.verbose_flag = false;
    //timer.start_timer("full previous filter");

    filter.resize(filter_offset*2 +1,1);

    uint64_t offset_min;
    uint64_t offset_max;

    const int x_num_m = filter_img.x_num;
    const int y_num_m = filter_img.y_num;
    const int z_num_m = filter_img.z_num;


    std::vector<float> temp_vec;


    timer.start_timer("compute gradient y no interp temp_vec");

    for(int r = 0;r < num_repeats;r++){



        for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = part_new.access_data.x_num[depth];
            const unsigned int z_num_ = part_new.access_data.z_num[depth];

            CurrentLevel<float,uint64_t> curr_level;
            curr_level.set_new_depth(depth,part_new);

#pragma omp parallel for default(shared) private(z_,x_,j_,y_,offset_min,offset_max) firstprivate(curr_level,temp_vec) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure

                temp_vec.resize(y_dim*x_dim,0);

                for(x_ = 0;x_ < x_num_;x_++){

                    curr_level.set_new_xz(x_,z_,part_new);

                    for(j_ = 0;j_ < curr_level.j_num;j_++){

                        bool iscell = curr_level.new_j(j_,part_new);

                        if (iscell){
                            //Indicates this is a particle cell node
                            curr_level.update_cell(part_new);

                            y_ = curr_level.y;

                            //float temp =  curr_level.get_val(part_data);

                            temp_vec[y_*x_num_ + x_] =  curr_level.get_val(part_data);




                        } else {

                            curr_level.update_gap();

                        }


                    }
                }
            }
        }
    }

    timer.stop_timer();

    float time_vec = (timer.t2 - timer.t1)/num_repeats;

    std::cout << "time: " << time_vec << std::endl;



}

void filter_slice(){





}


#endif