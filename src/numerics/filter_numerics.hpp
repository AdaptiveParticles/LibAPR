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

#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "src/data_structures/APR/ExtraPartCellData.hpp"
#include "benchmarks/development/Tree/PartCellParent.hpp"

#include "filter_help/FilterOffset.hpp"
#include "filter_help/FilterLevel.hpp"

#include "src/data_structures/APR/CurrLevel.hpp"
#include "filter_help/NeighOffset.hpp"

//#include "../../test/utils.h"
#include "../../benchmarks/analysis/AnalysisData.hpp"

#include "misc_numerics.hpp"
#include "parent_numerics.hpp"


template<typename T>
static void iterate_temp_vec(std::vector<T>& temp_vec,std::vector<T>& temp_vec_depth){
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
static void iterate_temp_vec(std::vector<T>& temp_vec){
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
static void particle_linear_neigh_access_alt_1(PartCellStructure<S,uint64_t>& pc_struct){
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
void lin_access_parts(PartCellStructure<S,uint64_t>& pc_struct){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //
    
    ExtraPartCellData<S> filter_output;
    filter_output.initialize_structure_parts(pc_struct.part_data.particle_data);
    
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
                                
                                float temp = 0;
                                //first set the particle index value in the particle_data array (stores the intensities)
                                pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                                //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                                
                                pc_struct.part_data.get_part_neighs_all(p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                                
                                for(int dir = 0;dir < 6;++dir){
                                    for(int n = 0; n < neigh_part_keys.neigh_face[dir].size();n++){
                                        // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                        uint64_t neigh_key = neigh_part_keys.neigh_face[dir][n];
                                        
                                        if(neigh_key > 0){
                                            //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                            temp += pc_struct.part_data.get_part(neigh_key);
                                        }
                                        
                                    }
                                    
                                }
                                
                                filter_output.data[i][offset_pc_data][part_offset+p] = temp;
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
    
    std::cout << "Get Neigh Old: " << time << std::endl;
    
}

static void neigh_cells(PartCellData<uint64_t>& pc_data){
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
static void particle_linear_neigh_access(PartCellStructure<float,uint64_t>& pc_struct,float num_repeats,AnalysisData& analysis_data){   //  Calculate connected component from a binary mask
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
static void move_cells_random(PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t> part_new){
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


static void particle_random_access(PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){   //  Calculate connected component from a binary mask
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
        
        depth = std::rand()%(pc_data.depth_max-pc_data.depth_min +1) + pc_data.depth_min;

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


template<typename U,typename V>
Mesh_data<U> pixel_filter_full(Mesh_data<V>& input_data,std::vector<U>& filter,float num_repeats,AnalysisData& analysis_data){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    int filter_offset = (filter.size()-1)/2;

    unsigned int x_num = input_data.x_num;
    unsigned int y_num = input_data.y_num;
    unsigned int z_num = input_data.z_num;

    Mesh_data<U> output_data;
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");

    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);

    uint64_t offset_min;
    uint64_t offset_max;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    for(int r = 0;r < num_repeats;r++){

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    offset_max = std::min((int)(k + filter_offset),(int)(y_num-1));
                    offset_min = std::max((int)(k - filter_offset),(int)0);

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

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    offset_max = std::min((int)(i + filter_offset),(int)(x_num-1));
                    offset_min = std::max((int)(i - filter_offset),(int)0);

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

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){


                for(k = 0;k < y_num;k++){

                    offset_max = std::min((int)(j + filter_offset),(int)(z_num-1));
                    offset_min = std::max((int)(j - filter_offset),(int)0);

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

    std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " all took: " << (time+time2+time3) << std::endl;

    analysis_data.add_float_data("pixel_filter_y",time);
    analysis_data.add_float_data("pixel_filter_x",time2);
    analysis_data.add_float_data("pixel_filter_z",time3);

    analysis_data.add_float_data("pixel_filter_all",time + time2 + time3);

    return output_data;

}
template<typename U,typename V>
Mesh_data<U> pixel_filter_full_mult(Mesh_data<V> input_data,std::vector<U> filter_y,std::vector<U> filter_x,std::vector<U> filter_z,float num_repeats,AnalysisData& analysis_data){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    int filter_offset = (filter_y.size()-1)/2;

    unsigned int x_num = input_data.x_num;
    unsigned int y_num = input_data.y_num;
    unsigned int z_num = input_data.z_num;

    Mesh_data<U> output_data;
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");

    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);

    uint64_t offset_min;
    uint64_t offset_max;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    for(int r = 0;r < num_repeats;r++){

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    offset_max = std::min((int)(k + filter_offset),(int)(y_num-1));
                    offset_min = std::max((int)(k - filter_offset),(int)0);

                    uint64_t f = 0;

                    output_data.mesh[j*x_num*y_num + i*y_num + k] = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + i*y_num + c]*filter_y[f];
                        f++;
                    }

                }
            }
        }

    }

    (void) output_data.mesh;

    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;


    std::swap(output_data.mesh,input_data.mesh);

    //std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " y took: " << time << std::endl;

    // x loop

    for(int r = 0;r < num_repeats;r++){

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    offset_max = std::min((int)(i + filter_offset),(int)(x_num-1));
                    offset_min = std::max((int)(i - filter_offset),(int)0);

                    uint64_t f = 0;
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = 0;

                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + c*y_num + k]*filter_x[f];
                        f++;
                    }

                }
            }
        }

    }


    timer.stop_timer();
    float time2 = (timer.t2 - timer.t1)/num_repeats;

    std::swap(output_data.mesh,input_data.mesh);

    //std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " x took: " << time2 << std::endl;

    // z loop

    for(int r = 0;r < num_repeats;r++){

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){


                for(k = 0;k < y_num;k++){

                    offset_max = std::min((int)(j + filter_offset),(int)(z_num-1));
                    offset_min = std::max((int)(j - filter_offset),(int)0);

                    output_data.mesh[j*x_num*y_num + i*y_num + k] = 0;

                    uint64_t f = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[c*x_num*y_num + i*y_num + k]*filter_z[f];
                        f++;
                    }

                }
            }
        }

    }


    timer.stop_timer();
    float time3 = (timer.t2 - timer.t1)/num_repeats;

    // std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " z took: " << time3 << std::endl;

    std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " all took: " << (time+time2+time3) << std::endl;

    analysis_data.add_float_data("pixel_filter_y",time);
    analysis_data.add_float_data("pixel_filter_x",time2);
    analysis_data.add_float_data("pixel_filter_z",time3);

    analysis_data.add_float_data("pixel_filter_all",time + time2 + time3);

    return output_data;

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
    
    std::cout << "Pixel Linear Neigh: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    std::cout << "per 1000000 pixel took: " << (time)/((1.0*x_num*y_num*z_num)/1000000.0) << std::endl;

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
    
    float num_repeats = 10000000;
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

    float est_full_time = (time-time2)*(1.0*x_num*y_num*z_num)/num_repeats;
    
    //std::cout << "Random Access Pixel: Size: " << (x_num*y_num*z_num) << " took: " << (time-time2) << std::endl;
    //std::cout << "per 1000000 pixel took: " << (time-time2)/((1.0*x_num*y_num*z_num)/1000000.0) << std::endl;

    analysis_data.add_float_data("random_access_pixel_neigh_total",est_full_time);
    analysis_data.add_float_data("random_access_pixel_neigh_perm",(est_full_time)/((1.0*x_num*y_num*z_num)/1000000.0));
    
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
ExtraPartCellData<U> sep_neigh_filter(PartCellData<uint64_t>& pc_data,ExtraPartCellData<U>& input_data,std::vector<U>& filter,std::vector<bool> filter_dir = {true,true,true}){
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
        return filter_output;
    }
    
    
    timer.verbose_flag = false;

    
    timer.start_timer("neigh_cell_comp");
    
    for(int dir = 0;dir < 3;dir++){

        if(filter_dir[dir]) {

            for (uint64_t i = pc_data.depth_min; i <= pc_data.depth_max; i++) {
                //loop over the resolutions of the structure
                const unsigned int x_num_ = pc_data.x_num[i];
                const unsigned int z_num_ = pc_data.z_num[i];
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
                for (z_ = 0; z_ < z_num_; z_++) {
                    //both z and x are explicitly accessed in the structure
                    curr_key = 0;

                    pc_data.pc_key_set_z(curr_key, z_);
                    pc_data.pc_key_set_depth(curr_key, i);


                    for (x_ = 0; x_ < x_num_; x_++) {

                        pc_data.pc_key_set_x(curr_key, x_);

                        const size_t offset_pc_data = x_num_ * z_ + x_;

                        const size_t j_num = pc_data.data[i][offset_pc_data].size();

                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for (j_ = 0; j_ < j_num; j_++) {

                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            node_val_pc = pc_data.data[i][offset_pc_data][j_];

                            if (!(node_val_pc & 1)) {
                                //Indicates this is a particle cell node
                                //y_coord++;

                                pc_data.pc_key_set_j(curr_key, j_);

                                pc_data.get_neighs_axis(curr_key, node_val_pc, neigh_cell_keys, dir);

                                U temp_int = 0;
                                int counter = 0;

                                U curr_int = input_data.get_val(curr_key);

                                U accum_int = filter[1] * curr_int;

                                if (curr_int > 0) {
                                    //(+direction)
                                    //loop over the nieghbours
                                    for (int n = 0; n < neigh_cell_keys.neigh_face[2 * dir].size(); n++) {
                                        // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                        uint64_t neigh_key = neigh_cell_keys.neigh_face[2 * dir][n];

                                        if (neigh_key > 0) {
                                            //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                            temp_int += input_data.get_val(neigh_key);
                                            counter++;
                                        }


                                    }

                                    if (temp_int == 0) {
                                        temp_int = curr_int;
                                    } else {
                                        temp_int = temp_int / counter;
                                    }
                                    counter = 0;

                                    accum_int += temp_int * filter[2];

                                    temp_int = 0;
                                    //(-direction)
                                    //loop over the nieghbours
                                    for (int n = 0; n < neigh_cell_keys.neigh_face[2 * dir + 1].size(); n++) {
                                        // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                        uint64_t neigh_key = neigh_cell_keys.neigh_face[2 * dir + 1][n];

                                        if (neigh_key > 0) {
                                            //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
                                            temp_int += input_data.get_val(neigh_key);
                                            counter++;
                                        }
                                    }

                                    if (temp_int == 0) {
                                        temp_int = curr_int;
                                    } else {
                                        temp_int = temp_int / counter;
                                    }

                                    accum_int += temp_int * filter[0];

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


    }
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1);
    
    std::cout << "Seperable Smoothing Filter: " << time << std::endl;

    //swap em back
    std::swap(filter_output,input_data);

    return filter_output;
    
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
template<typename U>
ExtraPartCellData<U> filter_apr_by_slice(PartCellStructure<float,uint64_t>& pc_struct,std::vector<U>& filter,bool debug = false){

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    Mesh_data<U> slice;

    Part_timer timer;
    timer.verbose_flag = true;

    std::vector<U> filter_d = shift_filter(filter);


    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new,pc_data);

    ExtraPartCellData<U> filter_output;
    filter_output.initialize_structure_parts(part_new.particle_data);

    ExtraPartCellData<U> filter_input;
    filter_input.initialize_structure_parts(part_new.particle_data);

    filter_input.data = part_new.particle_data.data;

    int num_slices = 0;


    timer.start_timer("filter all dir");

    for(int dir = 0; dir <1;++dir) {

        if (dir != 1) {
            num_slices = pc_struct.org_dims[1];
        } else {
            num_slices = pc_struct.org_dims[2];
        }

        if (dir == 0) {
            //yz
            slice.initialize(y_vec.org_dims[0], y_vec.org_dims[2], 1, 0);
        } else if (dir == 1) {
            //xy
            slice.initialize(y_vec.org_dims[1], y_vec.org_dims[0], 1, 0);

        } else if (dir == 2) {
            //zy
            slice.initialize(y_vec.org_dims[2], y_vec.org_dims[0], 1, 0);

        }

        //set to zero
        set_zero_minus_1(filter_output);

        int i = 0;
#pragma omp parallel for default(shared) private(i) firstprivate(slice) schedule(guided)
        for (i = 0; i < num_slices; ++i) {
            interp_slice(slice, y_vec, filter_input, dir, i);

            filter_slice(filter,filter_d,filter_output,slice,y_vec,dir,i);
        }

        //std::swap(filter_input,filter_output);
    }

    timer.stop_timer();

    // std::swap(filter_input,filter_output);

    if(debug == true) {

        Mesh_data<float> img;

        interp_img(img, pc_data, part_new, filter_output);

        for (int k = 0; k < img.mesh.size(); ++k) {
            img.mesh[k] = 10 * fabs(img.mesh[k]);
        }

        debug_write(img, "filter_img");
    }

    return filter_output;




};


template<typename U>
ExtraPartCellData<U> filter_apr_input_img(Mesh_data<U>& input_img,PartCellStructure<float,uint64_t>& pc_struct,std::vector<U>& filter,AnalysisData& analysis_data,float num_repeats = 1,bool debug = false){

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    Mesh_data<U> slice;

    Part_timer timer;
    timer.verbose_flag = false;

    std::vector<U> filter_d = shift_filter(filter);

    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new,pc_data);

    ExtraPartCellData<U> filter_output;
    filter_output.initialize_structure_parts(part_new.particle_data);

    ExtraPartCellData<U> filter_input;
    filter_input.initialize_structure_parts(part_new.particle_data);

    filter_input.data = part_new.particle_data.data;


    timer.start_timer("filter y");

    //Y Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_mesh_dir(input_img,y_vec,filter_output,filter_input,filter,filter_d,0);
    }

    timer.stop_timer();

    float time_y = (timer.t2 - timer.t1)/num_repeats;

    //std::swap(filter_input,filter_output);

    timer.start_timer("filter x");

    //X Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_mesh_dir(input_img,y_vec,filter_output,filter_input,filter,filter_d,1);
    }

    timer.stop_timer();

    float time_x = (timer.t2 - timer.t1)/num_repeats;


    //std::swap(filter_input,filter_output);

    timer.start_timer("filter z");

    //X Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_mesh_dir(input_img,y_vec,filter_output,filter_input,filter,filter_d,2);
    }

    timer.stop_timer();

    float time_z = (timer.t2 - timer.t1)/num_repeats;

    analysis_data.add_float_data("part_filter_input_y",time_y);
    analysis_data.add_float_data("part_filter_input_x",time_x);
    analysis_data.add_float_data("part_filter_input_z",time_z);

    analysis_data.add_float_data("part_filter_input_all",time_y + time_x + time_z);


    std::cout << "Part Filter Input Image: " << (time_y + time_x + time_z) << std::endl;


    if(debug == true) {

        Mesh_data<float> img;

        interp_img(img, pc_data, part_new, filter_output);

        for (int k = 0; k < img.mesh.size(); ++k) {
            img.mesh[k] = 10 * fabs(img.mesh[k]);
        }

        debug_write(img, "filter_img_input");
    }

    return filter_output;

};



template<typename U>
ExtraPartCellData<U> filter_apr_by_slice(PartCellStructure<float,uint64_t>& pc_struct,std::vector<U>& filter,AnalysisData& analysis_data,float num_repeats = 1,bool debug = false){

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    Mesh_data<U> slice;

    Part_timer timer;
    timer.verbose_flag = false;

    std::vector<U> filter_d = shift_filter(filter);

    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new,pc_data);

    ExtraPartCellData<U> filter_output;
    filter_output.initialize_structure_parts(part_new.particle_data);

    ExtraPartCellData<U> filter_input;
    filter_input.initialize_structure_parts(part_new.particle_data);

    filter_input.data = part_new.particle_data.data;


    timer.start_timer("filter y");

    //Y Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_dir(y_vec,filter_output,filter_input,filter,filter_d,0);
    }

    timer.stop_timer();

    float time_y = (timer.t2 - timer.t1)/num_repeats;

    //std::swap(filter_input,filter_output);

    timer.start_timer("filter x");

    //X Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_dir(y_vec,filter_output,filter_input,filter,filter_d,1);
    }

    timer.stop_timer();

    float time_x = (timer.t2 - timer.t1)/num_repeats;


    //std::swap(filter_input,filter_output);

    timer.start_timer("filter z");

    //X Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_dir(y_vec,filter_output,filter_input,filter,filter_d,2);
    }

    timer.stop_timer();

    float time_z = (timer.t2 - timer.t1)/num_repeats;

    analysis_data.add_float_data("part_filter_y",time_y);
    analysis_data.add_float_data("part_filter_x",time_x);
    analysis_data.add_float_data("part_filter_z",time_z);

    analysis_data.add_float_data("part_filter_all",time_y + time_x + time_z);


    std::cout << "Part Filter: " << (time_y + time_x + time_z) << std::endl;


    if(debug == true) {

        Mesh_data<float> img;

        interp_img(img, pc_data, part_new, filter_output);

        for (int k = 0; k < img.mesh.size(); ++k) {
            img.mesh[k] = 10 * fabs(img.mesh[k]);
        }

        debug_write(img, "filter_img");
    }

    return filter_output;




};

template<typename U>
ExtraPartCellData<U> filter_apr_by_slice_mult(PartCellStructure<float,uint64_t>& pc_struct,std::vector<U>& filter_y,std::vector<U>& filter_x,std::vector<U>& filter_z,AnalysisData& analysis_data,float num_repeats = 1,bool debug = false){

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    Mesh_data<U> slice;

    Part_timer timer;
    timer.verbose_flag = false;

    std::vector<U> filter_d_y = shift_filter(filter_y);
    std::vector<U> filter_d_x = shift_filter(filter_x);
    std::vector<U> filter_d_z = shift_filter(filter_z);

    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new,pc_data);

    ExtraPartCellData<U> filter_output;
    filter_output.initialize_structure_parts(part_new.particle_data);

    ExtraPartCellData<U> filter_input;
    filter_input.initialize_structure_parts(part_new.particle_data);

    filter_input.data = part_new.particle_data.data;

    timer.start_timer("filter y");

    //Y Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_dir(y_vec,filter_output,filter_input,filter_y,filter_d_y,0);
    }

    timer.stop_timer();



    float time_y = (timer.t2 - timer.t1)/num_repeats;

    std::swap(filter_input,filter_output);

    set_zero(filter_output);

    timer.start_timer("filter x");

    //X Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_dir(y_vec,filter_output,filter_input,filter_x,filter_d_x,1);
    }

    timer.stop_timer();

    float time_x = (timer.t2 - timer.t1)/num_repeats;



    std::swap(filter_input,filter_output);

    set_zero(filter_output);

    timer.start_timer("filter z");

    //Z Direction

    for (int i = 0; i < num_repeats ; ++i) {
        filter_apr_dir(y_vec,filter_output,filter_input,filter_z,filter_d_z,2);
    }

    timer.stop_timer();

    float time_z = (timer.t2 - timer.t1)/num_repeats;


    analysis_data.add_float_data("part_filter_y",time_y);
    analysis_data.add_float_data("part_filter_x",time_x);
    analysis_data.add_float_data("part_filter_z",time_z);

    analysis_data.add_float_data("part_filter_all",time_y + time_x + time_z);


    std::cout << "Part Filter: " << (time_y + time_x + time_z) << std::endl;


    if(debug == true) {

        Mesh_data<float> img;

        interp_img(img, pc_data, part_new, filter_output);

        for (int k = 0; k < img.mesh.size(); ++k) {
            img.mesh[k] = 10 * fabs(img.mesh[k]);
        }

        debug_write(img, "filter_img");
    }

    return filter_output;


};


template<typename U>
ExtraPartCellData<U> sep_neigh_grad(PartCellData<uint64_t>& pc_data,ExtraPartCellData<U>& input_data,int dir,U delta = 1.0f){
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



    timer.verbose_flag = false;


    timer.start_timer("neigh_cell_comp");




    for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){


        float h_depth = pow(2,pc_data.depth_max - i);

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

                        U accum_int = 0;

                        if(curr_int > 0){
                            //(+direction)
                            //loop over the nieghbours
                            float h = h_depth;

                            for(int n = 0; n < neigh_cell_keys.neigh_face[2*dir].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[2*dir][n];

                                uint64_t ndepth = pc_data.pc_key_get_depth(neigh_key);
                                h = .5*pow(2,pc_data.depth_max - ndepth) + .5*h_depth;

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

                            accum_int += .5*(temp_int - curr_int)/(h*delta);

                            temp_int = 0;
                            //(-direction)
                            //loop over the nieghbours
                            for(int n = 0; n < neigh_cell_keys.neigh_face[2*dir+1].size();n++){
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[2*dir+1][n];

                                uint64_t ndepth = pc_data.pc_key_get_depth(neigh_key);
                                h = .5*pow(2,pc_data.depth_max - ndepth) + .5*h_depth;

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

                            accum_int += .5*(curr_int - temp_int)/(h*delta);

                        }

                        filter_output.get_val(curr_key) = accum_int;



                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node

                    }

                }

            }

        }
    }





    timer.stop_timer();

    float time = (timer.t2 - timer.t1);

    //std::cout << "Seperable Smoothing Filter: " << time << std::endl;
    return filter_output;


}
template<typename T>
ExtraPartCellData<T> adaptive_smooth(PartCellData<uint64_t>& pc_data,ExtraPartCellData<T>& input_data,int num_tap,std::vector<T> filter = {.1,.8,.1}){
    //
    //  Bevan Cheeseman 2017
    //  Adaptive Smooth
    //

    ExtraPartCellData<float> output_data;

    output_data = sep_neigh_filter(pc_data,input_data,filter);

    for (int i = 0; i < (num_tap); ++i) {
        output_data = sep_neigh_filter(pc_data,output_data,filter);
    }

    return output_data;

}
template<typename T>
ExtraPartCellData<T> adaptive_grad(PartCellData<uint64_t>& pc_data,ExtraPartCellData<T>& input_data,int dir,std::vector<T> delta = {1,1,1}){


    if(dir < 3 && dir >= 0){
        // just compute one direction gradient

        ExtraPartCellData<float> output =  sep_neigh_grad(pc_data,input_data,dir,delta[dir]);

        return output;

    } else {
        // compute gradient magnitude

        ExtraPartCellData<float> output =  sep_neigh_grad(pc_data,input_data,0,delta[0]);

        ExtraPartCellData<float> temp =  sep_neigh_grad(pc_data,input_data,1,delta[1]);

        //square the two gradients
        output = transform_parts(output,square<float>);
        temp = transform_parts(temp,square<float>);

        //then add them together
        transform_parts(output,temp,std::plus<float>());

        temp =  sep_neigh_grad(pc_data,input_data,2,delta[2]);

        temp = transform_parts(temp,square<float>);
        transform_parts(output,temp,std::plus<float>());

        output = transform_parts(output,square_root<float>);

        return output;
    }



}
template<typename T>
ExtraPartCellData<std::array<float,3>> adaptive_gradient_normal(PartCellStructure<float,uint64_t>& pc_struct,std::vector<T> delta = {1,1,1}){


    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    ExtraPartCellData<float> particle_data;

    part_new.create_particles_at_cell_structure(particle_data);


    // compute gradient magnitude

    ExtraPartCellData<float> grad_y =  sep_neigh_grad(pc_data,particle_data,0,delta[0]);

    ExtraPartCellData<float> grad_x =  sep_neigh_grad(pc_data,particle_data,1,delta[1]);

    ExtraPartCellData<float> mag;
    ExtraPartCellData<float> temp;

    //square the two gradients
    temp = transform_parts(grad_y,square<float>);
    mag = transform_parts(grad_x,square<float>);

    //then add them together
    transform_parts(mag,temp,std::plus<float>());

    ExtraPartCellData<float> grad_z =  sep_neigh_grad(pc_data,particle_data,2,delta[2]);

    temp = transform_parts(grad_z,square<float>);
    transform_parts(mag,temp,std::plus<float>());

    ExtraPartCellData<std::array<float,3>> normal_vec;

    //first add the layers
    normal_vec.depth_max = particle_data.depth_max;
    normal_vec.depth_min = particle_data.depth_min;

    normal_vec.z_num.resize(normal_vec.depth_max+1);
    normal_vec.x_num.resize(normal_vec.depth_max+1);

    normal_vec.data.resize(normal_vec.depth_max+1);

    normal_vec.org_dims = particle_data.org_dims;

    for(uint64_t i = normal_vec.depth_min;i <= normal_vec.depth_max;i++){
        normal_vec.z_num[i] = particle_data.z_num[i];
        normal_vec.x_num[i] = particle_data.x_num[i];
        normal_vec.data[i].resize(normal_vec.z_num[i]*normal_vec.x_num[i]);

        for(int j = 0;j < particle_data.data[i].size();j++){
            normal_vec.data[i][j].resize(particle_data.data[i][j].size());
        }

    }

    int z_,x_,j_,y_;

    for(uint64_t depth = (normal_vec.depth_min);depth <= normal_vec.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = normal_vec.x_num[depth];
        const unsigned int z_num_ = normal_vec.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < normal_vec.data[depth][pc_offset].size(); ++j_) {

                    normal_vec.data[depth][pc_offset][j_][0] = grad_y.data[depth][pc_offset][j_]/mag.data[depth][pc_offset][j_];
                    normal_vec.data[depth][pc_offset][j_][1] = grad_x.data[depth][pc_offset][j_]/mag.data[depth][pc_offset][j_];
                    normal_vec.data[depth][pc_offset][j_][2] = grad_z.data[depth][pc_offset][j_]/mag.data[depth][pc_offset][j_];

                }

            }
        }
    }



    return normal_vec;




}


template<typename T>
Mesh_data<float> compute_grad(Mesh_data<T> gt_image,std::vector<float> delta = {1,1,1}){
    //
    //
    //  Computes Gradient Magnitude Using FD
    //
    //

    AnalysisData analysis_data;
    float num_repeats = 1;

    std::vector<float> filter_f = {-.5,0,.5};
    std::vector<float> filter_b = {0,1,0};


    std::vector<float> filter_y = {(float)-.5/delta[0],0,(float).5/delta[0]};
    std::vector<float> filter_x = {(float)-.5/delta[1],0,(float).5/delta[1]};
    std::vector<float> filter_z = {(float)-.5/delta[2],0,(float).5/delta[2]};

    Mesh_data<float> gt_image_f;
    gt_image_f.initialize(gt_image.y_num,gt_image.x_num,gt_image.z_num,0);
    std::copy(gt_image.mesh.begin(),gt_image.mesh.end(),gt_image_f.mesh.begin());

    Mesh_data<float> gt_output;

    Mesh_data<float> temp;
    temp.initialize(gt_image.y_num,gt_image.x_num,gt_image.z_num,0);

    //first y direction
    gt_output =  pixel_filter_full_mult(gt_image_f,filter_y,filter_b,filter_b,num_repeats,analysis_data);

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        temp.mesh[k] += pow(gt_output.mesh[k],2);
    }

    //first x direction
    gt_output =  pixel_filter_full_mult(gt_image_f,filter_b,filter_x,filter_b,num_repeats,analysis_data);

    //std::transform (temp.mesh.begin(), temp.mesh.end(), gt_output.mesh.begin(), gt_output.mesh.begin(), std::plus<float>());

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        temp.mesh[k] += pow(gt_output.mesh[k],2);
    }

    //first z direction
    gt_output =  pixel_filter_full_mult(gt_image_f,filter_b,filter_b,filter_z,num_repeats,analysis_data);

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        temp.mesh[k] += pow(gt_output.mesh[k],2);
    }

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        gt_output.mesh[k] = sqrt(temp.mesh[k]);
    }

    return gt_output;


}
template<typename U,typename V,typename T>
ExtraPartCellData<U> compute_normalized_grad_mag(PartCellStructure<V,T>& pc_struct,int num_taps,std::vector<float> delta){
    //
    //
    //
    //

    ExtraPartCellData<U> norm_grad;


    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {1,2,3};
    std::vector<unsigned int> status_offsets_max = {1,2,3};

    std::vector<float> filter = {.0125,.975,.0125};



    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    ExtraPartCellData<float> particle_data;

    part_new.create_particles_at_cell_structure(particle_data);

    ExtraPartCellData<float> smoothed_parts;

    if(num_taps > 0) {
        smoothed_parts = adaptive_smooth(pc_data, particle_data, num_taps, filter);
    } else {
        smoothed_parts = particle_data;
    }

    ExtraPartCellData<float> gradient_mag = adaptive_grad(pc_data,smoothed_parts,3,delta);

    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;


    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,0,0);

    transform_parts(adaptive_max,adaptive_min,std::minus<float>());

    float min_th = 5;
    float set_val = 10000000;

    threshold_parts(adaptive_max,min_th,set_val,std::less<float>());

    ExtraPartCellData<float> local_scale =  convert_cell_to_part(pc_struct,adaptive_max);

    convert_from_old_structure(adaptive_max,pc_struct,pc_data,local_scale,false);

    part_new.create_particles_at_cell_structure(local_scale,adaptive_max);

    transform_parts(gradient_mag,local_scale,std::divides<float>());




    return gradient_mag;


}





#endif
