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

#define LOWER_RESOLUTION -1
#define SAME_RESOLUTION 0
#define HIGHER_RESOLUTION 1

#define ACTIVE 1
#define INACTIVE 0

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "filter_help/FilterOffset.hpp"
#include "filter_help/FilterLevel.hpp"

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
template<typename U,typename V>
void convolution_filter_y(PartCellStructure<U,uint64_t>& pc_struct,ExtraPartCellData<V>& filter_output){
    //
    //
    //  Loops per particle (Layer, Status vs. Non)
    //
    //
    
    std::vector<float> filter;
    
    int filter_offset = 1;
    filter.resize(filter_offset*2 +1,1);
    
    ///////////////
    //
    //
    //
    //////////////
    
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    
    ////////////////////////
    //
    //  Seed loop (max resolution) example
    //
    /////////////////////////
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    timer.start_timer("y filter loop");
    
    //doing seed level (four different particle paths)
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
        
        for(uint64_t depth = (pc_struct.depth_max); depth >= (pc_struct.depth_min); depth--){
            
            //initialize layer iterators
            FilterLevel<uint64_t,float> curr_level;
            curr_level.set_new_depth(depth,pc_struct);
            
            curr_level.initialize_temp_vecs(filter,pc_struct);
            
            const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
           
            bool layer_active = INACTIVE;
            if(depth > (pc_struct.depth_min)){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_plus(LOWER_RESOLUTION,layer_active);
            
            layer_plus.set_offsets(0,0,filter_offset,-1); //one layer below
            layer_plus.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if(depth > (pc_struct.depth_min+1)){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_plus_2(LOWER_RESOLUTION,layer_active);
            layer_plus_2.set_offsets(0,0,filter_offset,-2); //one layer below
            layer_plus_2.set_new_depth(depth,pc_struct); //intialize for the depth
            
            //always set
            FilterOffset<uint64_t,float> layer_equal(SAME_RESOLUTION,ACTIVE);
            layer_equal.set_offsets(0,0,filter_offset,0); //one layer below
            layer_equal.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if(depth < pc_struct.depth_max){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_minus0(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus1(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus2(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus3(HIGHER_RESOLUTION,layer_active);
            
            layer_minus0.set_offsets(0,0,filter_offset,1); //one layer below
            layer_minus0.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus1.set_offsets(1,0,filter_offset,1); //one layer below
            layer_minus1.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus2.set_offsets(0,1,filter_offset,1); //one layer below
            layer_minus2.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus3.set_offsets(1,1,filter_offset,1); //one layer below
            layer_minus3.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if (depth == pc_struct.depth_max){
                
#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(layer_plus,layer_equal,curr_level,layer_plus_2) if(z_num_*x_num_ > 100)
                for(z_ = 0;z_ < z_num_;z_++){
                    //both z and x are explicitly accessed in the structure
                    
                    for(x_ = 0;x_ < x_num_;x_++){
                        
                        //NEED TO FIX THESE THEY HAVE THE WRONG INPUT
                        //shift layers
                        layer_plus.set_new_xz(x_,z_,pc_struct);
                        layer_plus_2.set_new_xz(x_,z_,pc_struct);
                        layer_equal.set_new_xz(x_,z_,pc_struct);
                      
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        
                        
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                //Indicates this is a particle cell node
                                curr_level.update_cell(pc_struct);
                                
                                curr_level.iterate_temp_vecs();
                                
                                //update and incriment
                                layer_plus.incriment_y_and_update(pc_struct,curr_level);
                                layer_plus_2.incriment_y_and_update(pc_struct,curr_level);
                                layer_equal.incriment_y_and_update(pc_struct,curr_level);
                     
                                
                                curr_level.compute_filter(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    //iterate the vectors
                                    curr_level.iterate_temp_vecs();
                                    
                                    //update them with new values
                                    layer_equal.incriment_y_and_update(pc_struct,curr_level);
                                    
                          
                                    
                                    //compute the filter
                                    curr_level.compute_filter(filter_output);
                                }
                                
                                
                            } else {
                                // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                                
                                uint64_t y_init = curr_level.y_global;
                                
                                curr_level.update_gap(pc_struct);
                                
                                //will need to initialize things here..
                                y_init = std::max(y_init,curr_level.y_global - filter_offset);
                                
                                for(uint64_t q = y_init;q < curr_level.y_global + (filter_offset-1);q++){
                                    
                                    curr_level.iterate_temp_vecs();
                                    layer_plus.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_equal.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_plus_2.incriment_y_and_update(q,pc_struct,curr_level);
                                    
                                   
                                }
                            }
                        }
                    }
                }
            } else {
                
#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(layer_plus,layer_equal,curr_level,layer_plus_2,layer_minus0,layer_minus1,layer_minus2,layer_minus3) if(z_num_*x_num_ > 100)
                for(z_ = 0;z_ < z_num_;z_++){
                    //both z and x are explicitly accessed in the structure
                    
                    for(x_ = 0;x_ < x_num_;x_++){
                        
                        //NEED TO FIX THESE THEY HAVE THE WRONG INPUT
                        //shift layers
                        layer_plus.set_new_xz(x_,z_,pc_struct);
                        layer_plus_2.set_new_xz(x_,z_,pc_struct);
                        layer_equal.set_new_xz(x_,z_,pc_struct);
                        layer_minus0.set_new_xz(x_,z_,pc_struct);
                        layer_minus1.set_new_xz(x_,z_,pc_struct);
                        layer_minus2.set_new_xz(x_,z_,pc_struct);
                        layer_minus3.set_new_xz(x_,z_,pc_struct);
                        
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        
                        
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                //Indicates this is a particle cell node
                                curr_level.update_cell(pc_struct);
                                
                                curr_level.iterate_temp_vecs();
                                
                                //update and incriment
                                layer_plus.incriment_y_and_update(pc_struct,curr_level);
                                layer_plus_2.incriment_y_and_update(pc_struct,curr_level);
                                layer_equal.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus0.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus1.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus2.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus3.incriment_y_and_update(pc_struct,curr_level);
                                
                                curr_level.compute_filter(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    //iterate the vectors
                                    curr_level.iterate_temp_vecs();
                                    
                                    //update them with new values
                                    layer_equal.incriment_y_and_update(pc_struct,curr_level);
                                    
                                    layer_minus0.incriment_y_and_update(pc_struct,curr_level);
                                    layer_minus1.incriment_y_and_update(pc_struct,curr_level);
                                    layer_minus2.incriment_y_and_update(pc_struct,curr_level);
                                    layer_minus3.incriment_y_and_update(pc_struct,curr_level);
                                    
                                    //compute the filter
                                    curr_level.compute_filter(filter_output);
                                }
                                
                                
                            } else {
                                // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                                
                                uint64_t y_init = curr_level.y_global;
                                
                                curr_level.update_gap(pc_struct);
                                
                                //will need to initialize things here..
                                y_init = std::max(y_init,curr_level.y_global - filter_offset);
                                
                                for(uint64_t q = y_init;q < curr_level.y_global + (filter_offset-1);q++){
                                    
                                    curr_level.iterate_temp_vecs();
                                    layer_plus.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_equal.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_plus_2.incriment_y_and_update(q,pc_struct,curr_level);
                                    
                                    layer_minus0.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_minus1.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_minus2.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_minus3.incriment_y_and_update(q,pc_struct,curr_level);
                                }
                            }
                        }
                    }
                }
                
                
                
            }
            
        }
        
    }
    
    
    
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Particle Filter Size: " << pc_struct.get_number_parts() << " took: " << time << std::endl;
    
}


template<typename U,typename V>
void convolution_filter_y_old(PartCellStructure<U,uint64_t>& pc_struct,ExtraPartCellData<V>& filter_output){
    //
    //
    //  Loops per particle (Layer, Status vs. Non)
    //
    //
    
    std::vector<float> filter;
    
    
    int filter_offset = 1;
    filter.resize(filter_offset*2 +1,1);
    
    std::rotate(filter.begin(),filter.begin() + 1,filter.end());
    
    ///////////////
    //
    //
    //
    //////////////
    
    
    //initialize variables required
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    
    uint64_t part_offset;
    uint64_t status;
    
    uint64_t y_coord;
    
    uint64_t depth = pc_struct.depth_max;
    
    std::vector<float> temp_vec;
    
    std::vector<float> temp_vec1;
    std::vector<float> temp_vec2;
    std::vector<float> temp_vec3;
    
    
    std::vector<float> temp_vec_depth;
    
    temp_vec.resize(filter.size());
    temp_vec1.resize(filter.size());
    temp_vec2.resize(filter.size());
    temp_vec3.resize(filter.size());
    
    temp_vec_depth.resize(filter.size());
    
    ////////////////////////
    //
    //  Seed loop (max resolution) example
    //
    /////////////////////////
    
    Part_timer timer;
    
    //doing seed level (four different particle paths)
    
    uint64_t seed_offset = 0;
    
    uint64_t y_coord_p = 0;
    
    const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
    const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
    
    timer.verbose_flag = false;
    timer.start_timer("y filter loop");
    
    FilterOffset<uint64_t,float> layer_plus(LOWER_RESOLUTION,ACTIVE);
    layer_plus.set_offsets(0,0,filter_offset,-1); //one layer below
    layer_plus.set_new_depth(depth,pc_struct); //intialize for the depth
    
    FilterOffset<uint64_t,float> layer_plus_2(LOWER_RESOLUTION,ACTIVE);
    layer_plus_2.set_offsets(0,0,filter_offset,-2); //one layer below
    layer_plus_2.set_new_depth(depth,pc_struct); //intialize for the depth
    
    FilterOffset<uint64_t,float> layer_equal(HIGHER_RESOLUTION,ACTIVE);
    layer_equal.set_offsets(0,0,filter_offset,0); //one layer below
    layer_equal.set_new_depth(depth,pc_struct); //intialize for the depth
    
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(z_,x_,j_,y_coord,y_coord_p,node_val_part,curr_key,status,part_offset) firstprivate(temp_vec,temp_vec_depth,layer_plus,layer_equal,temp_vec1,temp_vec2,temp_vec3) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                //shift layers
                layer_plus.set_new_xz(x_,z_,pc_struct);
                layer_plus_2.set_new_xz(x_,z_,pc_struct);
                layer_equal.set_new_xz(x_,z_,pc_struct);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[depth][offset_pc_data].size();
                
                y_coord = 0;
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_part = pc_struct.part_data.access_data.data[depth][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //Indicates this is a particle cell node
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        
                        y_coord++;
                        
                        y_coord_p = 2*y_coord; // on seed level
                        
                        //these two operations need to be done
                        
                        //seed offset accoutns for which (x,z) you are doing
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part) + seed_offset*(status==SEED);
                        
                        iterate_temp_vec(temp_vec);
                        iterate_temp_vec(temp_vec1);
                        iterate_temp_vec(temp_vec2);
                        iterate_temp_vec(temp_vec3);
                        
                        layer_plus.incriment_y(y_coord_p,pc_struct);
                        
                        layer_plus.update_temp_vec(pc_struct,temp_vec,0);
                        layer_plus.update_temp_vec(pc_struct,temp_vec1,0);
                        layer_plus.update_temp_vec(pc_struct,temp_vec2,0);
                        layer_plus.update_temp_vec(pc_struct,temp_vec3,0);
                        
                        
                        layer_plus_2.incriment_y(y_coord_p,pc_struct);
                        
                        layer_plus_2.update_temp_vec(pc_struct,temp_vec,0);
                        layer_plus_2.update_temp_vec(pc_struct,temp_vec1,0);
                        layer_plus_2.update_temp_vec(pc_struct,temp_vec2,0);
                        layer_plus_2.update_temp_vec(pc_struct,temp_vec3,0);
                        
                        
                        for(uint64_t p = 0;p < 1 + (status==SEED);p++){
                            
                            y_coord_p += p;
                            part_offset += p;
                            //first rotate forward the filter array
                            if(p >0){
                                iterate_temp_vec(temp_vec);
                                iterate_temp_vec(temp_vec1);
                                iterate_temp_vec(temp_vec2);
                                iterate_temp_vec(temp_vec3);
                            }
                            layer_equal.incriment_y(y_coord_p,pc_struct);
                            layer_equal.update_temp_vec(pc_struct,temp_vec,0);
                            layer_equal.update_temp_vec(pc_struct,temp_vec1,2);
                            layer_equal.update_temp_vec(pc_struct,temp_vec2,4);
                            layer_equal.update_temp_vec(pc_struct,temp_vec3,6);
                            
                            if(status ==SEED){
                                //perform the filter
                                
                                for(uint64_t f = 0;f < filter.size();f++){
                                    filter_output.data[depth][offset_pc_data][part_offset] += temp_vec[f]*filter[f];
                                }
                                
                                for(uint64_t f = 0;f < filter.size();f++){
                                    filter_output.data[depth][offset_pc_data][part_offset + 2] += temp_vec1[f]*filter[f];
                                }
                                
                                for(uint64_t f = 0;f < filter.size();f++){
                                    filter_output.data[depth][offset_pc_data][part_offset + 4] += temp_vec2[f]*filter[f];
                                }
                                
                                for(uint64_t f = 0;f < filter.size();f++){
                                    filter_output.data[depth][offset_pc_data][part_offset + 6] += temp_vec3[f]*filter[f];
                                }
                            } else {
                                
                                for(uint64_t f = 0;f < filter.size();f++){
                                    filter_output.data[depth][offset_pc_data][part_offset] += temp_vec[f]*filter[f];
                                }
                                
                            }
                            
                        }
                        
                        
                    } else {
                        //skip node have to then initialize the arrays
                        uint64_t y_init = y_coord;
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                        
                        //will need to initialize things here..
                        y_init = std::max(y_init,y_coord - filter_offset);
                        
                        for(uint64_t q = 2*y_init;q < 2*y_coord + (filter_offset-1);q++){
                            
                            
                            
                            
                            iterate_temp_vec(temp_vec,temp_vec_depth);
                            layer_plus.incriment_y_and_update(q,pc_struct,temp_vec,temp_vec_depth);
                            layer_equal.incriment_y_and_update(q,pc_struct,temp_vec,temp_vec_depth);
                            layer_plus_2.incriment_y_and_update(q,pc_struct,temp_vec,temp_vec_depth);
                        }
                        
                    }
                    
                }
                
            }
            
        }
        
    }
    
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Particle Filter Size OLD: " << pc_struct.get_number_parts() << " took: " << time << std::endl;
    
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



#endif