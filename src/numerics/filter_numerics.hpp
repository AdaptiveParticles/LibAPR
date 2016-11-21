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
#include "../data_structures/Tree/PartCellOffset.hpp"

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




template<typename U>
void convolution_filter_y(PartCellStructure<U,uint64_t>& pc_struct){
    //
    //
    //  Loops per particle (Layer, Status vs. Non)
    //
    //
    
    std::vector<float> filter;
    
    
    uint64_t filter_offset = 3;
    filter.resize(filter_offset*2 +1,0);
    
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
    std::vector<float> temp_vec_depth;
    
    temp_vec.resize(filter.size());
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
    
    timer.verbose_flag = true;
    timer.start_timer("y filter loop");
    
    PartCellOffset<uint64_t> layer_plus;
    layer_plus.set_offsets(0,0,filter_offset,-1); //one layer below
    
    layer_plus.set_new_depth(depth,pc_struct); //intialize for the depth
    
#pragma omp parallel for default(shared) private(z_,x_,j_,y_coord,y_coord_p,node_val_part,curr_key,status,part_offset) firstprivate(temp_vec,temp_vec_depth,layer_plus) if(z_num_*x_num_ > 100)
    for(z_ = 0;z_ < z_num_;z_++){
        //both z and x are explicitly accessed in the structure
        curr_key = 0;
        
        pc_struct.pc_data.pc_key_set_z(curr_key,z_);
        pc_struct.pc_data.pc_key_set_depth(curr_key,depth);
        
        for(x_ = 0;x_ < x_num_;x_++){
            
            //shift layer
            layer_plus.set_new_xz(x_,z_,pc_struct);
            
            pc_struct.pc_data.pc_key_set_x(curr_key,x_);
            
            const size_t offset_pc_data = x_num_*z_ + x_;
            
            const size_t j_num = pc_struct.pc_data.data[depth][offset_pc_data].size();
            
            //the y direction loop however is sparse, and must be accessed accordinagly
            for(j_ = 0;j_ < j_num;j_++){
                
                //particle cell node value, used here as it is requried for getting the particle neighbours
                node_val_part = pc_struct.part_data.access_data.data[depth][offset_pc_data][j_];
                
                if (!(node_val_part&1)){
                    //Indicates this is a particle cell node
                    
                    status = pc_struct.part_data.access_node_get_status(node_val_part);
                    
                    y_coord++;
                    
                    y_coord_p = 2*y_coord;
                    
                    if(status == SEED){
                        //these two operations need to be done
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        //seed offset accoutns for which (x,z) you are doing
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part) + seed_offset;
                        
                        //there are two particles (part_offset) and part_offset + 1;
                        
                        //first rotate forward the filter array
                        iterate_temp_vec(temp_vec,temp_vec_depth);
                        
                        layer_plus.incriment_y_and_update(y_coord,pc_struct,temp_vec,temp_vec_depth);
                        
                        
                        
                        //then update forward the up and down iterators and array
                        
                        //then compute the filter.
                        
                        
                    }
                    
                    
                } else {
                    uint64_t y_init = y_coord;
                    
                    y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                    y_coord--;
                    
                    //will need to initialize things here..
                    y_init = std::max(y_init,y_coord - filter_offset);
                    
                    for(uint64_t q = y_init;q < y_coord + (filter_offset-1);q++){
                        
                        layer_plus.incriment_y_and_update(q,pc_struct,temp_vec,temp_vec_depth);
                        
                    }
                    
                }
                
            }
            
        }
        
    }

    timer.stop_timer();


}
#endif