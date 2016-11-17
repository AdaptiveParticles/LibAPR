#include <algorithm>
#include <iostream>

#include "segmentation_apr.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellBase.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"
#include "../../src/numerics/parent_numerics.hpp"
#include "../../src/numerics/misc_numerics.hpp"
#include "../../src/numerics/graph_cut_seg.hpp"

bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep){
    
    cmdLineOptions result;
    
    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile -d directory [-t] [-o outputfile]\"" << std::endl;
        exit(1);
    }
    
    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    
    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    
    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }
    
    if(command_option_exists(argv, argv + argc, "-t"))
    {
        part_rep.timer.verbose_flag = true;
    }
    
    return result;
    
}

int main(int argc, char **argv) {
    
    Part_rep part_rep;
    
    // INPUT PARSING
    
    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);
    
    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;
    
    // Filename
    std::string file_name = options.directory + options.input;
    
    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);
    
    uint64_t x_;
    uint64_t z_;
    uint64_t p;
    uint64_t curr_key;
    uint64_t j_;
    
    uint64_t node_val_part;
    uint64_t node_val;
    uint64_t counter = 0;
    uint64_t counter_j = 0;
    uint64_t counter_c = 0;
    
    uint64_t status = 0;
    uint64_t part_offset = 0;
    
    uint64_t num_cells=0;
    
    
    uint64_t num_cells_ = 0;
    uint64_t num_parts_ = 0;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
        //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
        
        // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status) reduction(+:num_cells_,num_parts_)
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.part_data.access_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        //in this loop there is a cell
                        num_cells_++;
                        status = pc_struct.pc_data.get_status(node_val);
                        //determine how many particles in the cell
                        if(status==SEED){
                            num_parts_=num_parts_ + 8;
                        } else {
                            num_parts_= num_parts_ + 1;
                        }
                        
                    }
                }
                
            }
        }
    }
    

    
    int stopw = 1;
    
    
    for(uint64_t i = pc_struct.part_data.particle_data.depth_min;i <=  pc_struct.part_data.particle_data.depth_max;i++){
        
        size_t x_num_ =  pc_struct.part_data.particle_data.x_num[i];
        size_t z_num_ =  pc_struct.part_data.particle_data.z_num[i];
        
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                const size_t j_num_p = pc_struct.part_data.particle_data.data[i][offset_pc_data].size();
                const size_t j_num_a = pc_struct.part_data.access_data.data[i][offset_pc_data].size();
                
                
                if(j_num != j_num_a){
                    int stop =1;
                }
                
                if(counter != counter_j){
                    int stop = 1;
                }
                
                counter_j += j_num_p;
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if((node_val_part&1) != (node_val&1)){
                        int stop = 1;
                    }
                    
                    
                    if (!(node_val_part&1)){
                        //Indicates this is a particle cell node
                        num_cells++;
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        uint64_t global_part_index;
                        
                        if(status==SEED){
                            counter_c = counter_c + 8;
                        } else {
                            counter_c = counter_c + 1;
                        }
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            
                            
                            counter++;
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    
    
    //Part Segmentation

    ExtraPartCellData<uint8_t> seg_parts;
    
    calc_graph_cuts_segmentation(pc_struct, seg_parts);
    
    //Now we will view the output by creating the binary image implied by the segmentation
    
    Mesh_data<uint8_t> seg_img;
    
    pc_struct.interp_parts_to_pc(seg_img,seg_parts);
    
    debug_write(seg_img,"segmentation_mask");
    
    interp_depth_to_mesh(seg_img,pc_struct);
    
    debug_write(seg_img,"k_mask");
    
    interp_status_to_mesh(seg_img,pc_struct);
    
    debug_write(seg_img,"status_mask");
    
}


