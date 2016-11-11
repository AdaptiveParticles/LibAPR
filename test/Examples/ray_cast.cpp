#include <algorithm>
#include <iostream>

#include "compute_cells.h"
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
#include "../../src/data_structures/Tree/PartCellParent.hpp"
#include "../utils.h"

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
    
    // COMPUTATIONS
    PartCellStructure<float,uint64_t> pc_struct;
    
    //output
    std::string file_name = options.directory + options.input;
    
    read_apr_pc_struct(pc_struct,file_name);
    
  
    PartCellParent<uint64_t> Parent_cells(pc_struct);
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    
    std::cout << "Number cells: " << num_cells << std::endl;
    std::cout << "Number parts: " << num_parts << std::endl;
    
    //run test
    
    //initialize looping vars
    uint64_t x_;
    uint64_t y_coord;
    uint64_t z_;
    uint64_t j_;
    uint64_t node_val_part;
    uint64_t status;
    
    Particle_map<float> particle_map;
    
    particle_map.k_max = pc_struct.depth_max;
    particle_map.k_min = pc_struct.depth_min;
    
    particle_map.layers.resize(pc_struct.depth_max+1);
    particle_map.downsampled.resize(pc_struct.depth_max+2);
    
    particle_map.downsampled[pc_struct.depth_max + 1].x_num = pc_struct.org_dims[1];
    particle_map.downsampled[pc_struct.depth_max + 1].y_num = pc_struct.org_dims[0];
    particle_map.downsampled[pc_struct.depth_max + 1].z_num = pc_struct.org_dims[2];
    particle_map.downsampled[pc_struct.depth_max + 1].mesh.resize(pc_struct.org_dims[1]*pc_struct.org_dims[0]*pc_struct.org_dims[2]);
    
    std::cout << "DIM1: " << pc_struct.org_dims[1] << std::endl;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        particle_map.layers[i].mesh.resize(x_num_*z_num_*y_num_,0);
        particle_map.layers[i].x_num = x_num_;
        particle_map.layers[i].y_num = y_num_;
        particle_map.layers[i].z_num = z_num_;
        
        particle_map.downsampled[i].x_num = x_num_;
        particle_map.downsampled[i].y_num = y_num_;
        particle_map.downsampled[i].z_num = z_num_;
        particle_map.downsampled[i].mesh.resize(x_num_*z_num_*y_num_,0);
        
        // First create the particle map
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //get the index gap node
                        y_coord++;
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        particle_map.layers[i].mesh[offset_p_map + y_coord] = status;
                        
                    } else {
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
        
    }
    
    
    //intensity set up
    // Set the intensities
    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
        
        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
            particle_map.downsampled[depth].mesh[i] = (uint16_t)i;
        }
        
    }
    
    
    
    
    compare_sparse_rep_neighpart_with_part_map(particle_map,pc_struct);
    
    parent_structure_test(pc_struct);

}


