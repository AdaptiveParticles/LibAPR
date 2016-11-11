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
    
  
    PartCellParent<uint64_t> parent_cells(pc_struct);
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    
    std::cout << "Number cells: " << num_cells << std::endl;
    std::cout << "Number parts: " << num_parts << std::endl;
    
    // FIND POINT X,Y,Z  in structure
    
    uint64_t y = 49;
    uint64_t x = 33;
    uint64_t z = 25;
    
    uint64_t factor =pow(2,pc_struct.depth_max + 1 - pc_struct.depth_min);
    //calculate on min layer
    uint64_t y_min = y/factor;
    uint64_t x_min = x/factor;

    uint64_t z_min = z/factor;
    
    uint64_t j = parent_cells.neigh_info.get_j_from_y(x_min,z_min,pc_struct.depth_min,y_min);
    
    
    
    uint64_t curr_key = 0;
    
    parent_cells.neigh_info.pc_key_set_x(curr_key,x_min);
    parent_cells.neigh_info.pc_key_set_z(curr_key,z_min);
    parent_cells.neigh_info.pc_key_set_j(curr_key,j);
    parent_cells.neigh_info.pc_key_set_depth(curr_key,pc_struct.depth_min);
    
    std::vector<uint64_t> children_keys;
    std::vector<uint64_t> children_flag;
    uint64_t index;
    uint64_t y_curr;
    uint64_t x_curr;
    uint64_t z_curr;
    
    uint64_t child_y;
    uint64_t child_x;
    uint64_t child_z;
    uint64_t child_depth;
    
    for(int i = pc_struct.depth_min; i < pc_struct.depth_max; i++){
        
        parent_cells.get_children_keys(curr_key,children_keys,children_flag);
        
        factor =pow(2,pc_struct.depth_max + 1 - i - 1);
        //calculate on min layer
        y_curr = y/factor;
        x_curr = x/factor;
        
        z_curr = z/factor;
        
        index = 4*(z_curr&1) + 2*(x_curr&1) + (y_curr&1);
        
        curr_key = children_keys[index];
        
        parent_cells.get_child_coordinates_cell(children_keys,index,y_curr/2,child_y,child_x,child_z,child_depth);
        
        curr_key = children_keys[index];
        
        if (children_flag[index] == 1){
            //found the cell;
            break;
            
        }
    }
    
    uint64_t check = pc_struct.pc_data.get_val(curr_key);
    
    int stop = 1;

}


