#include <algorithm>
#include <iostream>

#include "ray_cast.h"
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
    
    parent_structure_test(pc_struct);
    
    // FIND POINT X,Y,Z  in structure
//
//    for(uint64_t x = 0; x < 1024; x++) {
//        for(uint64_t y = 0; y < 1024; y++) {
//            for(uint64_t z = 0; z < 100; z++) {
//                uint64_t pc_key = parent_cells.find_partcell(x, y, z, pc_struct);
//                uint64_t check = pc_struct.pc_data.get_val(pc_key);
//
//                std::cout << check << std::endl;
//            }
//        }
//    }

    std::cout << "Propagating through APR now..." << std::endl;

    uint64_t x = 12, y = 25, z = 10;
    uint64_t dir_x = 2, dir_y = -1, dir_z = 1;
#define MAX_DIST 100
    for(unsigned int step = 0; step < MAX_DIST; step++) {

        uint64_t pc_key = parent_cells.find_partcell(x+step*dir_x, y+step*dir_y, z+step*dir_z, pc_struct);
        uint64_t check = pc_struct.pc_data.get_val(pc_key);

        std::cout << check << std::endl;
    }

    std::cout << std::endl;

    part_rep.timer.start_timer("find cell");
    
    find_part_cell_test(pc_struct);
    
    part_rep.timer.stop_timer();
}


