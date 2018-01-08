#include <algorithm>
#include <iostream>

#include "segmentation_apr.h"
#include "src/data_structures/Mesh/meshclass.h"
#include "benchmarks/development/old_io/readimage.h"

#include "benchmarks/development/old_structures/particle_map.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "benchmarks/development/Tree/ParticleDataNew.hpp"
#include "benchmarks/development/old_algorithm/level.hpp"
#include "benchmarks/development/old_io/writeimage.h"
#include "benchmarks/development/old_io/write_parts.h"
#include "benchmarks/development/old_io/partcell_io.h"
#include "src/numerics/parent_numerics.hpp"
#include "src/numerics/misc_numerics.hpp"
#include "src/numerics/apr_segment.hpp"

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
    
    //Part Segmentation
    
    ExtraPartCellData<uint8_t> binary_mask;
    
    threshold_part(pc_struct,binary_mask,200);
    
    
    ExtraPartCellData<uint16_t> component_label;
    
   //calculate the connected component
    
    calc_connected_component(pc_struct,binary_mask,component_label);
    
    //calc_connected_component_alt(pc_struct,binary_mask,component_label);
    
    //Now we will view the output by creating the binary image implied by the segmentation
    
    Mesh_data<uint8_t> binary_img;
    Mesh_data<uint16_t> comp_img;
//    
    pc_struct.interp_parts_to_pc(binary_img,binary_mask);
    pc_struct.interp_parts_to_pc(comp_img,component_label);
//    
    debug_write(binary_img,"binary_mask");
    debug_write(comp_img,"comp_mask");
    
   


}


