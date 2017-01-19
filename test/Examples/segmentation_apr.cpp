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
    
    //Part Segmentation

    ExtraPartCellData<uint8_t> seg_parts;
    
    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,1,1,1,1,1,1,1,0,0};
    
    //nuclei
    std::array<uint64_t,10> parameters_mem = {100,1,2,2,2,1,1,1,1,1};
    
    calc_graph_cuts_segmentation(pc_struct, seg_parts,parameters_nuc);
    
    Mesh_data<uint8_t> seg_mesh;
    
    //calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc);
    
    //Now we will view the output by creating the binary image implied by the segmentation
    
    Mesh_data<uint8_t> seg_img;
    
    pc_struct.interp_parts_to_pc(seg_img,seg_parts);
    
    debug_write(seg_img,"segmentation_mask");
    
    //debug_write(seg_mesh,"segmentation_mesh_mask");
    
    
//    interp_depth_to_mesh(seg_img,pc_struct);
//    
//    debug_write(seg_img,"k_mask");
//    
//    interp_status_to_mesh(seg_img,pc_struct);
//    
//    debug_write(seg_img,"status_mask");
    
}


