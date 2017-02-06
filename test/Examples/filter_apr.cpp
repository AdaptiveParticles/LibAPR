#include <algorithm>
#include <iostream>

#include "filter_apr.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellBase.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/data_structures/Tree/ParticleDataNew.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"

#include "../../test/utils.h"

#include "../../src/numerics/misc_numerics.hpp"
#include "../../src/numerics/filter_numerics.hpp"



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
    
    
    //////////////////////////////////
    //
    //  Different access and filter test examples
    //
    //////////////////////////////////
    
    //set up some new structures used in this test
    AnalysisData analysis_data;

    float num_repeats = 1;

    //Get neighbours (linear)

    //particles
    //
   // particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);

    //particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);

    //particle_linear_neigh_access_alt_1(pc_struct);

    //pixels
   // pixels_linear_neigh_access(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],num_repeats,analysis_data);


    //Get neighbours (random access)

    //particle_random_access(pc_struct,analysis_data);

   // pixel_neigh_random(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],analysis_data);


    // Filtering

    uint64_t filter_offset = 3;

    //apr_filter_full(pc_struct,filter_offset,num_repeats,analysis_data);

  //  pixel_filter_full(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],filter_offset,num_repeats,analysis_data);

    //new_filter_part(pc_struct,filter_offset,num_repeats,analysis_data);

    int num = 800;
    int dir = 2;

    //interp_slice<float,float>(pc_struct,pc_struct.part_data.particle_data,dir,num);

    //get_slices<float>(pc_struct);

    Mesh_data<uint16_t> output;

    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("full interp");

    //pc_struct.interp_parts_to_pc(output,pc_struct.part_data.particle_data);

    timer.stop_timer();

    std::vector<float> filter;

    filter.resize(filter_offset*2 + 1,1.0/(filter_offset*2 + 1));

    filter_apr_by_slice<float>(pc_struct,filter);


}


