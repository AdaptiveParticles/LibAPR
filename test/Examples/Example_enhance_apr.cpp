//
// Created by cheesema on 25/02/17.
//

#include "Example_enhance_apr.hpp"

#include <algorithm>
#include <iostream>

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
#include "../../src/numerics/enhance_parts.hpp"



bool command_option_exists_filter(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option_filter(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions_filter read_command_line_options_filter(int argc, char **argv, Part_rep& part_rep){

    cmdLineOptions_filter result;

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

    if(command_option_exists(argv, argv + argc, "-org_image"))
    {
        result.original_file = std::string(get_command_option(argv, argv + argc, "-org_image"));
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-gt"))
    {
        result.gt = std::string(get_command_option(argv, argv + argc, "-gt"));
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

    cmdLineOptions_filter options = read_command_line_options_filter(argc, argv, part_rep);

    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);

    Mesh_data<float> output_img;

    compute_guided_apr(output_img,pc_struct,part_rep);



}