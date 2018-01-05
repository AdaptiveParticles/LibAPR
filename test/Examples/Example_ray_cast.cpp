#include <algorithm>
#include <iostream>

#include "Example_ray_cast.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"
#include "../../src/data_structures/Tree/PartCellParent.hpp"
#include "../../src/numerics/ray_cast.hpp"
#include "../../src/numerics/filter_numerics.hpp"
#include "../../src/numerics/misc_numerics.hpp"

#include "../../src/data_structures/APR/APR.hpp"

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

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile [-t] [-s statsfile -d directory] [-o outputfile]\"" << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }



    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }



    if(command_option_exists(argv, argv + argc, "-org_file"))
    {
        result.org_file = std::string(get_command_option(argv, argv + argc, "-org_file"));
    }

    return result;

}


int main(int argc, char **argv) {

    // INPUT PARSING
    
    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // APR datastructure
    APR<float> apr;

    //read file
    apr.read_apr(file_name);

    Part_timer timer;

    /////////////////
    ///
    ///  Raycast Parameters
    ////////////////

    proj_par proj_pars;

    proj_pars.theta_0 = -3.14; //start
    proj_pars.theta_final = 3.14; //stop radians
    proj_pars.radius_factor = .98f; //radius scaling
    proj_pars.theta_delta = .1f; //steps
    proj_pars.scale_z = apr.pars.aniso; //z scaling

    proj_pars.name = apr.name;

    Mesh_data<float> output;

    /////////////
    ///
    ///  Compute APR raycast
    ///
    /////////////

    apr_raycast(apr,apr.particles_int,proj_pars,output,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});

    //////////////
    ///
    ///  Write the output to tiff
    ///
    //////////////////

    std::string output_loc = options.directory + apr.name + "_ray_cast_views.tif";

    output.write_image_tiff_uint16(output_loc);

}
