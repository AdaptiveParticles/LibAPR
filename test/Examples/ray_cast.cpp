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
#include "../../src/numerics/ray_cast.hpp"

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
    if(command_option_exists(argv, argv + argc, "-s"))
    {
        result.stats = std::string(get_command_option(argv, argv + argc, "-s"));
        get_image_stats(part_rep.pars, result.directory, result.stats);
        result.stats_file = true;
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
    
    //single_ray_parrallel(pc_struct);
    
    int projection_type = 1;
    int direction = 3;
    
    Part_timer timer;
    
    timer.start_timer("parrallel projection content");
    timer.verbose_flag = true;

    proj_par proj_pars;

    proj_pars.direction = 4;
    proj_pars.proj_type = 1;
    proj_pars.Ip_th = 125;
    proj_pars.start_th = 5;
    proj_pars.status_th = 20;
    
    proj_pars.avg_flag = true;
    
    //multi_ray_parrallel(pc_struct,proj_pars);
    
    timer.stop_timer();


    //multi_ray_parrallel_raster(pc_struct,proj_pars);

    //multi_ray_parrallel_raster_alt(pc_struct,proj_pars);

    //multi_ray_parrallel_raster_mesh(pc_struct,proj_pars);

    multi_ray_parrallel_raster_alt_d(pc_struct,proj_pars);

   // multi_ray_parrallel_raster_alt_d_off(pc_struct,proj_pars);

    //gen_raster_cast(pc_struct,proj_pars);
}


