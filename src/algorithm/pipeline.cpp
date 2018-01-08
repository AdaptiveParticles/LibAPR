
#include <algorithm>
#include <iostream>

#include "pipeline.h"
#include "../data_structures/meshclass.h"
#include "../io/readimage.h"

#include "gradient.hpp"
#include "../data_structures/particle_map.hpp"
#include "benchmarks/development/Tree/Content.hpp"
#include "benchmarks/development/Tree/LevelIterator.hpp"
#include "benchmarks/development/Tree/Tree.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "level.hpp"
#include "../io/writeimage.h"
#include "../io/write_parts.h"
#include "../io/partcell_io.h"

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
        std::cerr << "Usage: \"pipeline -i inputfile [-t] [-s example_name -d stats_directory] [-o outputfile]\"" << std::endl;
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
        result.stats_directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-s"))
    {
        result.stats = std::string(get_command_option(argv, argv + argc, "-s"));
        get_image_stats(part_rep.pars, result.stats_directory, result.stats);
        result.stats_file = true;
    }
    if(command_option_exists(argv, argv + argc, "-l"))
    {
        part_rep.pars.lambda = (float)std::atof(get_command_option(argv, argv + argc, "-l"));
        if(part_rep.pars.lambda == 0.0){
            std::cerr << "Lambda can't be zero" << std::endl;
            exit(3);
        }
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

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    {
        Mesh_data<uint16_t> input_image;

        load_image_tiff(input_image, options.input);

        gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

        input_image_float = input_image.to_type<float>();

        // After this block, input_image will be freed.
    }

    if(!options.stats_file) {
        // defaults

        part_rep.pars.dy = part_rep.pars.dx = part_rep.pars.dz = 1;
        part_rep.pars.psfx = part_rep.pars.psfy = part_rep.pars.psfz = 1;
        part_rep.pars.rel_error = 0.1;
        part_rep.pars.var_th = 0;
        part_rep.pars.var_th_max = 0;
    }

    Part_timer t;
    t.verbose_flag = true;


    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<float>> down_sampled_images;

    // variables for tree
    std::vector<uint64_t> tree_mem(gradient.y_num * gradient.x_num * gradient.z_num * 1.25, 0);
    std::vector<Content> contents(gradient.y_num * gradient.x_num * gradient.z_num, {0});

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");
    
    part_map.downsample(input_image_float);
    
    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);
    
    part_rep.timer.start_timer("get_gradient_3D");
    get_gradient_3D(part_rep, input_image_float, gradient);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("get_variance_3D");
    get_variance_3D(part_rep, input_image_float, variance);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("get_level_3D");
    get_level_3D(variance, gradient, part_rep, part_map, temp);
    part_rep.timer.stop_timer();
    
    
    // free memory (not used anymore)
    std::vector<float>().swap( gradient.mesh );
    std::vector<float>().swap( variance.mesh );
    

    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();
    
 
    part_rep.timer.start_timer("estimate_part_intensity");
    
    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);
    
    Tree<float> tree(part_map, tree_mem, contents);
    part_rep.timer.stop_timer();

    t.stop_timer();


    part_rep.timer.stop_timer();
    
    std::cout << "Size of data structure: " << tree.get_tree_size() << std::endl;
    std::cout << "Number of parts : " << tree.get_content_size() << std::endl;
    std::cout << "Ratio : " << ((float)tree.get_tree_size())/((float)tree.get_content_size()) << std::endl;
    
    std::cout << "Size of data structure (MB): " << tree.get_tree_size()*8.0/1000000.0 << std::endl;
    std::cout << "Size of parts (MB): " << tree.get_content_size()*4.0/1000000.0 << std::endl;
    
    std::cout << "Size of image (MB): " << part_rep.org_dims[0]*part_rep.org_dims[1]*part_rep.org_dims[2]*4.0/1000000.0 << std::endl;
    
    
    //output
    std::string save_loc = options.output;
    std::string file_name = options.stats;
    

    
    //testing sparse format
    
    part_rep.timer.start_timer("compute new structure");
    PartCellStructure<float,uint64_t> pcell_test(part_map);
    part_rep.timer.stop_timer();
    
    write_apr_full_format(pcell_test,save_loc,file_name);
    write_apr_pc_struct(pcell_test,save_loc,file_name);

    
    PartCellStructure<float,uint64_t> pc_read;
    read_apr_pc_struct(pc_read,save_loc + file_name + "_pcstruct_part.h5");

    std::cout << pc_read.get_number_cells() << " " << pcell_test.get_number_cells() << std::endl;
    std::cout << pc_read.get_number_parts() << " " << pcell_test.get_number_parts() << std::endl;
    
    write_apr_full_format(pc_read,save_loc,file_name + "2");
    
}


