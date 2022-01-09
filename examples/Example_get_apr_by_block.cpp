//
// Created by joel on 03.11.20.
//

const char* usage = R"(
Form the APR form image: Takes an uint16_t input tiff image and forms the APR and saves it as hdf5. The hdf5 output of this program
can be used with the other apr examples, and also viewed with HDFView.

Usage:

Auto parameters are currently not supported in this APR pipeline. Minimal usage:

Example_get_apr_by_block -i input_image_tiff -d input_directory [-o name_of_output -od output_directory] -Ip_th intensity_threshold
                         -sigma_th intensity_scale_threshold -grad_th gradient_threshold -lambda lambda_value

Parameter explanations:

-Ip_th intensity_threshold            (will ignore areas of image below this threshold, useful for removing camera artifacts or auto-flouresence)
-sigma_th intensity_scale_threshold   (the computed local intensity scale is clipped from below to this value)
-grad_th gradient_threshold           (gradients lower than this threshold are set to 0)
-lambda lambda_value                  (directly set the value of the gradient smoothing parameter lambda (reasonable range 0.1-10, default: 3)

Additional settings controlling the memory usage:

-z_block_size value     (number of z-slices to process in each tile. default: 128)
-z_ghost value          (number of ghost slices to use in both directions for the blocked APR pipeline. default: 16)
-z_ghost_sampling value (number of ghost slices to use in both directions for the blocked particle sampling. default: 64)

Note that these parameters affect the solution. To ensure consistency between z-blocks, sufficiently many ghost slices
must be used. The exact influence of this has not yet been studied.

)";

#include <algorithm>
#include <iostream>
#include "ConfigAPR.h"
#include "Example_get_apr_by_block.hpp"
#include "io/APRFile.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/APR/APR.hpp"
#include "algorithm/APRConverterBatch.hpp"


int runAPR(cmdLineOptions options) {

    APR apr;
    APRConverterBatch<uint16_t> aprConverter;

    //read in the command line options into the parameters file
    aprConverter.par.Ip_th = options.Ip_th;
    aprConverter.par.rel_error = options.rel_error;
    aprConverter.par.lambda = options.lambda;
    aprConverter.par.mask_file = options.mask_file;
    aprConverter.par.sigma_th = options.sigma_th;
    aprConverter.par.neighborhood_optimization = options.neighborhood_optimization;
    aprConverter.par.output_steps = options.output_steps;
    aprConverter.par.grad_th = options.grad_th;

    //where things are
    aprConverter.par.input_image_name = options.input;
    aprConverter.par.input_dir = options.directory;
    aprConverter.par.name = options.output;
    aprConverter.par.output_dir = options.output_dir;

    aprConverter.fine_grained_timer.verbose_flag = false;
    aprConverter.method_timer.verbose_flag = false;
    aprConverter.computation_timer.verbose_flag = false;
    aprConverter.allocation_timer.verbose_flag = false;
    aprConverter.total_timer.verbose_flag = true;

    aprConverter.z_block_size = options.z_block_size;
    aprConverter.ghost_z = options.z_ghost;
    aprConverter.verbose = true;

    aprConverter.set_sparse_pulling_scheme(false); // use sparse particle cell tree in pulling scheme?
    aprConverter.set_generate_linear(true);        // generate linear or random access data structure?

    APRTimer timer(true);
    timer.start_timer("Get APR by block");
    bool success = aprConverter.get_apr(apr);
    timer.stop_timer();

    if(success){
        float num_pix = (float)apr.org_dims(0) * (float)apr.org_dims(1) * (float)apr.org_dims(2);
        float num_parts = apr.total_number_particles();
        float cr = num_pix / num_parts;
        std::cout << "APR Conversion successful! CR = " << cr << std::endl;

        timer.start_timer("Sample particles by block");
        ParticleData<uint16_t> parts;
        parts.sample_parts_from_img_blocked(apr, options.directory + options.input, options.z_block_size, options.z_ghost_sampling);
        timer.stop_timer();

        //output
        std::string save_loc = options.output_dir;
        std::string file_name = options.output;

        std::cout << std::endl;
        float original_pixel_image_size = (2.0f* apr.org_dims(0)* apr.org_dims(1)* apr.org_dims(2))/1000000.0f;
        std::cout << "Original image size: " << original_pixel_image_size << " MB" << std::endl;

        //write the APR and particles to hdf5 file
        timer.start_timer("write output to file");
        APRFile aprFile;
        aprFile.open(save_loc + file_name + ".apr");

        aprFile.write_apr(apr, 0, "t", options.store_tree);
        aprFile.write_particles("particles", parts);
        timer.stop_timer();

        float apr_file_size = aprFile.current_file_size_MB();
        float compression_ratio = original_pixel_image_size / apr_file_size;
        float computational_ratio = (1.0f* apr.org_dims(0)* apr.org_dims(1)* apr.org_dims(2))/(1.0f*apr.total_number_particles());

        std::cout << std::endl;
        std::cout << "Computational Ratio (Pixels/Particles): " << computational_ratio << std::endl;
        std::cout << "Lossy Compression Ratio: " << compression_ratio << std::endl;
        std::cout << std::endl;

    } else {
        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
    }
    return 0;
}


int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options = read_command_line_options(argc, argv);
    return runAPR(options);
}



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
        std::cerr << argv[0] << std::endl;
        std::cerr << "APR version " << ConfigAPR::APR_VERSION << std::endl;
        std::cerr << "Short usage: \"" << argv[0] << " -i inputfile [-d directory] [-o outputfile]\"" << std::endl;

        std::cerr << usage << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-od"))
    {
        result.output_dir = std::string(get_command_option(argv, argv + argc, "-od"));
    } else {
        result.output_dir = result.directory;
    }

    if(command_option_exists(argv, argv + argc, "-lambda"))
    {
        result.lambda = std::stof(std::string(get_command_option(argv, argv + argc, "-lambda")));
    }

    if(command_option_exists(argv, argv + argc, "-Ip_th"))
    {
        result.Ip_th = std::stof(std::string(get_command_option(argv, argv + argc, "-Ip_th")));
    }

    if(command_option_exists(argv, argv + argc, "-grad_th"))
    {
        result.grad_th = std::stof(std::string(get_command_option(argv, argv + argc, "-grad_th")));
    }

    if(command_option_exists(argv, argv + argc, "-sigma_th"))
    {
        result.sigma_th = std::stof(std::string(get_command_option(argv, argv + argc, "-sigma_th")));
    }

    if(command_option_exists(argv, argv + argc, "-rel_error"))
    {
        result.rel_error = std::stof(std::string(get_command_option(argv, argv + argc, "-rel_error")));
    }

    if(command_option_exists(argv, argv + argc, "-z_block_size"))
    {
        result.z_block_size = std::stoi(std::string(get_command_option(argv, argv + argc, "-z_block_size")));
    }

    if(command_option_exists(argv, argv + argc, "-z_ghost"))
    {
        result.z_ghost = std::stoi(std::string(get_command_option(argv, argv + argc, "-z_ghost")));
    }

    if(command_option_exists(argv, argv + argc, "-z_ghost_sampling"))
    {
        result.z_ghost_sampling = std::stoi(std::string(get_command_option(argv, argv + argc, "-z_ghost_sampling")));
    }

    if(command_option_exists(argv, argv + argc, "-mask_file"))
    {
        result.mask_file = std::string(get_command_option(argv, argv + argc, "-mask_file"));
    }

    if(command_option_exists(argv, argv + argc, "-neighborhood_optimization_off"))
    {
        result.neighborhood_optimization = false;

    }

    if(command_option_exists(argv, argv + argc, "-output_steps"))
    {
        result.output_steps = true;
    }

    if(command_option_exists(argv, argv + argc, "-store_tree"))
    {
        result.store_tree = true;
    }

    return result;
}