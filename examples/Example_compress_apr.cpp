//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(
APR Compression example:

Example of performing additional compression on the APR intensities (APR adaptation of the within noise lossy compression from Balázs et al. 2017, A real-time compression library for
 microscopy images)

Usage:

Example_compress_apr -i input_image_tiff -d input_directory

Optional:

-compress_type number (1 or 2) (1 - WNL compression (Default), 2 - prediction step with lossless, potential rounding error)

e.g. Example_compress_apr -i nuc_apr.h5 -d /Test/Input_examples/ -compress_type 2

Note: fine grained parameters can be tuned within the file, to play with lossless compression level, method used, and other parameters.

)";


#include <algorithm>
#include <iostream>

#include "Example_compress_apr.h"
#include "io/TiffUtils.hpp"

int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    apr.parameters.input_dir = options.directory;

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    APRCompress<uint16_t> comp;
    ExtraParticleData<uint16_t> symbols;

    comp.set_quantization_factor(1);
    comp.set_compression_type(2);

    timer.start_timer("compress");
    apr.write_apr(options.directory ,name + "_compress",comp,BLOSC_ZSTD,1,2);
    timer.stop_timer();

    timer.start_timer("decompress");
    apr.read_apr(options.directory + name + "_compress_apr.h5");
    timer.stop_timer();

    MeshData<uint16_t> img;
    apr.interp_img(img,apr.particles_intensities);
    std::string output = options.directory + name + "_compress.tif";
    TiffUtils::saveMeshAsTiff(output, img);
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

        if(command_option_exists(argv, argv + argc, "-d"))
        {
            result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
        }

        if(command_option_exists(argv, argv + argc, "-compress_type"))
        {
            result.compress_type = (unsigned int)std::stoi(std::string(get_command_option(argv, argv + argc, "-compress_type")));
        }

        if(result.compress_type > 2 || result.compress_type == 0){

            std::cerr << "Invalid Compression setting (1 or 2)" << std::endl;
            exit(1);
        }

        return result;

    }


