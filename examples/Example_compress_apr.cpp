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

-compress_type number (1 or 2) (1 - WNL compression, only variance stabalization step (Default), 2 - variance stabalization and x,y,z prediction (note slower for ~30% compression gain)
-quantization_level (Default 1: higher increasing the loss nature of the WNL compression aproach)
-compress_level (the IO uses BLOSC for lossless compression of the APR, this can be set from 1-9, where higher increases the compression level. Note, this can come at a significant time increase.)

e.g. Example_compress_apr -i nuc_apr.h5 -d /Test/Input_examples/ -compress_type 1

Note: fine grained parameters can be tuned within the file, to play with lossless compression level, method used, and other parameters.

)";


#include <algorithm>
#include <iostream>

#include "Example_compress_apr.h"
#include "io/TiffUtils.hpp"
#include "data_structures/APR/ParticleData.hpp"
#include "io/APRFile.hpp"
#include "numerics/APRReconstruction.hpp"

int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR apr;

    //read file
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);

    ParticleData<uint16_t>parts;
    aprFile.read_particles(apr,"particle_intensities",parts);

    aprFile.close();

    apr.parameters.input_dir = options.directory;

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    ParticleData<uint16_t> symbols;

    APRFile compFile;
    compFile.open(options.directory,"WRITE");

    compFile.aprCompress.set_quantization_factor(options.quantization_level); //set this to adjust the compression factor for WNL
    compFile.aprCompress.set_compression_type(options.compress_type);

    std::cerr << "This will be updated soon, the options just need to be exposed through the APRFILE" << std::endl;

    //compress the APR and write to disk
    timer.start_timer("compress and write");

    compFile.write_apr(apr);
    compFile.write_particles(apr,"comp_parts",parts);

    timer.stop_timer();

    //float time_write = (float) timer.timings.back();

//    //read the APR and decompress
//    timer.start_timer("read and decompress");
//    apr.read_apr(options.directory + name + "_compress_apr.h5");
//    timer.stop_timer();
//
//    float time_read = (float) timer.timings.back();
//
//    float original_pixel_image_size = (2.0f*apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2))/(1000000.0);
//    std::cout << std::endl;
//    std::cout << std::endl;
//    std::cout << "Original image size: " << original_pixel_image_size << " MB" << std::endl;
//
//    float apr_compressed_file_size = fileSizeInfo.total_file_size;
//
//    std::cout << "Compressed (Lossy - WNL) APR: " << apr_compressed_file_size << " MB" << std::endl;
//    std::cout << "Compression Ratio: " << original_pixel_image_size/apr_compressed_file_size << std::endl;
//    std::cout << std::endl;
//    std::cout << std::endl;
//
//    std::cout << "Effective Datarate Write (by original image size): " << original_pixel_image_size/time_write << " MB*/s" << std::endl;
//    std::cout << "Effective Datarate Read (by original image size): " << original_pixel_image_size/time_read << " MB*/s" << std::endl;
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//
//    //writes the piece-wise constant reconstruction of the APR to file for comparison
//    if(options.output_tiff) {
//        PixelData<uint16_t> img;
//        APRReconstruction::interp_img(apr,img, parts);
//        std::string output = options.directory + name + "_compress.tif";
//        TiffUtils::saveMeshAsTiff(output, img);
//    }
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

    if(result.compress_type > 2){

        std::cerr << "Invalid Compression setting (0,1 or 2)" << std::endl;
        exit(1);
    }


    if(command_option_exists(argv, argv + argc, "-quantization_level"))
    {
        result.quantization_level =std::stof(std::string(get_command_option(argv, argv + argc, "-quantization_level")));
    }

    if(command_option_exists(argv, argv + argc, "-compress_level"))
    {
        result.compress_level = (unsigned int)std::stoi(std::string(get_command_option(argv, argv + argc, "-compress_level")));
    }

    if(command_option_exists(argv, argv + argc, "-output_tiff"))
    {
        result.output_tiff = true;
    }


    return result;

}


