//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(
APR Compression example:

Example of performing additional compression on the APR intensities (APR adaptation of the within noise lossy compression from Balázs et al. 2017, A real-time compression library for
 microscopy images)

Usage:

(using *.apr output of Example_get_apr)

Example_compress_apr -i input_apr_hdf5 -d input_directory

Note: inpute file path is 'input_apr_hdf5 + input_directory'
      writes the compressed APR to a file *_compressed.apr in the input directory
      optionally writes a reconstructed TIFF image to *_compressed.tif (if the flag -output_tiff is given)

Optional:

-compress_type number (1 or 2) (1 - WNL compression, only variance stabalization step (Default), 2 - variance stabalization and x,y,z prediction (note slower for ~30% compression gain)
-quantization_level (Default 1: higher increasing the loss nature of the WNL compression aproach)
-compress_level (the IO uses BLOSC for lossless compression of the APR, this can be set from 1-9, where higher increases the compression level. Note, this can come at a significant time increase.)

e.g. Example_compress_apr -i nuclei.apr -d /Test/Input_examples/ -compress_type 1

Note: fine grained parameters can be tuned within the file, to play with lossless compression level, method used, and other parameters.

)";


#include <algorithm>
#include <iostream>

#include "Example_compress_apr.h"
#include "io/TiffUtils.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "io/APRFile.hpp"
#include "numerics/APRReconstruction.hpp"

int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    std::string name = options.input;
    name.erase(name.end()-4,name.end());
    std::string output_file_name = options.directory + name + "_compressed.apr";

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
    aprFile.read_particles(apr,parts);

    aprFile.close();


    APRFile compFile;
    compFile.open(output_file_name,"WRITE");

    parts.compressor.set_quantization_factor(options.quantization_level); //set this to adjust the compression factor for WNL
    parts.compressor.set_compression_type(options.compress_type);

    //set the background to the minimum
    float background = *std::min_element(parts.data.begin(),parts.data.end());
    parts.compressor.set_background(background);

    compFile.write_apr(apr);

    //compress the APR and write to disk
    timer.start_timer("compress and write");

    compFile.write_particles("particles",parts);

    timer.stop_timer();

    float time_write = timer.timings.back();


    float original_pixel_image_size = (2.0f*apr.org_dims(0)*apr.org_dims(1)*apr.org_dims(2))/(1000000.0f);
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Original image size: " << original_pixel_image_size << " MB" << std::endl;


    float apr_compressed_file_size = compFile.current_file_size_MB();
    compFile.close();

    timer.start_timer("decompress and read");
    compFile.open(output_file_name,"READ");
    compFile.read_particles(apr,"particles",parts);
    compFile.close();
    timer.stop_timer();

    float time_read = timer.timings.back();

    std::cout << "Compressed (Lossy - WNL) APR: " << apr_compressed_file_size << " MB" << std::endl;
    std::cout << "Compression Ratio: " << original_pixel_image_size/apr_compressed_file_size << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Effective Datarate Write (by original image size): " << original_pixel_image_size/time_write << " MB*/s" << std::endl;
    std::cout << "Effective Datarate Read (by original image size): " << original_pixel_image_size/time_read << " MB*/s" << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;

    //writes the piece-wise constant reconstruction of the APR to file for comparison
    if(options.output_tiff) {
        PixelData<uint16_t> img;
        APRReconstruction::reconstruct_constant(apr,img, parts);
        std::string output = options.directory + name + "_compressed.tif";
        TiffUtils::saveMeshAsTiff(output, img);
    }
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
    return nullptr;
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


