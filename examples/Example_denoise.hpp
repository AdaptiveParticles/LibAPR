//
// Created by bevan on 29/11/2020.
//

#ifndef LIBAPR_EXAMPLE_DENOISE_HPP
#define LIBAPR_EXAMPLE_DENOISE_HPP

#include <functional>
#include <string>

#include <io/TiffUtils.hpp>
#include "data_structures/APR/APR.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

#include "data_structures/APR/particles/ParticleData.hpp"
#include "io/APRFile.hpp"
#include "numerics/APRDenoise.hpp"


#include "ExampleHelpers.hpp"

const char* usage = R"(
Examples of denoising an APR

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_denoise

Note:

)";

struct cmdLineOptionsDenoise{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    bool stats_file = false;
};

bool denoise_example(cmdLineOptionsDenoise& options);

cmdLineOptionsDenoise read_command_line_options(int argc, char **argv);

bool denoise_example(cmdLineOptionsDenoise& options){
    // Filename
    std::string file_name = options.directory + options.input;

    APRTimer timer;

    timer.verbose_flag = true;

    // APR data structure
    APR apr;
    ParticleData<uint16_t>parts;

    //read APR and particles from file
    timer.start_timer("read apr");
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);
    aprFile.read_particles(apr,parts); //default read, will take the first particles added to the file
    aprFile.close();
    timer.stop_timer();

    // reconstruct noisy input image by piecewise constant interpolation
    timer.start_timer("pc interp");
    PixelData<uint16_t> recon_pc;
    APRReconstruction::interp_img(apr,recon_pc, parts);
    timer.stop_timer();

    // save reconstructed input as TIFF
    std::string image_file_name = options.directory +  apr.name + "_org.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, recon_pc,false);

    /// Start of APR denoising
    timer.start_timer("train APR denoise");

    // init denoising stencils
    APRStencils aprStencils;
    auto it = apr.iterator();
    aprStencils.dim = it.number_dimensions(); //get the dimension from the file.

    // learn stencil weights
    APRDenoise aprDenoise;
    aprDenoise.verbose = false;
    aprDenoise.train_denoise(apr,parts,aprStencils);
    timer.stop_timer();


    // apply denoising stencils
    timer.start_timer("apply APR denoise");
    ParticleData<uint16> parts_denoised;
    aprDenoise.apply_denoise(apr,parts,parts_denoised,aprStencils);
    timer.stop_timer();

    // reconstruct image from denoised particles by piecewise constant interpolation
    timer.start_timer("pc interp");
    APRReconstruction::interp_img(apr,recon_pc, parts_denoised);
    timer.stop_timer();

    // save denosied image as TIFF
    image_file_name = options.directory +  apr.name + "_denoised.tif";
    TiffUtils::saveMeshAsTiff(image_file_name, recon_pc,false);

    // write APR and denoised particles to file
    timer.start_timer("write denoised APR to file");
    file_name = options.directory + options.output;
    aprFile.open(file_name,"WRITE");
    aprFile.write_apr(apr);
    aprFile.write_particles("particles",parts_denoised);
    aprFile.close();
    timer.stop_timer();

    return true;
}

cmdLineOptionsDenoise read_command_line_options(int argc, char **argv){

    cmdLineOptionsDenoise result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_iterate -i input_apr_file -d directory\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;

}



#endif //LIBAPR_EXAMPLE_DENOISE_HPP
