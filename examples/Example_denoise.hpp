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

    // APR datastructure
    APR apr;

    timer.start_timer("read apr");
    //read file
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);

    ParticleData<uint16_t>parts;
    aprFile.read_particles(apr,parts); //default read, will take the first particles added to the file

    aprFile.close();
    timer.stop_timer();

    APRStencils aprStencils;

    auto it = apr.iterator();

    aprStencils.dim = it.number_dimensions(); //get the dimension from the file.

    PixelData<float> recon_pc;

    timer.start_timer("pc interp");
    //perform piece-wise constant interpolation
    APRReconstruction::interp_img(apr,recon_pc, parts);
    timer.stop_timer();

    std::string image_file_name = options.directory +  apr.name + "_org.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, recon_pc,false);

    //load in an APR
    APRDenoise aprDenoise;

    aprDenoise.iteration_others = 1; //default = 1 (Changed)

    aprDenoise.N_ = 100;
    aprDenoise.N_max = 100;

    aprDenoise.train_denoise(apr,parts,aprStencils);

    APRDenoise aprDenoise_test;


    aprDenoise_test.apply_denoise(apr,parts,aprStencils);

    timer.start_timer("pc interp");
    //perform piece-wise constant interpolation
    APRReconstruction::interp_img(apr,recon_pc, parts);
    timer.stop_timer();

    image_file_name = options.directory +  apr.name + "_denoised.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, recon_pc,false);

    file_name = options.directory + options.output;

    aprFile.open(file_name,"WRITE");
    aprFile.write_apr(apr);
    aprFile.write_particles("particles",parts);
    aprFile.close();

    //IO is to do, just need to extend the existing code to handle the APR case, and load and read appropriately.


    std::string stencil_file_name = "test";
    Stencil<double> stencil_read;

    aprStencils.write_stencil(file_name,aprStencils.stencils.back());

    aprStencils.read_stencil(file_name,stencil_read);

    //compare stencils
    int stop = 1;

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
