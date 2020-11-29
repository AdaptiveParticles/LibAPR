//
// Created by bevan on 29/11/2020.
//

#ifndef LIBAPR_EXAMPLE_DENOISE_HPP
#define LIBAPR_EXAMPLE_DENOISE_HPP

#include <functional>
#include <string>

#include "data_structures/APR/APR.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include"data_structures/APR/particles/ParticleData.hpp"
#include"io/APRFile.hpp"

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

    // Read the apr file into the part cell structure
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
    aprFile.read_particles(apr,parts);

    aprFile.close();
    timer.stop_timer();

    ///////////////////////////
    ///
    /// Serial Iteration (For use with neighbour access see Example_apr_neigh)
    ///
    /// Looping over with full access to particle information and access to particle datasets.
    ///
    /////////////////////////////////

    //Create particle datasets, once intiailized this has the same layout as the Particle Cells
    ParticleData<float> calc_ex(apr.total_number_particles());

    auto it = apr.iterator(); // not STL type iteration

    timer.start_timer("APR serial iterator loop");

    for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level, z, x); it < it.end(); it++) {

                    //you can then also use it to access any particle properties stored as ExtraParticleData
                    calc_ex[it] = 10.0f * parts[it];
                }
            }
        }

    }

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
