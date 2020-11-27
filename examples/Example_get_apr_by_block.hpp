//
// Created by joel on 03.11.20.
//

#ifndef LIBAPR_EXAMPLE_GET_APR_BY_BLOCK_HPP
#define LIBAPR_EXAMPLE_GET_APR_BY_BLOCK_HPP

#include <functional>
#include <string>

#include "algorithm/APRParameters.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/APR.hpp"
#include "algorithm/APRConverter.hpp"
#include "io/APRWriter.hpp"

struct cmdLineOptions{
    std::string output_dir = "";
    std::string output = "output";
    std::string directory = "";
    std::string input = "";
    std::string mask_file = "";
    bool neighborhood_optimization = true;
    bool output_steps = false;
    bool store_tree = false;

    float sigma_th = 5;
    float Ip_th = 0;
    float lambda = 3;
    float rel_error = 0.1;
    float grad_th = 1;

    int z_block_size = 128;
    int z_ghost = 16; // number of "ghost slices" to use in the APR pipeline
    int z_ghost_sampling = 64; // number of "ghost slices" to use when sampling intensities
};

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

cmdLineOptions read_command_line_options(int argc, char **argv);

#endif //LIBAPR_EXAMPLE_GET_APR_BY_BLOCK_HPP
