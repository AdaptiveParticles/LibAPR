//
// Created by cheesema on 21.01.18.
//

#ifndef LIBAPR_EXAMPLE_COMPUTE_GRADIENT_HPP
#define LIBAPR_EXAMPLE_COMPUTE_GRADIENT_HPP

#include <functional>
#include <string>
#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "io/TiffUtils.hpp"
#include "io/APRFile.hpp"
#include "numerics/APRNumerics.hpp"
#include "numerics/APRReconstruction.hpp"

struct cmdLineOptions{
    std::string output = "";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    bool sobel = false;
    float dx = 1.0f;
    float dy = 1.0f;
    float dz = 1.0f;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);


#endif //LIBAPR_EXAMPLE_COMPUTE_GRADIENT_HPP
