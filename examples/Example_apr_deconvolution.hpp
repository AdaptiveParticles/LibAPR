//
// Created by joel on 30.11.20.
//

#ifndef LIBAPR_EXAMPLE_APR_DECONVOLUTION_HPP
#define LIBAPR_EXAMPLE_APR_DECONVOLUTION_HPP

#include <functional>
#include <string>
#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "io/TiffUtils.hpp"
#include "io/APRFile.hpp"
#include "numerics/APRNumerics.hpp"
#include "numerics/APRReconstruction.hpp"

#ifdef APR_USE_CUDA
#include "numerics/APRNumericsGPU.hpp"
#endif

struct cmdLineOptions{
    std::string output = "";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    int number_iterations = 10;
    bool use_cuda = false;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

#endif //LIBAPR_EXAMPLE_APR_DECONVOLUTION_HPP
