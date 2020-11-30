//
// Created by cheesema on 14/03/17.
//

#ifndef LIBAPR_EXAMPLE_NEIGHBOUR_ACCESS_HPP
#define LIBAPR_EXAMPLE_NEIGHBOUR_ACCESS_HPP

#include <functional>
#include <string>

#include "data_structures/APR/APR.hpp"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);


#endif //LIBAPR_EXAMPLE_NEIGHBOUR_ACCESS_HPP
