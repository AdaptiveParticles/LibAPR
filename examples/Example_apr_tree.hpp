//
// Created by cheesema on 13.02.18.
//

#ifndef LIBAPR_EXAMPLE_APR_TREE_HPP
#define LIBAPR_EXAMPLE_APR_TREE_HPP

#include <functional>
#include <string>

#include "src/data_structures/APR/APR.hpp"
#include "src/data_structures/APR/APRTree.hpp"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    bool stats_file = false;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);


#endif //LIBAPR_EXAMPLE_APR_TREE_HPP
