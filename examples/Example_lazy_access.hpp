//
// Created by joel on 06.01.22.
//

#ifndef APR_EXAMPLE_LAZY_ACCESS_HPP
#define APR_EXAMPLE_LAZY_ACCESS_HPP

#include <string>
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/iterators/LazyIterator.hpp"

struct cmdLineOptions{
    std::string directory = "";
    std::string input = "";
    bool stats_file = false;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);


#endif //APR_EXAMPLE_LAZY_ACCESS_HPP
