#ifndef PARTPLAY_FILTER_H
#define PARTPLAY_FILTER_H

#include <functional>
#include <string>

#include "../../src/data_structures/structure_parts.h"

struct cmdLineOptions_filter{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    bool stats_file = false;
    std::string original_file = "";
    std::string gt = "";
};

cmdLineOptions_filter read_command_line_options_filter(int argc, char **argv, Part_rep& part_rep);

bool command_option_exists_filter(char **begin, char **end, const std::string &option);

char* get_command_option_filter(char **begin, char **end, const std::string &option);

#endif //PARTPLAY_FILTER_H