#ifndef PARTPLAY_SEGMENT_RC_H
#define PARTPLAY_SEGMENT_RC_H

#include <functional>
#include <string>

#include "src/data_structures/structure_parts.h"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    bool stats_file = false;
};

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

#endif //PARTPLAY_PIPELINE_H