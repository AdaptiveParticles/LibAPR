#ifndef PARTPLAY_RAYCAST_H
#define PARTPLAY_RAYCAST_H


#include "../../src/numerics/ray_cast.hpp"
#include "../../src/data_structures/APR/APR.hpp"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    float jitter = 0;
    float aniso = 1.0;
    unsigned int num_views= 60;
};

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

#endif //PARTPLAY_PIPELINE_H