#ifndef PARTPLAY_COMPRESS
#define PARTPLAY_COMPRESS

#include <functional>
#include <string>

#include "data_structures/APR/APR.hpp"


struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    unsigned int compress_type=1;
    float quantization_level=1;
    bool stats_file = false;
    unsigned int compress_level = 2;
    float output_tiff = false;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

#endif //PARTPLAY_PIPELINE_H