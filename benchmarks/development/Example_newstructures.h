#ifndef PARTPLAY_COMPRESS
#define PARTPLAY_COMPRESS

#include <functional>
#include <string>

#include "src/data_structures/APR/APR.hpp"
#include "src/data_structures/APR/APRAccess.hpp"
#include "src/data_structures/APR/APRIteratorNew.hpp"

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

#endif //PARTPLAY_PIPELINE_H