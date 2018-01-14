#ifndef PARTPLAY_GETAPR_H
#define PARTPLAY_GETAPR_H

#include <functional>
#include <string>

#include "src/algorithm/APRParameters.hpp"

#include "src/data_structures/Mesh/meshclass.h"

//#include "benchmarks/development/old_algorithm/apr_pipeline.hpp"

#include "src/algorithm/APR_converter.hpp"

#include "src/data_structures/APR/APR.hpp"



struct cmdLineOptions{
    std::string gt_input = "";
    std::string output_dir = "";
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    std::string mask_file = "";
    bool stats_file = false;

    float Ip_th = -1;
    float SNR_min = -1;
    float lambda = -1;
    float min_signal = -1;
    float rel_error = 0.1;

    std::string img_type = "uint16";

};

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

cmdLineOptions read_command_line_options(int argc, char **argv);


#endif //PARTPLAY_PIPELINE_H