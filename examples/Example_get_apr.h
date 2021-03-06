#ifndef LIBAPR_EXAMPLE_GET_APR_H
#define LIBAPR_EXAMPLE_GET_APR_H

#include <functional>
#include <string>

#include "algorithm/APRParameters.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/APR.hpp"
#include "algorithm/APRConverter.hpp"
#include "io/APRWriter.hpp"

struct cmdLineOptions{
    std::string gt_input = "";
    std::string output_dir = "";
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    std::string mask_file = "";
    bool stats_file = false;
    bool normalize_input = false;
    bool neighborhood_optimization = true;
    bool output_steps = false;
    unsigned int compress_level = 2;
    unsigned int compress_type = 0;
    bool store_tree = false;
    float quantization_factor = 0.5;

    bool auto_parameters = false;

    float Ip_th = 0;
    float lambda = -1;
    float sigma_th = 0;
    float rel_error = 0.1;
    float grad_th = 1;
};

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

cmdLineOptions read_command_line_options(int argc, char **argv);


#endif //LIBAPR_EXAMPLE_GET_APR_H