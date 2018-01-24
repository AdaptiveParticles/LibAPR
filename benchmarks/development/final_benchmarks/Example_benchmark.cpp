//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Examples of simple iteration an access to Particle Cell, and particle information. (See Example_neigh, for neighbor access)
///
/// Usage:
///
/// (using output of Example_get_apr)
///
/// Example_apr_iterate -i input_image_tiff -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <src/algorithm/APRConverter.hpp>

#include "Example_benchmark.h"

bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_iterate -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;

}


int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = false;

    // APR datastructure
    APR<uint16_t> apr;

    APRConverter<uint16_t> apr_converter;

    apr_converter.par.Ip_th = 1032;
    apr_converter.par.sigma_th = 242.822;
    apr_converter.par.sigma_th_max = 95.9877;
    apr_converter.par.rel_error = 0.1;
    apr_converter.par.lambda = 3;

    apr_converter.par.input_dir = options.directory;
    apr_converter.par.input_image_name  = options.input;

    TiffUtils::TiffInfo inputTiff(apr_converter.par.input_dir + apr_converter.par.input_image_name);
    MeshData<uint16_t> input_image = TiffUtils::getMesh<uint16_t>(inputTiff);

    apr_converter.get_apr_method(apr, input_image);

    APRBenchmark apr_benchmarks;

    float num_repeats = 1;

    apr_benchmarks.pixels_linear_neighbour_access<uint16_t,float>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),num_repeats);
    apr_benchmarks.apr_linear_neighbour_access<uint16_t,float>(apr,num_repeats);

    float num_repeats_random = 10000000;

    apr_benchmarks.pixel_neighbour_random<uint16_t,float>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2), num_repeats_random);
    apr_benchmarks.apr_random_access<uint16_t,float>(apr,num_repeats_random);

}
