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

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-4,name.end());

    APRConverter<uint16_t> apr_converter;

    apr_converter.par.Ip_th = 1032;
    apr_converter.par.sigma_th = 242.822;
    apr_converter.par.sigma_th_max = 95.9877;
    apr_converter.par.rel_error = 0.1;
    apr_converter.par.lambda = 3;

    apr_converter.par.input_dir = options.directory;
    apr_converter.par.input_image_name  = options.input;

    apr_converter.fine_grained_timer.verbose_flag = false;
    apr_converter.method_timer.verbose_flag = false;
    apr_converter.allocation_timer.verbose_flag = false;
    apr_converter.computation_timer.verbose_flag = false;


    APRBenchmark apr_benchmarks;

    apr_benchmarks.analysis_data.name = "test_benchmarking";

    apr_benchmarks.analysis_data.file_name = "test";

    apr_benchmarks.benchmark_dataset(apr_converter);

    apr_benchmarks.analysis_data.write_analysis_data_hdf5();

}
