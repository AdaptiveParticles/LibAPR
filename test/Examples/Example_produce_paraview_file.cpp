//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Example of producing a hdf5 file of the APR that can be read and visualized by Paraview using Xdmf.
///
/// Produces *_paraview.h5 file and *_paraview.xmf
///
/// To use load the xmf file in Paraview, and select Xdmf Reader. Then click the small eye, to visualize the dataset. (Enable opacity mapping for surfaces, option can be useful)
///
/// Usage:
///
/// (using output of Example_produce_paraview_file) - for input first compute an APR using Example_get_apr
///
/// Example_produce_paraview_file -i input_apr_hdf5 -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "Example_produce_paraview_file.hpp"

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
        std::cerr << "Usage: \"Example_produce_paraview_file -i input_apr_file -d directory\"" << std::endl;
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

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    apr.write_apr_paraview(options.directory,name,apr.particles_int);
    std::cout << "Written the combination of h5 and xmf file that can be read by Paraview, load the xmf file in Paraview and select Xdmf Reader" << std::endl;

}
