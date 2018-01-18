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
#include <benchmarks/development/old_numerics/filter_numerics.hpp>

#include "benchmarks/development/Example_newstructures.h"





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

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    apr.parameters.input_dir = options.directory;

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    APRAccess apr_access;


    //just run old code and initialize it there
    //apr_access.test_method(apr);

    /////
    //
    //  Now new data-structures
    //
    /////

    APRAccess apr_access2;
    std::vector<std::vector<uint8_t>> p_map;

    timer.start_timer("generate pmap");
    apr_access2.generate_pmap(apr,p_map);
    timer.stop_timer();

    timer.start_timer("generate map structure");
    apr_access2.initialize_structure_from_particle_cell_tree(apr,p_map);
    timer.stop_timer();

    MapStorageData map_data;

    timer.start_timer("flatten");
    apr_access2.flatten_structure(apr,map_data);
    timer.stop_timer();

    std::cout << apr_access2.total_number_parts << std::endl;
    std::cout << apr_access2.total_number_gaps << std::endl;
    std::cout << apr_access2.total_number_non_empty_rows << std::endl;

}

