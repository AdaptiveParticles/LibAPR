//
// Created by cheesema on 14/03/17.
//

#include <algorithm>
#include <iostream>

#include "Example_neigh.hpp"
#include "../../test/utils.h"

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

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile -d directory [-t] [-o outputfile]\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-t"))
    {
        part_rep.timer.verbose_flag = true;
    }

    return result;

}

int main(int argc, char **argv) {

    Part_rep part_rep;

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);

    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);

    read_write_structure_test(pc_struct);

    utest_moore_neighbours(pc_struct);

}


