//
// Created by cheesema on 13.02.18
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
const char* usage = R"(
Example setting the APR iterator using random access

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_random_accesss -i input_apr_hdf5 -d input_directory

Note: There is no output, this file is best utilized by looking at the source code for example (test/Examples/Example_random_access.cpp) of how to code different
random access strategies on the APR.

)";


#include <algorithm>
#include <iostream>

#include "Example_apr_tree.hpp"



int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR <uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end() - 3, name.end());

    APRTree<uint16_t> apr_tree(apr);

    APRTreeIterator<uint16_t> apr_tree_iterator(apr_tree);

    ExtraParticleData<float> tree_data(apr_tree);

    uint64_t counter = 0;
    uint64_t counter_interior = 0;
    uint64_t particle_number;
    //Basic serial iteration over all particles
    for (particle_number = 0; particle_number < apr_tree.total_number_parent_cells(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        apr_tree_iterator.set_iterator_to_particle_by_number(particle_number);
        counter++;
        std::cout << apr_tree_iterator.x() << " " << apr_tree_iterator.y() << " " << (int)apr_tree_iterator.type() << " " << apr_tree_iterator.global_index() << std::endl;

        if(apr_tree_iterator.type() == 6){
            //count those nodes that do not have children that are in the APR
            counter_interior++;
        }

        tree_data[apr_tree_iterator] = apr_tree_iterator.type();
    }

    std::cout << counter << std::endl;
    std::cout << counter_interior << std::endl;
    std::cout << counter/(apr.total_number_particles()*1.0f) << std::endl;

}
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
        std::cerr << "Usage: \"Example_random_access -i input_apr_file -d directory\"" << std::endl;
        std::cerr << usage << std::endl;
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
