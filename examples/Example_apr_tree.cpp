//
// Created by cheesema on 05.07.18.
//



//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
const char* usage = R"(
Example using the APR Tree

Usage:

(using *.apr output of Example_get_apr)

Example_apr_tree -i input_apr_hdf5 -d input_directory

Note: There is no output, this file is best utilized by looking at the source code for an example of how to code different access strategies on the APR tree (interior nodes).

)";


#include <algorithm>
#include <iostream>

#include "Example_apr_tree.hpp"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include  "io/APRFile.hpp"
#include  "numerics/APRTreeNumerics.hpp"


int main(int argc, char **argv) {

    // INPUT PARSING
    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    APRTimer timer(true);

    // APR datastructure
    APR apr;

    //read file
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);

    ParticleData<uint16_t>parts;
    aprFile.read_particles(apr,"particles",parts);

    aprFile.close();

    // Fill the interior tree nodes by successively averaging the values of child nodes
    ParticleData<float> partsTree;
    APRTreeNumerics::fill_tree_mean(apr,parts,partsTree);

    // Tree iterator using the linear access structure
    auto tree_iterator_linear = apr.tree_iterator();

    ParticleData<uint16_t> partsTreelevel(apr.total_number_tree_particles());

    timer.start_timer("APR interior tree loop, linear iterator");

    for(int level = tree_iterator_linear.level_min(); level <= tree_iterator_linear.level_max(); ++level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(tree_iterator_linear)
#endif
        for (int z = 0; z < tree_iterator_linear.z_num(level); z++) {
            for (int x = 0; x < tree_iterator_linear.x_num(level); ++x) {
                for(tree_iterator_linear.begin(level, z, x); tree_iterator_linear < tree_iterator_linear.end();
                    ++tree_iterator_linear) {

                    // do something, e.g.
                    if(level < tree_iterator_linear.level_max()) {
                        partsTreelevel[tree_iterator_linear] = 2 * partsTree[tree_iterator_linear];
                    } else {
                        partsTreelevel[tree_iterator_linear] = partsTree[tree_iterator_linear];
                    }
                }
            }
        }
    }

    timer.stop_timer();


    auto tree_iterator_random = apr.random_tree_iterator();

    timer.start_timer("APR interior tree loop, random iterator");

    //iteration over the interior tree is identical to that over the standard APR, simply using the APRTreeIterator.

    for (int level = tree_iterator_random.level_min(); level <= tree_iterator_random.level_max(); ++level) {
#ifdef HAVE_OPENMP
        #pragma omp parallel for schedule(dynamic) firstprivate(tree_iterator_random)
#endif
        for (int z = 0; z < tree_iterator_random.z_num(level); z++) {
            for (int x = 0; x < tree_iterator_random.x_num(level); ++x) {
                for (tree_iterator_random.begin(level, z, x); tree_iterator_random < tree_iterator_random.end();
                     tree_iterator_random++) {

                    // do something, e.g.
                    if(level < tree_iterator_random.level_max()) {
                        partsTreelevel[tree_iterator_random] = 2 * partsTree[tree_iterator_random];
                    } else {
                        partsTreelevel[tree_iterator_random] = partsTree[tree_iterator_random];
                    }
                }
            }
        }
    }

    timer.stop_timer();

    //Also neighbour access can be done between neighboring particle cells on the same level

    auto neigh_tree_iterator = apr.random_tree_iterator();

    timer.start_timer("APR parallel iterator neighbour loop");

    for (int level = tree_iterator_random.level_min(); level <= tree_iterator_random.level_max(); ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(tree_iterator_random,neigh_tree_iterator)
#endif
        for (int z = 0; z < tree_iterator_random.z_num(level); z++) {
            for (int x = 0; x < tree_iterator_random.x_num(level); ++x) {
                for (tree_iterator_random.begin(level, z, x); tree_iterator_random < tree_iterator_random.end();
                     tree_iterator_random++) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        tree_iterator_random.find_neighbours_same_level(direction);
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        for (int index = 0; index < tree_iterator_random.number_neighbours_in_direction(direction); ++index) {

                            if (neigh_tree_iterator.set_neighbour_iterator(tree_iterator_random, direction, index)) {
                                //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)

                            }
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();




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
