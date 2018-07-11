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

    APRTreeIterator<uint16_t> apr_tree_iterator(apr);

    apr.apr_tree.init(apr);

    apr.apr_tree.fill_tree_mean_downsample(apr.particles_intensities);

    timer.start_timer("APR interior tree loop");

    ExtraParticleData<uint16_t> partsTreelevel(apr.apr_tree);

    //iteration over the interior tree is identical to that over the standard APR, simply using the APRTreeIterator.

    for (unsigned int level = apr_tree_iterator.level_min(); level <= apr_tree_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
        #pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_tree_iterator)
#endif
        for (z = 0; z < apr_tree_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_tree_iterator.spatial_index_x_max(level); ++x) {
                for (apr_tree_iterator.set_new_lzx(level, z, x); apr_tree_iterator.global_index() < apr_tree_iterator.end_index;
                     apr_tree_iterator.set_iterator_to_particle_next_particle()) {

                    if(apr_tree_iterator.level() < apr_tree_iterator.level_max()) {
                        partsTreelevel[apr_tree_iterator] = (uint16_t)2*apr.apr_tree.particles_ds_tree[apr_tree_iterator];
                    } else {
                        partsTreelevel[apr_tree_iterator] = apr.apr_tree.particles_ds_tree[apr_tree_iterator];
                    }


                }
            }
        }
    }

    timer.stop_timer();

    //Also neighbour access can be done between neighboring particle cells on the same level

    APRTreeIterator<uint16_t> neigh_tree_iterator(apr);

    timer.start_timer("APR parallel iterator neighbour loop");

    for (unsigned int level = apr_tree_iterator.level_min(); level <= apr_tree_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator,neighbour_iterator)
#endif
        for (z = 0; z < apr_tree_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_tree_iterator.spatial_index_x_max(level); ++x) {
                for (apr_tree_iterator.set_new_lzx(level, z, x); apr_tree_iterator.global_index() < apr_tree_iterator.end_index;
                     apr_tree_iterator.set_iterator_to_particle_next_particle()) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        apr_tree_iterator.find_neighbours_same_level(direction);
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        for (int index = 0; index < apr_tree_iterator.number_neighbours_in_direction(direction); ++index) {

                            if (neigh_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, index)) {
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
