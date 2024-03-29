//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(
Examples of iteration and access to particle neighbours on the face of the Particle Cells.

Usage:

(using *.apr output of Example_get_apr)

Example_apr_neighbour_access -i input_image_tiff -d input_directory

Note: There is no output, this file is best utilized by looking at the source code for example (test/Examples/Example_apr_neighbour_access.cpp) of how to code different
neighbour iteration strategies on the APR.

)";


#include <algorithm>
#include <iostream>

#include "Example_apr_neighbour_access.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "io/APRFile.hpp"


int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR apr;

    //read file

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);

    ParticleData<uint16_t>parts;
    aprFile.read_particles(apr, parts);

    aprFile.close();

    ///////////////////////////
    ///
    /// Serial Neighbour Iteration (Only Von Neumann (Face) neighbours)
    ///
    /////////////////////////////////

    ParticleData<uint16_t> neigh_avg(apr.total_number_particles());

    auto neighbour_iterator = apr.random_iterator();
    auto apr_iterator = apr.random_iterator();


    timer.start_timer("APR serial iterator neighbours loop");

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator++) {

                    //now we only update the neighbours, and directly access them through a neighbour iterator

                    float counter = 0;
                    float temp = 0;

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        apr_iterator.find_neighbours_in_direction(direction);

                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                            // on each face, there can be 0-4 neighbours accessed by index
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //will return true if there is a neighbour defined

                                temp += parts[neighbour_iterator];
                                counter++;

                            }
                        }
                    }

                    neigh_avg[apr_iterator] = temp / counter;

                }
            }
        }
    }

    timer.stop_timer();


    ////////////////////////////
    ///
    /// OpenMP Parallel loop iteration
    ///
    ///////////////////////////

    //initialization of the iteration structures

    ParticleData<float> neigh_xm(apr.total_number_particles());

    timer.start_timer("APR parallel iterator neighbour loop");

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator,neighbour_iterator)
#endif
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        apr_iterator.find_neighbours_in_direction(direction);
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
                                neigh_xm[apr_iterator] += parts[neighbour_iterator] *
                                                          (apr_iterator.y() - neighbour_iterator.y());
                            }
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();


    /*
     *  Access only one directions neighbour
     */

    ParticleData<float> type_sum(apr.total_number_particles());

    timer.start_timer("APR parallel iterator neighbour loop x direction");

    //need to initialize the neighbour iterator with the APR you are iterating over.

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator,neighbour_iterator)
#endif
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    const int direction = 3;
                    //now we only update the neighbours, and directly access them through a neighbour iterator
                    apr_iterator.find_neighbours_in_direction(direction); // 3 = -x face

                    for (int index = 0;
                         index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                        // from 0 to 4 neighbours
                        if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                            //access data and perform a conditional sum (neighbour_iterator has all access like the normal iterator)
                            if (neighbour_iterator.level() <= neighbour_iterator.level_max()) {
                                type_sum[apr_iterator] +=
                                        parts[neighbour_iterator];
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
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
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