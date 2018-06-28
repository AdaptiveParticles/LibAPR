//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(
Examples of simple iteration an access to Particle Cell, and particle information. (See Example_neigh, for neighbor access)

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_apr_iterate -i input_image_tiff -d input_directory

Note: There is no output, this file is best utilized by looking at the source code for example (test/Examples/Example_apr_iterate.cpp) of how to code different
iteration strategies on the APR.

)";

#include <algorithm>
#include <iostream>
#include <cmath>
#include "Example_apr_iterate.h"


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

    ///////////////////////////
    ///
    /// Serial Iteration (For use with neighbour access see Example_apr_neigh)
    ///
    /// Looping over with full access to particle information and access to particle datasets.
    ///
    /////////////////////////////////

    //Create particle datasets, once intiailized this has the same layout as the Particle Cells
    ExtraParticleData<float> calc_ex(apr);

    APRIterator<uint16_t> apr_iterator(apr); // not STL type iteration

    timer.start_timer("APR serial iterator loop");

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    //you can then also use it to access any particle properties stored as ExtraParticleData
                    calc_ex[apr_iterator] = 10.0f * apr.particles_intensities[apr_iterator];
                }
            }
        }

    }

    timer.stop_timer();

    //
    //  You can also iterate over by level, this is in the datastrucrure called depth, Particle Cells range from depth_min() to depth_max(), coinciding with level = l_min and level = l_max
    //

    timer.start_timer("APR serial iterator loop by level");

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    if (apr.particles_intensities[apr_iterator] > 100) {
                        //set all particles in calc_ex with an particle intensity greater then 100 to 0.
                        calc_ex[apr_iterator] = 0;
                    }
                }
            }
        }
    }

    timer.stop_timer();

    ////////////////////////////
    ///
    /// OpenMP Parallel loop iteration (For use with neighbour access see Example_apr_neigh)
    ///
    ///////////////////////////

    //create particle dataset
    ExtraParticleData<float> calc_example_2(apr);

    timer.start_timer("APR parallel iterator loop");

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {
                    if (apr_iterator.level() < apr_iterator.level_max()) {
                        //get global y co-ordinate of the particle and put result in calc_example_2 at the current Particle Cell (PC) location
                        calc_example_2[apr_iterator] = apr_iterator.y_global();
                    }
                }
            }
        }
    }

    timer.stop_timer();

    ////////////////////////////////////////
    ///
    /// One shot operations
    ///
    /// Efficient helpers for performing single operations over one or two particle datasets. Uses OpenMP and std::transform
    ///
    /// See std::transform and the std functional header and below for examples
    ///
    /// These are faster, but do not allow access to particle meta info (depth,type,x,y,z...ect.) (Nor neighbour operations see Example_apr_neigh)
    ///
    ////////////////////////////////////////

    /// Single dataset, unary operation, overwrite result
    //compute the square the of the dataset
    timer.start_timer("Using map: square the dataset");
    calc_ex.map_inplace(apr,[](const float &a) { return pow(a, 2); });
    timer.stop_timer();

    //compare to explicit loop
    timer.start_timer("Using parallel iterator loop: square the dataset");

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                }
                calc_ex[apr_iterator] = pow(calc_ex[apr_iterator], 2.0f);
            }
        }
    }

    timer.stop_timer();


    /*
    *
    * Alternative iteration strategy
    *
    */


    /// Single dataset, unary operation, return new dataset for result
    timer.start_timer("Take the absolute value and output");
    ExtraParticleData<float> output_1;
    //return the absolute value of the part dataset (includes initialization of the output result)
    calc_ex.map(apr,output_1,[](const float &a) { return std::abs(a); });
    timer.stop_timer();

    /// Two datasets, binary operation, return result to the particle dataset form which it is performed.
    timer.start_timer("Add two particle datasets");
    calc_example_2.zip_inplace(apr,calc_ex, std::plus<float>()); // adds calc_ex to calc_example_2 and returns the result to calc_ex
    timer.stop_timer();

    /// Two datasets, binary operation, return result to the particle dataset form which it is performed.

    ExtraParticleData<float> output_2;
    //return the maximum of the two datasets
    timer.start_timer("Calculate and return the max of two particle datasets");
    calc_ex.zip(apr,calc_example_2,output_2, [](const float &a, const float &b) { return std::max(a, b); });
    timer.stop_timer();

    /// All of the operations can be done for Particle Cells of a fixed level
    timer.start_timer("Using map: square the dataset only for particle cells at the highest level (level_max)");
    calc_ex.map_inplace(apr,[](const float &a) { return pow(a, 2); },apr.level_max());
    timer.stop_timer();

    /////////////////////////////////////////////
    ///
    /// Reading and writing additional particle information
    ///
    ///
    /// Requires the APR that was used to write it to be read back in.
    ///
    /////////////////////////////////////////////

    //write one of the above results to file
    apr.write_particles_only(options.directory,"example_output",output_2);

    std::string extra_file_name = options.directory + "example_output" + "_apr_extra_parts.h5";

    ExtraParticleData<float> output_2_read;

    //you need the same apr used to write it to load it (doesn't save location data)
    apr.read_parts_only(extra_file_name,output_2_read);



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
        std::cerr << "Usage: \"Example_apr_iterate -i input_apr_file -d directory\"" << std::endl;
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
