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

#include "Example_apr_iterate.h"

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
    Part_timer timer;

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

    APRIterator<uint16_t> apr_iterator(apr);

    timer.start_timer("APR serial iterator loop");

    uint64_t particle_number;
    //Basic serial iteration over all particles
    for (particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        //once set you can access all informaiton on the particle cell
        uint64_t current_particle_cell_spatial_index_x = apr_iterator.x();
        uint64_t current_particle_cell_spatial_index_y = apr_iterator.y();
        uint64_t current_particle_cell_spatial_index_z = apr_iterator.z();
        uint64_t current_particle_cell_level = apr_iterator.level();
        uint64_t current_particle_cell_type = apr_iterator.type();

        //you can then also use it to access any particle properties stored as ExtraParticleData
        calc_ex[apr_iterator]= 10.0*apr.particles_intensities.get_particle(apr_iterator);
        calc_ex.set_particle(apr_iterator,12.0f);

    }

    timer.stop_timer();

    //
    //  You can also iterate over by level, this is in the datastrucrure called depth, Particle Cells range from depth_min() to depth_max(), coinciding with level = l_min and level = l_max
    //

    timer.start_timer("APR serial iterator loop by level");

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        for (particle_number = apr_iterator.particles_level_begin(level); particle_number < apr_iterator.particles_level_end(level); ++particle_number) {

            apr_iterator.set_iterator_to_particle_by_number(particle_number); // (Required step), sets the iterator to the particle

            //you can also retrieve global co-ordinates of the particles
            float x_global = apr_iterator.x_global();
            float y_global = apr_iterator.y_global();
            float z_global = apr_iterator.z_global();

            //or the closest pixel in the original image
            unsigned int x_nearest_pixel = apr_iterator.x_nearest_pixel();
            unsigned int y_nearest_pixel = apr_iterator.y_nearest_pixel();
            unsigned int z_nearest_pixel = apr_iterator.z_nearest_pixel();

            if(apr_iterator(apr.particles_intensities) > 100){
                //set all particles in calc_ex with an particle intensity greater then 100 to 0.
                apr_iterator(calc_ex) = 0;
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
    ExtraParticleData<float> calc_ex2(apr);

    timer.start_timer("APR parallel iterator loop");

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
    for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
        //needed step for any parallel loop (update to the next part)
        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        if(apr_iterator.level() < apr_iterator.level_max()) {
            //get global y co-ordinate of the particle and put result in calc_ex2 at the current Particle Cell (PC) location
            apr_iterator(calc_ex2) = apr_iterator.y_global();

            int x = apr_iterator.x(); // gets the Particle cell spatial index x.

            int z_pixel = apr_iterator.z_nearest_pixel(); //gets the rounded up nearest pixel to the co-ordinate from original image (Larger then PC pixels don't align with pixels)
             // gets the level of the Particle Cell (higher the level (depth), the smaller the Particle Cell --> higher resolution locally) PC at pixel resolution depth = depth_max();
            unsigned int level = apr_iterator.level(); //same as above
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

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
    for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
        //needed step for any parallel loop (update to the next part)
        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        apr_iterator(calc_ex) = pow(apr_iterator(calc_ex),2);
    }

    timer.stop_timer();

    /// Single dataset, unary operation, return new dataset for result
    timer.start_timer("Take the absolute value and output");
    ExtraParticleData<float> output_1;
    //return the absolute value of the part dataset (includes initialization of the output result)
    calc_ex.map(apr,output_1,[](const float &a) { return abs(a); });
    timer.stop_timer();

    /// Two datasets, binary operation, return result to the particle dataset form which it is performed.
    timer.start_timer("Add two particle datasets");
    calc_ex2.zip_inplace(apr,calc_ex, std::plus<float>()); // adds calc_ex to calc_ex2 and returns the result to calc_ex
    timer.stop_timer();

    /// Two datasets, binary operation, return result to the particle dataset form which it is performed.

    ExtraParticleData<float> output_2;
    //return the maximum of the two datasets
    timer.start_timer("Calculate and return the max of two particle datasets");
    calc_ex.zip(apr,calc_ex2,output_2, [](const float &a, const float &b) { return std::max(a, b); });
    timer.stop_timer();

    /// All of the operations can be done for Particle Cells of a fixed level

    timer.start_timer("Using map: square the dataset only for particle cells at the highest level (level_max)");
    calc_ex.map_inplace(apr,[](const float &a) { return pow(a, 2); },apr.level_max());
    timer.stop_timer();

    /////////////////////////////////////////////
    ///
    /// Reading and writing additional particle information
    ///
    /// (Current supports uint8, uint16, and float)
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

    //////////////////////////////////////////////////
    ///
    /// Advanced iteration strategies
    ///
    /// Below are some examples of further loop splitting that can be useful in image processing applications (See CompressAPR)
    ///
    ////////////////////////////////////////////////////

    ///
    ///  By level and z slice (loop over x and then y)
    ///

    timer.start_timer("by level and z");

    uint64_t counter = 0;

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        for(unsigned int z = 0; z < apr_iterator.spatial_index_z_max(level); ++z) {

            uint64_t  begin = apr_iterator.particles_z_begin(level,z);
            uint64_t  end = apr_iterator.particles_z_end(level,z);

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
            for (particle_number = apr_iterator.particles_z_begin(level,z);
                 particle_number < apr_iterator.particles_z_end(level,z); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                if (apr_iterator.z() == z) {

                } else {
                    std::cout << "broken" << std::endl;
                }
            }
        }
    }



    timer.stop_timer();

    ///
    /// Alternative Paralellization Strategy (OpenMP splits up the z-slices)
    ///

    timer.start_timer("By level parallelize by slice");

    counter = 0;

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

        int z = 0;
#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number,z) firstprivate(apr_iterator)
#endif
        for(z = 0; z < apr.spatial_index_z_max(level); ++z) {

            uint64_t  begin = apr_iterator.particles_z_begin(level,z);
            uint64_t  end = apr_iterator.particles_z_end(level,z);

            for (particle_number = apr_iterator.particles_z_begin(level,z);
                 particle_number < apr_iterator.particles_z_end(level,z); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                counter++;

                if (apr_iterator.z() == z) {

                } else {
                    std::cout << "broken" << std::endl;
                }
            }
        }
    }
    timer.stop_timer();

}


