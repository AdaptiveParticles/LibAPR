//
// Created by cheesema on 14/03/17.
//

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
        std::cerr << "Usage: \"Example_neigh -i input_apr_file -d directory [-t] [-o outputfile]\"" << std::endl;
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
    APR<float> apr;

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
    ExtraPartCellData<float> calc_ex(apr);


    timer.start_timer("APR serial iterator loop");

    //Basic serial iteration over all particles
    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
        // multiple the Particle Cell type by the particle intensity (the intensity is stored as a ExtraPartCellData and therefore is no different from any additional datasets)
        apr(calc_ex) = apr.type()*apr(apr.particles_int);
    }

    timer.stop_timer();

    ////////////////////////////
    ///
    /// OpenMP Parallel loop iteration (requires seperate iterators from the apr structure used in the serial examples above
    /// however, the functionality and access is exactly the same). (For use with neighbour access see Example_apr_neigh)
    ///
    ///////////////////////////

    //initialization of the iteration structures
    APR_iterator<float> apr_it(apr); //this is required for parallel access
    uint64_t part;

    //create particle dataset
    ExtraPartCellData<float> calc_ex2(apr);

    timer.start_timer("APR parallel iterator loop");

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it)
    for (part = 0; part < apr.num_parts_total; ++part) {
        //needed step for any parallel loop (update to the next part)
        apr_it.set_part(part);

        if(apr_it.depth() < apr_it.depth_max()) {
            //get global y co-ordinate of the particle and put result in calc_ex2 at the current Particle Cell (PC) location
            apr_it(calc_ex2) = apr_it.y_global();

            apr_it.x(); // gets the Particle cell spatial index x.

            apr_it.z_nearest_pixel(); //gets the rounded up nearest pixel to the co-ordinate from original image (Larger then PC pixels don't align with pixels)

            apr_it.depth(); // gets the level of the Particle Cell (higher the level (depth), the smaller the Particle Cell --> higher resolution locally) PC at pixel resolution depth = depth_max();

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
    timer.start_timer("Using transform_parts: square the dataset");
    calc_ex.transform_parts([] (const float& a){return pow(a,2);});
    timer.stop_timer();

    //compare to explicit loop
    timer.start_timer("Using parallel iterator loop: square the dataset");

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it)
    for (part = 0; part < apr.num_parts_total; ++part) {
        //needed step for any parallel loop (update to the next part)
        apr_it.set_part(part);

        apr_it(calc_ex) = pow(apr_it(calc_ex),2);
    }

    timer.stop_timer();

    /// Single dataset, unary operation, return new dataset for result
    timer.start_timer("Take the absolute value and output");
    ExtraPartCellData<float> output_1;
    //return the absolute value of the part dataset (includes initialization of the output result)
    output_1 = calc_ex.transform_parts_output([] (const float& a){return abs(a);});
    timer.stop_timer();

    /// Two datasets, binary operation, return result to the particle dataset form which it is performed.
    timer.start_timer("Add two particle datasets");
    calc_ex2.transform_parts(calc_ex,std::plus<float>()); // adds calc_ex to calc_ex2 and returns the result to calc_ex
    timer.stop_timer();

    /// Two datasets, binary operation, return result to the particle dataset form which it is performed.

    ExtraPartCellData<float> output_2;
    //return the maximum of the two datasets
    timer.start_timer("Calculate and return the max of two particle datasets");
    output_2 = calc_ex.transform_parts_output(calc_ex2,[] (const float& a,const float& b) {return std::max(a,b);});
    timer.stop_timer();

}


