//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
const char* usage = R"(

 Example calculating the gradient magnitude of an APR and saving the result as a reconstructed TIFF image

 Usage:

 (using output of Example_get_apr)

 Example_compute_gradient -i input_apr_hdf5 -d directory -o output_tiff_file

 Note: input_file_path = directory + input_apr_hdf5
       output_file_path = directory + output_tiff_file + ".tif"

       if argument -o is not given, no file is written

 Options:

 -original_file (original image file given with respect to input_directory) (Produces the finite difference gradient magnitude on the original image)
 -dx value
 -dy value  voxel size in each dimension
 -dz value
 -sobel     if this flag is given, gradients are computed using Sobel filters. (otherwise central finite differences are used)

)";
#include <algorithm>
#include <iostream>

#include "Example_compute_gradient.hpp"
#include "io/TiffUtils.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "io/APRFile.hpp"
#include "numerics/APRNumerics.hpp"
#include "numerics/APRReconstruction.hpp"


int main(int argc, char **argv) {

    // INPUT PARSING
    cmdLineOptions options = read_command_line_options(argc, argv);

    std::string file_name = options.directory + options.input;

    APRTimer timer(true);

    timer.start_timer("Read APR and particles from file");

    APR apr;
    ParticleData<uint16_t> parts;

    //read file
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);
    aprFile.read_particles(apr,"particles",parts);

    timer.stop_timer();

    //Calculate the gradient of the APR
    timer.start_timer("compute APR gradient");

    ParticleData<float> output;
    std::vector<float> deltas = {options.dy, options.dx, options.dz};

    if(options.sobel) {
        APRNumerics::gradient_magnitude_sobel(apr, parts, output, deltas);
    } else {
        APRNumerics::gradient_magnitude_cfd(apr, parts, output, deltas);
    }
    timer.stop_timer();

    if(options.output.length() > 0) {
        // reconstruct pixel image from gradient
        timer.start_timer("reconstruct pixel image");
        PixelData<float> gradient_magnitude_image;
        APRReconstruction::interp_img(apr, gradient_magnitude_image, output);
        timer.stop_timer();

        timer.start_timer("write pixel image to file");
        std::string image_file_name = options.directory + options.output + ".tif";
        TiffUtils::saveMeshAsTiff(image_file_name, gradient_magnitude_image);
        timer.stop_timer();
    }
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
        std::cerr << argv[0] << std::endl;
        std::cerr << "Short Usage: \"Example_compute_gradient -i input_apr_file -d directory\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-sobel"))
    {
        result.sobel = true;
    }

    if(command_option_exists(argv, argv + argc, "-dx"))
    {
        result.dx =  std::stof(get_command_option(argv, argv + argc, "-dx"));
    }

    if(command_option_exists(argv, argv + argc, "-dy"))
    {
        result.dy =  std::stof(get_command_option(argv, argv + argc, "-dy"));
    }

    if(command_option_exists(argv, argv + argc, "-dz"))
    {
        result.dz =  std::stof(get_command_option(argv, argv + argc, "-dz"));
    }


    return result;

}
