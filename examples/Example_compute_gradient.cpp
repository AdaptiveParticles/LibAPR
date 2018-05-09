//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
const char* usage = R"(

 Example calculating the gradient, and gradient magnitude of the APR.

 Produces *_paraview.h5 file and *_paraview.xmf and tiff images of the gradient

 To use load the xmf file in Paraview, and select Xdmf Reader. Then click the small eye, to visualize the dataset. (Enable opacity mapping for surfaces, option can be useful)

 Usage:

 (using output of Example_compute_gradient)

 Example_compute_gradient -i input_apr_hdf5 -d input_directory

 Options:

 -original_file (original image file given with respect to input_directory) (Produces the finite difference gradient magnitude on the original image)
 -smooth *number (integer) (perform a seperable smoothing in each direction *number of times before taking the gradient) (Feel free to change the weights)
 -aniso_z anisotropy scaling for gradient calculation (default = 1, scaling factor 3 = 3*dx)

)";
#include <algorithm>
#include <iostream>

#include "Example_compute_gradient.hpp"
#include "io/TiffUtils.hpp"


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

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    if(options.smooth_number>0) {
        //smooth the image with a simply sepeable filter *smooth_number times

        APRNumerics aprNumerics;
        ExtraParticleData<uint16_t> smooth(apr);
        std::vector<float> filter = {0.1f, 0.8f, 0.1f}; // << Feel free to play with these
        aprNumerics.seperable_smooth_filter(apr, apr.particles_intensities, smooth, filter, options.smooth_number);

        std::swap(apr.particles_intensities.data, smooth.data);
    }

    //Calculate the gradient of the APR
    ExtraParticleData<std::vector<float>> gradient; //vector for holding the derivative in the three directions, initialized to have the same number of elements as particles.

    std::vector<float> delta = {1,1,options.anisotropy_z};

    APRNumerics::compute_gradient_vector(apr,gradient,false,delta);

    ExtraParticleData<float> gradient_magnitude(apr);
    //compute the magnitude of the gradient, scale it by 5 for visualization when writing as uint16 int
    gradient.map(apr,gradient_magnitude,[](const std::vector<float> &a) { return 20.0f*sqrt(pow(a[0], 2.0f) + pow(a[1], 2.0f) + pow(a[2], 2.0f)); });

    // write result to image
    PixelData<float> gradient_magnitude_image;
    apr.interp_img(gradient_magnitude_image,gradient_magnitude);
    //apr.interp_img(gradient_magnitude_image,apr.particles_intensities);

    std::string image_file_name = options.directory + name + "_gradient_magnitude.tif";
    TiffUtils::saveMeshAsTiff(image_file_name, gradient_magnitude_image);

    //////////////////////
    //
    //  Perform same operation on original image
    //
    //////////////////////


    if(options.original_image.size() > 0) {

        TiffUtils::TiffInfo inputTiff(options.directory + options.original_image);
        PixelData<uint16_t> original_image = TiffUtils::getMesh<uint16_t>(inputTiff);

        std::vector<PixelData<float>> gradient_mesh;

        MeshNumerics::compute_gradient(original_image, gradient_mesh);

        for (uint64_t i = 0; i < gradient_mesh[0].mesh.size(); ++i) {
            gradient_mesh[0].mesh[i] = 5.0f *
                                       sqrt(pow(gradient_mesh[0].mesh[i], 2.0f) + pow(gradient_mesh[1].mesh[i], 2.0f) +
                                            pow(gradient_mesh[2].mesh[i], 2.0f));
        }

        image_file_name = options.directory + name + "_gradient_mesh.tif";
        TiffUtils::saveMeshAsTiffUint16(image_file_name, gradient_mesh[0]);
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

    if(command_option_exists(argv, argv + argc, "-original_image"))
    {
        result.original_image = std::string(get_command_option(argv, argv + argc, "-original_image"));
    }

    if(command_option_exists(argv, argv + argc, "-smooth"))
    {
        result.smooth_number = (unsigned int) std::stoi(get_command_option(argv, argv + argc, "-smooth"));
    }

    if(command_option_exists(argv, argv + argc, "-aniso_z"))
    {
        result.anisotropy_z =  std::stof(get_command_option(argv, argv + argc, "-aniso_z"));
    }


    return result;

}
