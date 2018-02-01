//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Example calculating the gradient, and gradient magnitude of the APR.
///
/// Produces *_paraview.h5 file and *_paraview.xmf and tiff images of the gradient
///
/// To use load the xmf file in Paraview, and select Xdmf Reader. Then click the small eye, to visualize the dataset. (Enable opacity mapping for surfaces, option can be useful)
///
/// Usage:
///
/// (using output of Example_compute_gradient)
///
/// Example_compute_gradient -i input_apr_hdf5 -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "Example_compute_gradient.hpp"
#include "src/io/TiffUtils.hpp"

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
        std::cerr << "Usage: \"Example_compute_gradient -i input_apr_file -d directory\"" << std::endl;
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
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    //Calculate the gradient of the APR

    ExtraParticleData<std::vector<float>> gradient; //vector for holding the derivative in the three directions, initialized to have the same number of elements as particles.

    APRNumerics apr_numerics;

    apr_numerics.compute_gradient_vector(apr,gradient,false);


    MeshNumerics mesh_numerics;

    MeshData<uint16_t> pc_recon;

    apr.interp_img(pc_recon,apr.particles_intensities);

    std::vector<MeshData<float>> gradient_mesh;

    mesh_numerics.compute_gradient(pc_recon,gradient_mesh);

    for (uint64_t i = 0; i < gradient_mesh[0].mesh.size(); ++i) {
        gradient_mesh[0].mesh[i] = 5.0f*sqrt(pow(gradient_mesh[0].mesh[i], 2.0f) + pow(gradient_mesh[1].mesh[i], 2.0f) + pow(gradient_mesh[2].mesh[i], 2.0f));
    }

    ExtraParticleData<float> gradient_magnitude(apr);
    //compute the magnitude of the gradient, scale it by 5 for visualization when writing as uint16 int
    gradient.map(apr,gradient_magnitude,[](const std::vector<float> &a) { return 5.0f*sqrt(pow(a[0], 2.0f) + pow(a[1], 2.0f) + pow(a[2], 2.0f)); });

    // write result to image
    MeshData<float> gradient_magnitude_image;
    apr.interp_img(gradient_magnitude_image,gradient_magnitude);
    std::string image_file_name = options.directory + name + "_gradient.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, gradient_magnitude_image);

    // also write to a paraview viewable file
    apr.write_apr_paraview(options.directory,name + "_gradient",gradient_magnitude);

    ExtraParticleData<float> gradient_y(apr);
    gradient.map(apr,gradient_y,[](const std::vector<float> &a) { return 100*abs(a[0]); });

    apr.interp_img(gradient_magnitude_image,gradient_y);
    image_file_name = options.directory + name + "_gradient_y.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, gradient_magnitude_image);

    image_file_name = options.directory + name + "_gradient_mesh.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, gradient_mesh[0]);
}