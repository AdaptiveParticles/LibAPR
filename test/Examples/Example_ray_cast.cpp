////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///

const char* usage = R"(
Ray cast example:

Creates a maximum projection ray cast from an APR and outputs to a tiff image.

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_ray_cast -i inputfile [-d directory]

Optional:

-aniso z_stretch (stretches the ray cast in the z axis)
-jitter jitter_factor (0-1) (perturbs the particles randomly in the ray case, in an effort to remove artifacts from alignement of view in the ray-cast)
-numviews The number of views that are calculated and output to the tiff file.
-original_image give the file name of the original image, then also does a pixel image based raycast on the original image.
-view_radius distance of viewer (default 0.98)

e.g. Example_ray_cast -i nuc_apr.h5 -d /Test/Input_examples/ -aniso 2.0 -jitter 0.1 -numviews 60

)";
#include <algorithm>
#include <iostream>

#include "Example_ray_cast.h"
#include "src/io/TiffUtils.hpp"


int main(int argc, char **argv) {

    // INPUT PARSING
    
    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    APRTimer timer;

    /////////////////
    ///
    ///  Raycast Parameters
    ///
    ////////////////

    APRRaycaster apr_raycaster;

    apr_raycaster.theta_0 = -3.14f; //start
    apr_raycaster.theta_final = 3.14f; //stop radians
    apr_raycaster.radius_factor = options.view_radius; //radius scaling
    apr_raycaster.theta_delta = (apr_raycaster.theta_final - apr_raycaster.theta_0)/(options.num_views*1.0f); //steps
    apr_raycaster.scale_z = options.aniso; //z scaling

    apr_raycaster.jitter = (options.jitter > 0);
    apr_raycaster.jitter_factor = options.jitter;

    apr_raycaster.name = apr.name;

    MeshData<uint16_t> views;

    /////////////
    ///
    ///  Compute APR (maximum projection) raycast
    ///
    /////////////

    apr_raycaster.perform_raycast(apr,apr.particles_intensities,views,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});

    //////////////
    ///
    ///  Write the output to tiff
    ///
    //////////////////

    apr.name = options.output;

    std::string output_loc = options.directory + apr.name + "_ray_cast_apr_views.tif";
    TiffUtils::saveMeshAsTiff(output_loc, views);

    if(options.original_image.size() > 0){

        TiffUtils::TiffInfo inputTiff(options.directory + options.original_image);
        MeshData<uint16_t> original_image = TiffUtils::getMesh<uint16_t>(inputTiff);

        MeshData<uint16_t> mesh_views;

        apr_raycaster.perpsective_mesh_raycast(original_image,mesh_views);

        output_loc = options.directory + apr.name + "_ray_cast_mesh_views.tif";
        TiffUtils::saveMeshAsTiff(output_loc, mesh_views);

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
        std::cerr << usage << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        std::cerr << "Usage: \"Example_ray_cast -i inputfile [-d directory] [-aniso z_stretch] [-jitter jitter_factor] [-numviews number_views]\"" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-aniso"))
    {
        result.aniso = std::stof(std::string(get_command_option(argv, argv + argc, "-aniso")));
    }


    if(command_option_exists(argv, argv + argc, "-view_radius"))
    {
        result.view_radius = std::stof(std::string(get_command_option(argv, argv + argc, "-view_radius")));
    }

    if(command_option_exists(argv, argv + argc, "-jitter"))
    {
        result.jitter = std::stof(std::string(get_command_option(argv, argv + argc, "-jitter")));
    }

    if(command_option_exists(argv, argv + argc, "-numviews"))
    {
        result.num_views = std::stoi(std::string(get_command_option(argv, argv + argc, "-numviews")));
    }

    if(command_option_exists(argv, argv + argc, "-original_image"))
    {
        result.original_image = std::string(get_command_option(argv, argv + argc, "-original_image"));
    }

    return result;

}
