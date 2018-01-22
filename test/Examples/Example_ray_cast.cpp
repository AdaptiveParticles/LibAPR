////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Ray cast example:
///
/// Creates a maximum projection ray cast from an APR and outputs to a tiff image.
///
/// Usage:
///
/// Example_ray_cast -i inputfile [-d directory]
///
/// Optional:
///
/// -aniso z_stretch (stretches the ray cast in the z axis)
/// -jitter jitter_factor (0-1) (perturbs the particles randomly in the ray case, in an effort to remove artifacts from alignement of view in the ray-cast)
/// -numviews The number of views that are calculated and output to the tiff file.
///
/// e.g. Example_ray_cast -i nuc_apr.h5 -d /Test/Input_examples/ -aniso 2.0 -jitter 0.1 -numviews 60
///
////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "Example_ray_cast.h"

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
        std::cerr << "Usage: \"Example_ray_cast -i inputfile [-d directory] [-aniso z_stretch] [-jitter jitter_factor] [-numviews number_views]\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-jitter"))
    {
        result.jitter = std::stof(std::string(get_command_option(argv, argv + argc, "-jitter")));
    }

    if(command_option_exists(argv, argv + argc, "-numviews"))
    {
        result.num_views = std::stoi(std::string(get_command_option(argv, argv + argc, "-numviews")));
    }

    return result;

}


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

    apr_raycaster.theta_0 = -3.14; //start
    apr_raycaster.theta_final = 3.14; //stop radians
    apr_raycaster.radius_factor = .98f; //radius scaling
    apr_raycaster.theta_delta = (apr_raycaster.theta_final - apr_raycaster.theta_0)/(options.num_views*1.0); //steps
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

    std::string output_loc = options.directory + apr.name + "_ray_cast_views.tif";

    views.write_image_tiff(output_loc);

}
