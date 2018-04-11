//
// Created by cheesema on 10.04.18.
//

////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///

const char* usage = R"(
APR pixel image reconstruction example:

Outputs various reconstructed images from the APR.

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_reconstruct_image -i inputfile [-d directory] -o output_name

e.g. Example_reconstruct_image -i nuc_apr.h5 -d /Test/Input_examples/ -o nuclei

Default: Piece-wise constant reconstruction

Options:

-pc_recon (outputs piece-wise reconstruction (Default))
-smooth_recon (Outputs a smooth reconstruction)
-apr_properties (Outputs all Particle Cell information (x,y,z,l) and type to pc images

)";


#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"
#include "io/TiffUtils.hpp"


struct cmdLineOptions{
    std::string output = "output";
    std::string directory = "";
    std::string input = "";
    bool output_spatial_properties = false;
    bool output_pc_recon = false;
    bool output_smooth_recon = false;
    int x_start = 0;
    int x_end = -1;
    int y_start = 0;
    int y_end = -1;
    int z_start = 0;
    int z_end = -1;
    int level_delta = 0;

};

static bool command_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}


static const char* get_command_option(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return nullptr;
}

static cmdLineOptions read_command_line_options(int argc, char **argv) {
    cmdLineOptions result;

    if (argc == 1) {
        std::cerr << usage << std::endl;
        exit(1);
    }

    if (command_option_exists(argv, argv + argc, "-i")) {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    }
    else {
        std::cerr << "Input file required" << std::endl;
        exit(2);
    }

    if (command_option_exists(argv, argv + argc, "-d")) {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if (command_option_exists(argv, argv + argc, "-o")) {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if (command_option_exists(argv, argv + argc, "-pc_recon")) {
        result.output_pc_recon = true;
    }

    if (command_option_exists(argv, argv + argc, "-smooth_recon")) {
        result.output_smooth_recon = true;
    }

    if (command_option_exists(argv, argv + argc, "-apr_properties")) {
        result.output_spatial_properties = true;
    }

    if(!(result.output_pc_recon || result.output_smooth_recon || result.output_spatial_properties)){
        //default is pc recon
        result.output_pc_recon = true;
    }

    if(command_option_exists(argv, argv + argc, "-x_start"))
    {
        result.x_start = std::stoi(std::string(get_command_option(argv, argv + argc, "-x_start")));
    }

    if(command_option_exists(argv, argv + argc, "-x_end"))
    {
        result.x_end = std::stoi(std::string(get_command_option(argv, argv + argc, "-x_end")));
    }

    if(command_option_exists(argv, argv + argc, "-y_start"))
    {
        result.y_start = std::stoi(std::string(get_command_option(argv, argv + argc, "-y_start")));
    }

    if(command_option_exists(argv, argv + argc, "-y_end"))
    {
        result.y_end = std::stoi(std::string(get_command_option(argv, argv + argc, "-y_end")));
    }

    if(command_option_exists(argv, argv + argc, "-z_start"))
    {
        result.z_start = std::stoi(std::string(get_command_option(argv, argv + argc, "-z_start")));
    }

    if(command_option_exists(argv, argv + argc, "-z_end"))
    {
        result.z_end = std::stoi(std::string(get_command_option(argv, argv + argc, "-z_end")));
    }

    if(command_option_exists(argv, argv + argc, "-level_delta"))
    {
        result.level_delta = std::stoi(std::string(get_command_option(argv, argv + argc, "-level_delta")));
    }

    return result;
}

int main(int argc, char **argv) {
    // INPUT PARSING
    cmdLineOptions options = read_command_line_options(argc, argv);

    // Read the apr file into the part cell structure
    APRTimer timer;
    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    std::string file_name = options.directory + options.input;
    apr.read_apr(file_name);
    apr.name = options.output;

    APRReconstruction aprReconstruction;

    // Intentionaly block-scoped since local recon_pc will be destructed when block ends and release memory.
    {

        if(options.output_pc_recon) {
            //create mesh data structure for reconstruction
            MeshData<uint16_t> recon_pc;

            ReconPatch reconPatch;

            timer.start_timer("pc interp");
            //perform piece-wise constant interpolation
            aprReconstruction.interp_image_patch(apr,recon_pc, apr.particles_intensities,reconPatch);
            timer.stop_timer();

            float elapsed_seconds = timer.t2 - timer.t1;
            std::cout << "PC recon "
                      << (recon_pc.x_num * recon_pc.y_num * recon_pc.z_num * 2) / (elapsed_seconds * 1000000.0f)
                      << " MB per second" << std::endl;

            //write output as tiff
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_pc.tif", recon_pc);
        }
    }



}

