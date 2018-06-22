//
// Created by cheesema on 14/03/17.
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

    timer.start_timer("read input");
    //read file
    std::string file_name = options.directory + options.input;
    apr.read_apr(file_name);
    apr.name = options.output;
    timer.stop_timer();

    // Intentionaly block-scoped since local recon_pc will be destructed when block ends and release memory.
    {

        if(options.output_pc_recon) {
            //create mesh data structure for reconstruction
            PixelData<uint16_t> recon_pc;

            timer.start_timer("pc interp");
            //perform piece-wise constant interpolation
            apr.interp_img(recon_pc, apr.particles_intensities);
            timer.stop_timer();

            float elapsed_seconds = timer.t2 - timer.t1;
            std::cout << "PC recon "
                      << (recon_pc.x_num * recon_pc.y_num * recon_pc.z_num * 2) / (elapsed_seconds * 1000000.0f)
                      << " MB per second" << std::endl;

            //write output as tiff
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_pc.tif", recon_pc);
        }
    }

    //////////////////////////
    /// Create a particle dataset with the particle type and pc construct it
    ////////////////////////////

    if(options.output_spatial_properties) {

        //initialization of the iteration structures
        APRIterator<uint16_t> apr_iterator(apr); //this is required for parallel access

        //create particle dataset
        ExtraParticleData<uint16_t> type(apr.total_number_particles());
        ExtraParticleData<uint16_t> level(apr.total_number_particles());

        ExtraParticleData<uint16_t> x(apr.total_number_particles());
        ExtraParticleData<uint16_t> y(apr.total_number_particles());
        ExtraParticleData<uint16_t> z(apr.total_number_particles());

        timer.start_timer("APR parallel iterator loop");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(apr_iterator)
#endif
        for (uint64_t particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            type[apr_iterator] = apr_iterator.type();
            level[apr_iterator] = apr_iterator.level();

            x[apr_iterator] = apr_iterator.x();
            y[apr_iterator] = apr_iterator.y();
            z[apr_iterator] = apr_iterator.z();
        }
        timer.stop_timer();

        // Intentionaly block-scoped since local type_recon will be destructed when block ends and release memory.
        {
            PixelData<uint16_t> type_recon;

            apr.interp_img(type_recon, type);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_type.tif", type_recon);

            //pc interp
            apr.interp_img(type_recon, level);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_level.tif", type_recon);

            //pc interp
            apr.interp_img(type_recon, x);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_x.tif", type_recon);

            //pc interp
            apr.interp_img(type_recon, y);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_y.tif", type_recon);

            //pc interp
            apr.interp_img(type_recon, z);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_z.tif", type_recon);
        }
    }

    if(options.output_smooth_recon) {

        //smooth reconstruction - requires float
        PixelData<float> recon_smooth;
        std::vector<float> scale_d = {2, 2, 2};

        timer.start_timer("smooth reconstrution");
        apr.interp_parts_smooth(recon_smooth, apr.particles_intensities, scale_d);
        timer.stop_timer();

        float elapsed_seconds = timer.t2 - timer.t1;
        std::cout << "Smooth recon "
                  << (recon_smooth.x_num * recon_smooth.y_num * recon_smooth.z_num * 2) / (elapsed_seconds * 1000000.0f)
                  << " MB per second" << std::endl;

        //write to tiff casting to unsigned 16 bit integer
        TiffUtils::saveMeshAsTiffUint16(options.directory + apr.name + "_smooth.tif", recon_smooth);
    }
}
