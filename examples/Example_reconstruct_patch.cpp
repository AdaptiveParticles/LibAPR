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
-apr_properties (Outputs all Particle Cell information (x,y,z,l) and type to pc images
-level (outputs the particle cell level as an image (the resolution function everywhere))
-x_begin / x_end (starting/end index of local patch reconstruction)
-y_begin / y_end (starting/end index of local patch reconstruction)
-z_begin / z_end (starting/end index of local patch reconstruction) *if any are left blank the full range will be reconstructed
-level_delta (negative down-samples, positive upsamples)

)";

#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"
#include "io/TiffUtils.hpp"
#include "numerics/APRTreeNumerics.hpp"

struct cmdLineOptions{
    std::string output = "output";
    std::string directory = "";
    std::string input = "";
    bool output_spatial_properties = false;
    bool output_pc_recon = false;
    bool output_smooth_recon = false;
    bool output_level = false;
    int x_begin = 0;
    int x_end = -1;
    int y_begin = 0;
    int y_end = -1;
    int z_begin = 0;
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

    if (command_option_exists(argv, argv + argc, "-level")) {
        result.output_level = true;
    }

    if(!(result.output_pc_recon || result.output_smooth_recon || result.output_spatial_properties || result.output_level)){
        //default is pc recon
        result.output_pc_recon = true;
    }

    if(command_option_exists(argv, argv + argc, "-x_begin"))
    {
        result.x_begin = std::stoi(std::string(get_command_option(argv, argv + argc, "-x_begin")));
    }

    if(command_option_exists(argv, argv + argc, "-x_end"))
    {
        result.x_end = std::stoi(std::string(get_command_option(argv, argv + argc, "-x_end")));
    }

    if(command_option_exists(argv, argv + argc, "-y_begin"))
    {
        result.y_begin = std::stoi(std::string(get_command_option(argv, argv + argc, "-y_begin")));
    }

    if(command_option_exists(argv, argv + argc, "-y_end"))
    {
        result.y_end = std::stoi(std::string(get_command_option(argv, argv + argc, "-y_end")));
    }

    if(command_option_exists(argv, argv + argc, "-z_begin"))
    {
        result.z_begin = std::stoi(std::string(get_command_option(argv, argv + argc, "-z_begin")));
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
    timer.start_timer("read input");
    std::string file_name = options.directory + options.input;

    unsigned int read_delta = (unsigned int) std::max(0,-options.level_delta);

    //test of the two level read
    apr.read_apr(file_name,true,read_delta+1);
    apr.name = options.output;
    timer.stop_timer();

    timer.start_timer("read input 2");
    apr.read_apr(file_name,true,read_delta);
    timer.stop_timer();

    APRReconstruction aprReconstruction;

    ReconPatch reconPatch;

    reconPatch.x_begin = options.x_begin;
    reconPatch.x_end = options.x_end;

    reconPatch.y_begin = options.y_begin;
    reconPatch.y_end = options.y_end;

    reconPatch.z_begin = options.z_begin;
    reconPatch.z_end = options.z_end;

    reconPatch.level_delta = options.level_delta;

    timer.start_timer("init tree full");
    //apr.apr_tree.init(apr);
    timer.stop_timer();

    //options.output_pc_recon = false;

    //ExtraParticleData<uint16_t> partsTree;

    //timer.start_timer("fill tree");

    //APRTreeNumerics::fill_tree_mean(apr,apr.apr_tree,apr.particles_intensities,partsTree);
    //timer.stop_timer();

    // Intentionaly block-scoped since local recon_pc will be destructed when block ends and release memory.
    {

        if(options.output_pc_recon) {
            //create mesh data structure for reconstruction
            PixelData<uint16_t> recon_pc;



            //std::cout << partsTree.data.size()/(1.0f*apr.total_number_particles()) << std::endl;

            //APRWriter aprWriter;
            //float file_s = aprWriter.write_particles_only(options.directory,"tree_parts",partsTree);

            //std::cout << "File Size: " << file_s << std::endl;

            //APRTreeNumerics::fill_tree_from_particles(apr,aprTree,apr.particles_intensities,partsTree,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});

//            APRTreeNumerics::fill_tree_from_particles(apr, aprTree, apr.particles_intensities, partsTree,
//                    [](const uint16_t &a, const uint16_t &b) { return a + b; }, true);

            //partsTree = APRTreeNumerics::meanDownsampling<uint16_t,float>(apr,aprTree);



            timer.start_timer("pc interp");
            //perform piece-wise constant interpolation

            aprReconstruction.interp_image_patch(apr, apr.apr_tree, recon_pc, apr.particles_intensities, apr.apr_tree.particles_ds_tree,
                                                     reconPatch);

            timer.stop_timer();

            float elapsed_seconds = timer.t2 - timer.t1;
            std::cout << "PC recon "
                      << (recon_pc.x_num * recon_pc.y_num * recon_pc.z_num * 2) / (elapsed_seconds * 1000000.0f)
                      << " MB per second" << std::endl;

            //write output as tiff
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_pc.tif", recon_pc);
        }
    }


    {

        if(options.output_smooth_recon) {
            //create mesh data structure for reconstruction
            PixelData<uint16_t> recon_pc;

            ExtraParticleData<uint16_t> partsTree;

            std::vector<float> scale = {1,1,1};

            APRTreeNumerics::fill_tree_from_particles(apr,apr.apr_tree,apr.particles_intensities,partsTree,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});

            timer.start_timer("smooth interp");
            //perform smooth interpolation
            aprReconstruction.interp_parts_smooth_patch(apr,apr.apr_tree,recon_pc, apr.particles_intensities,partsTree,reconPatch,scale);
            timer.stop_timer();

            float elapsed_seconds = timer.t2 - timer.t1;
            std::cout << "Smooth recon "
                      << (recon_pc.x_num * recon_pc.y_num * recon_pc.z_num * 2) / (elapsed_seconds * 1000000.0f)
                      << " MB per second" << std::endl;

            //write output as tiff
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_smooth.tif", recon_pc);
        }
    }

    //////////////////////////
    /// Create a particle dataset with the particle type and pc construct it
    ////////////////////////////


    if(options.output_level) {

        //initialization of the iteration structures
        APRIterator<uint16_t> apr_iterator(apr); //this is required for parallel access

        //create particle dataset

        ExtraParticleData<uint16_t> level(apr);

        timer.start_timer("APR parallel iterator loop");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(apr_iterator)
#endif
        for (uint64_t particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            level[apr_iterator] = apr_iterator.level();
        }
        timer.stop_timer();

        // Intentionaly block-scoped since local type_recon will be destructed when block ends and release memory.
        {
            PixelData<uint8_t> type_recon;

            //level
            //aprReconstruction.interp_image_patch(apr,aprTree,type_recon, level,reconPatch);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_level.tif", type_recon);

        }
    }


}

