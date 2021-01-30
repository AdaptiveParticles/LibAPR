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

(using *.apr output of Example_get_apr)

Example_reconstruct_image -i inputfile [-d directory] -o output_name

e.g. Example_reconstruct_image -i nuclei.apr -d /Test/Input_examples/ -o nuclei

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
#include"data_structures/APR/particles/ParticleData.hpp"
#include"io/APRFile.hpp"
#include "numerics/APRReconstruction.hpp"
#include <random>

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
template<typename T>
void add_random_to_img(PixelData<T>& img,float sd){

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,sd);

    size_t size = (size_t)img.y_num * img.x_num * img.z_num;
    size_t i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
    for(i=0; i < size; i++){
        float number = distribution(generator);
        img.mesh[i] += number;
    }

}

int main(int argc, char **argv) {
    // INPUT PARSING
    cmdLineOptions options = read_command_line_options(argc, argv);

    // Read the apr file into the part cell structure
    APRTimer timer;
    timer.verbose_flag = true;

    // APR datastructure
    APR apr;

    timer.start_timer("read input");
    //read file
    std::string file_name = options.directory + options.input;
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);

    ParticleData<uint16_t>parts;
    aprFile.read_particles(apr,parts);

    aprFile.close();
    apr.name = options.output;
    timer.stop_timer();

    // Intentionaly block-scoped since local recon_pc will be destructed when block ends and release memory.
    {

        if(options.output_pc_recon) {
            //create mesh data structure for reconstruction
            bool add_random_gitter = true;

            PixelData<uint16_t> recon_pc;

            timer.start_timer("pc interp");
            //perform piece-wise constant interpolation
            APRReconstruction::interp_img(apr,recon_pc, parts);
            timer.stop_timer();

            if(add_random_gitter){
                add_random_to_img(recon_pc,1.0f);
            }

            float elapsed_seconds = timer.t2 - timer.t1;
            std::cout << "PC recon "
                      << (recon_pc.x_num * recon_pc.y_num * recon_pc.z_num * 2) / (elapsed_seconds * 1000000.0f)
                      << " MB per second" << std::endl;

            // write output as tiff
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_pc.tif", recon_pc);
        }
    }

    //////////////////////////
    /// Create a particle dataset with the particle type and pc construct it
    ////////////////////////////

    if(options.output_spatial_properties) {

        //initialization of the iteration structures
        //this is required for parallel access
        auto apr_iterator = apr.iterator();

        //create particle dataset

        ParticleData<uint16_t> levelp(apr.total_number_particles());

        ParticleData<uint16_t> xp(apr.total_number_particles());
        ParticleData<uint16_t> yp(apr.total_number_particles());
        ParticleData<uint16_t> zp(apr.total_number_particles());


        timer.start_timer("APR parallel iterator loop");
        for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.z_num(level); z++) {
                for (x = 0; x < apr_iterator.x_num(level); ++x) {
                    for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                         apr_iterator++) {

                        levelp[apr_iterator] = level;

                        xp[apr_iterator] = x;
                        yp[apr_iterator] = apr_iterator.y();
                        zp[apr_iterator] = z;
                    }
                }
            }
        }
        timer.stop_timer();

        // Intentionaly block-scoped since local type_recon will be destructed when block ends and release memory.
        {
            PixelData<uint16_t> type_recon;

            //pc interp
            APRReconstruction::interp_img(apr,type_recon, levelp);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_level.tif", type_recon);

            //pc interp
            APRReconstruction::interp_img(apr,type_recon, xp);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_x.tif", type_recon);

            //pc interp
            APRReconstruction::interp_img(apr,type_recon, yp);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_y.tif", type_recon);

            //pc interp
            APRReconstruction::interp_img(apr,type_recon, zp);
            TiffUtils::saveMeshAsTiff(options.directory + apr.name + "_z.tif", type_recon);
        }
    }

    if(options.output_smooth_recon) {

        //smooth reconstruction - requires float
        PixelData<float> recon_smooth;
        std::vector<float> scale_d = {2, 2, 2};

        timer.start_timer("smooth reconstrution");
        APRReconstruction::interp_parts_smooth(apr,recon_smooth, parts, scale_d); //#TODO: i'm not convinced this is working correclty.
        timer.stop_timer();

        float elapsed_seconds = timer.t2 - timer.t1;
        std::cout << "Smooth recon "
                  << (recon_smooth.x_num * recon_smooth.y_num * recon_smooth.z_num * 2) / (elapsed_seconds * 1000000.0f)
                  << " MB per second" << std::endl;

        //write to tiff casting to unsigned 16 bit integer
        TiffUtils::saveMeshAsTiffUint16(options.directory + apr.name + "_smooth.tif", recon_smooth);
    }
}
