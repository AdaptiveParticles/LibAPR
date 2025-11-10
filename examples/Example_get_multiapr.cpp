const char* usage = R"(
Form the APR form images: Takes an uint16_t input tiff images and forms the APRs and saves it as hdf5.
The hdf5 output of this program can be used with the other apr examples, and also viewed with HDFView.

Usage:
======
Example_get_multiapr -d input_directory [-od output_direcotry]

Additional settings (High Level):
=================================
-I_th       intensity_threshold (will ignore areas of image below this threshold, useful for removing camera artifacts or auto-fluorescence)
-sigma_th   lower threshold for the local intensity scale
-grad_th    ignore areas in the image where the gradient magnitude is lower than this value

Advanced (Direct) Settings:
===========================
-lambda lambda_value (directly set the value of the gradient smoothing parameter lambda (reasonable range 0.1-10, default: 3)
-rel_error rel_error_value (Reasonable ranges are from .08-.15), Default: 0.1
-neighborhood_optimization_off turns off the neighborhood optimization (This results in boundary Particle Cells also being increased in resolution after the Pulling Scheme step)
)";

#include <algorithm>
#include <iostream>
#include "ConfigAPR.h"
#include "io/APRFile.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/APR/APR.hpp"
#include "algorithm/APRConverter.hpp"


struct cmdLineOptions {
    std::string directory = "";
    std::string output_dir = "";

    float lambda = 3.0;
    float Ip_th = 0;
    float grad_th = 1;
    float sigma_th = 0;
    float rel_error = 0.1;

    bool neighborhood_optimization = true;
};

bool command_option_exists(const char **begin, const char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

const char* get_command_option(const char **begin, const char **end, const std::string &option)
{
    if (const char** itr = std::find(begin, end, option); itr != end && ++itr != end) {
        return *itr;
    }
    return nullptr;
}

void printUsage() {
    std::cerr << "APR version " << ConfigAPR::APR_VERSION << std::endl <<std::endl;
    std::cerr << usage << std::endl;
    exit(1);
}

cmdLineOptions read_command_line_options(const int argc, const char **argv) {

    cmdLineOptions options;

    // --------- Print usage if no args provided
    if (argc == 1) printUsage();

    // --------- Read params

    // Input Directory
    if (command_option_exists(argv, argv + argc, "-d")) {
        options.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    } else {
        std::cout << "Input directory required" << std::endl;
        exit(2);
    }

    // Output Directory
    if (command_option_exists(argv, argv + argc, "-od")) {
        options.output_dir = std::string(get_command_option(argv, argv + argc, "-od"));
    } else {
        options.output_dir = options.directory;
    }

    if (command_option_exists(argv, argv + argc, "-lambda")) {
        options.lambda = std::stof(std::string(get_command_option(argv, argv + argc, "-lambda")));
    }

    if (command_option_exists(argv, argv + argc, "-I_th")) {
        options.Ip_th = std::stof(std::string(get_command_option(argv, argv + argc, "-I_th")));
    }

    if (command_option_exists(argv, argv + argc, "-grad_th")) {
        options.grad_th = std::stof(std::string(get_command_option(argv, argv + argc, "-grad_th")));
    }

    if (command_option_exists(argv, argv + argc, "-sigma_th")) {
        options.sigma_th = std::stof(std::string(get_command_option(argv, argv + argc, "-sigma_th")));
    }

    if (command_option_exists(argv, argv + argc, "-rel_error")) {
        options.rel_error = std::stof(std::string(get_command_option(argv, argv + argc, "-rel_error")));
    }

    if (command_option_exists(argv, argv + argc, "-neighborhood_optimization_off")) {
        options.neighborhood_optimization = false;
    }

    return options;
}


int runAPR(const cmdLineOptions &options) {

    APRConverter<uint16_t> aprConverter;

    // read in the command line options into the parameters file
    aprConverter.par.input_dir = options.directory;
    aprConverter.par.output_dir = options.output_dir;

    aprConverter.par.lambda = options.lambda;
    aprConverter.par.Ip_th = options.Ip_th;
    aprConverter.par.grad_th = options.grad_th;
    aprConverter.par.sigma_th = options.sigma_th;
    aprConverter.par.rel_error = options.rel_error;

    aprConverter.par.neighborhood_optimization = options.neighborhood_optimization;



    // TODO: read here all input files instead of options.input
    PixelData<uint16_t> input_img = TiffUtils::getMesh<uint16_t>(options.directory + "TODO");

    //Gets the APR
    if(APR apr; aprConverter.get_apr(apr, input_img)){

        ParticleData<uint16_t> particle_intensities;
        particle_intensities.sample_image(apr, input_img); // sample your particles from your image

#ifdef APR_USE_CUDA
        //Below is IO and outputting of the Implied Resolution Function through the Particle Cell level.
        std::cout << apr.linearAccess.y_vec.size() << " particles in APR" << std::endl;
        std::cout << particle_intensities.size() << " intensities in CPU in APR" << std::endl;
        std::cout << aprConverter.parts.size() << " intensities in GPU in APR" << std::endl;

        for (int i = 0 ; i < particle_intensities.size(); ++i) {
            if (particle_intensities[i]  != aprConverter.parts[i]) {
                std::cout << "Mismatch at " << i << " CPU: " << particle_intensities[i] << " GPU: " << aprConverter.parts[i] << std::endl;
            }
        }
#endif


        //output
        std::string save_loc = options.output_dir;
        // TODO Change file_name to currently processed input file and add ".apr"
        std::string file_name = "TODO_fileName";

        APRTimer timer;

        timer.verbose_flag = true;

        std::cout << std::endl;
        float original_pixel_image_size = 2.0f * apr.org_dims(0) * apr.org_dims(1) * apr.org_dims(2) / 1000000.0f;
        std::cout << "Original image size: " << original_pixel_image_size << " MB" << std::endl;

        timer.start_timer("writing output");

        std::cout << "Writing the APR to hdf5..." << std::endl;

        //write the APR to hdf5 file
        APRFile aprFile;

        aprFile.open(save_loc + file_name + ".apr");

        aprFile.write_apr(apr, 0, "t", false);
        aprFile.write_particles("particles",particle_intensities);

        float apr_file_size = aprFile.current_file_size_MB();

        timer.stop_timer();

        float computational_ratio = 1.0f * apr.org_dims(0) * apr.org_dims(1) * apr.org_dims(2) / (1.0f * apr.total_number_particles());

        std::cout << std::endl;
        std::cout << "Computational Ratio (Pixels/Particles): " << computational_ratio << std::endl;
        std::cout << "Lossy Compression Ratio: " << original_pixel_image_size/apr_file_size << std::endl;
        std::cout << std::endl;
    } else {
        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
    }
    return 0;
}


int main(const int argc, const char **argv) {
    const cmdLineOptions options = read_command_line_options(argc, argv);
    const auto result = runAPR(options);

    return result;
}
