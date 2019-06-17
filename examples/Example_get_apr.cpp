//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///

const char* usage = R"(
Form the APR form image: Takes an uint16_t input tiff image and forms the APR and saves it as hdf5. The hdf5 output of this program
can be used with the other apr examples, and also viewed with HDFView.

Usage:

(minimal with auto-parameters)

Example_get_apr -i input_image_tiff -d input_directory [-o name_of_output]

Additional settings (High Level):

-I_th intensity_threshold  (will ignore areas of image below this threshold, useful for removing camera artifacts or auto-flouresence)
-SNR_min minimal_snr (minimal ratio of the signal to the standard deviation of the background, set by default to 6)

Advanced (Direct) Settings:

-lambda lambda_value (directly set the value of the gradient smoothing parameter lambda (reasonable range 0.1-10, default: 3)
-min_signal min_signal_val (directly sets a minimum absolute signal size relative to the local background, also useful for removing background, otherwise set using estimated background noise estimate and minimal SNR of 6)
-mask_file mask_file_tiff (takes an input image uint16_t, assumes all zero regions should be ignored by the APR, useful for pre-processing of isolating desired content, or using another channel as a mask)
-rel_error rel_error_value (Reasonable ranges are from .08-.15), Default: 0.1
-normalize_input (flag that will rescale the input from the input data range to 80% of the output data type range, useful for float scaled datasets)
-compress_level (the IO uses BLOSC for lossless compression of the APR, this can be set from 1-9, where higher increases the compression level. Note, this can come at a significant time increase.)
-compress_type (Default: 0, loss-less compression of partilce intensities, (1,2) WNL (Balázs et al. 2017) - approach compression applied to particles (1 = without prediction, 2 = with)

-neighborhood_optimization_off turns off the neighborhood opetimization (This results in boundary Particle Cells also being increased in resolution after the Pulling Scheme step)
-output_steps Writes tiff images of the individual steps (gradient magnitude, local intensity scale, and final level of the APR calculation).

)";

#include <algorithm>
#include <iostream>
#include "ConfigAPR.h"
#include "Example_get_apr.h"
#include "io/APRFile.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/APR/APR.hpp"
#include "algorithm/APRConverter.hpp"

#include <future>
#include <thread>

int runAPR(cmdLineOptions options) {
    //the apr datastructure
    APR apr;

    APRConverter<uint16_t> aprConverter;

    //read in the command line options into the parameters file
    aprConverter.par.Ip_th = options.Ip_th;
    aprConverter.par.rel_error = options.rel_error;
    aprConverter.par.lambda = options.lambda;
    aprConverter.par.mask_file = options.mask_file;
    aprConverter.par.min_signal = options.min_signal;
    aprConverter.par.SNR_min = options.SNR_min;
    aprConverter.par.normalized_input = options.normalize_input;
    aprConverter.par.neighborhood_optimization = options.neighborhood_optimization;
    aprConverter.par.output_steps = options.output_steps;

    //where things are
    aprConverter.par.input_image_name = options.input;
    aprConverter.par.input_dir = options.directory;
    aprConverter.par.name = options.output;
    aprConverter.par.output_dir = options.output_dir;

    aprConverter.fine_grained_timer.verbose_flag = false;
    aprConverter.method_timer.verbose_flag = false;
    aprConverter.computation_timer.verbose_flag = false;
    aprConverter.allocation_timer.verbose_flag = false;
    aprConverter.total_timer.verbose_flag = true;

    PixelData<uint16_t> input_img = TiffUtils::getMesh<uint16_t>(options.directory + options.input);

    //Gets the APR
    if(aprConverter.get_apr(apr, input_img)){

        ParticleData<uint16_t> particle_intensities;
        particle_intensities.sample_parts_from_img_downsampled(apr,input_img); // sample your particles from your image
        //Below is IO and outputting of the Implied Resolution Function through the Particle Cell level.

        //output
        std::string save_loc = options.output_dir;
        std::string file_name = options.output;

        APRTimer timer;

        timer.verbose_flag = true;

        std::cout << std::endl;
        float original_pixel_image_size = (2.0f* apr.org_dims(0)* apr.org_dims(1)* apr.org_dims(2))/(1000000.0);
        std::cout << "Original image size: " << original_pixel_image_size << " MB" << std::endl;

        timer.start_timer("writing output");

        std::cout << "Writing the APR to hdf5..." << std::endl;

        //write the APR to hdf5 file
        APRFile aprFile;

        aprFile.open(save_loc + file_name + ".apr");

        aprFile.write_apr(apr);
        aprFile.write_particles(apr,"intensities",particle_intensities);

        float apr_file_size = aprFile.current_file_size_MB();

        std::cerr << "File size functionality needs to be updated" << std::endl;

        timer.stop_timer();

        float computational_ratio = (1.0f* apr.org_dims(0)* apr.org_dims(1)* apr.org_dims(2))/(1.0f*apr.total_number_particles());

        std::cout << std::endl;
        std::cout << "Computational Ratio (Pixels/Particles): " << computational_ratio << std::endl;
        std::cout << "Lossy Compression Ratio: " << original_pixel_image_size/apr_file_size << std::endl;
        std::cout << std::endl;

        } else {
        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
    }
    return 0;
}


int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options = read_command_line_options(argc, argv);

//    std::vector<std::future<int>> fv;
//    fv.emplace_back(std::async(std::launch::async, [=]{ return runAPR(options); }));
//    fv.emplace_back(std::async(std::launch::async, [=]{ return runAPR(options); }));
//    fv.emplace_back(std::async(std::launch::async, [=]{ return runAPR(options); }));
//    int n = 2;
//    fv[0].
//    std::cout << "Waintig..." <<std::endl;
//    for (int i = 0; i < fv.size()*3; ++i) {
//        fv[i % n].wait();
//        fv[i % n] = std::async(std::launch::async, [=]{ return runAPR(options); });
//    }
//
//    for (int i = 0; i < fv.size(); ++i) fv[i].wait();
//
//    std::cout << "DONE!" <<std::endl;

    return runAPR(options);
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
        std::cerr << "APR version " << ConfigAPR::APR_VERSION << std::endl;
        std::cerr << "Short usage: \"" << argv[0] << " -i inputfile [-d directory] [-o outputfile]\"" << std::endl;

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

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-od"))
    {
        result.output_dir = std::string(get_command_option(argv, argv + argc, "-od"));
    } else {
        result.output_dir = result.directory;
    }

    if(command_option_exists(argv, argv + argc, "-gt"))
    {
        result.gt_input = std::string(get_command_option(argv, argv + argc, "-gt"));
    } else {
        result.gt_input = "";
    }

    if(command_option_exists(argv, argv + argc, "-lambda"))
    {
        result.lambda = std::stof(std::string(get_command_option(argv, argv + argc, "-lambda")));
    }

    if(command_option_exists(argv, argv + argc, "-I_th"))
    {
        result.Ip_th = std::stof(std::string(get_command_option(argv, argv + argc, "-I_th")));
    }

    if(command_option_exists(argv, argv + argc, "-SNR_min"))
    {
        result.SNR_min = std::stof(std::string(get_command_option(argv, argv + argc, "-SNR_min")));
    }

    if(command_option_exists(argv, argv + argc, "-min_signal"))
    {
        result.min_signal = std::stof(std::string(get_command_option(argv, argv + argc, "-min_signal")));
    }

    if(command_option_exists(argv, argv + argc, "-rel_error"))
    {
        result.rel_error = std::stof(std::string(get_command_option(argv, argv + argc, "-rel_error")));
    }

    if(command_option_exists(argv, argv + argc, "-mask_file"))
    {
        result.mask_file = std::string(get_command_option(argv, argv + argc, "-mask_file"));
    }

    if(command_option_exists(argv, argv + argc, "-compress_level"))
    {
        result.compress_level = (unsigned int)std::stoi(std::string(get_command_option(argv, argv + argc, "-compress_level")));
    }

    if(command_option_exists(argv, argv + argc, "-compress_type"))
    {
        result.compress_type = (unsigned int)std::stoi(std::string(get_command_option(argv, argv + argc, "-compress_type")));
    }

    if(command_option_exists(argv, argv + argc, "-quantization_factor"))
    {
        result.quantization_factor = (float)std::stof(std::string(get_command_option(argv, argv + argc, "-quantization_factor")));
    }

    if(command_option_exists(argv, argv + argc, "-normalize_input"))
    {
        result.normalize_input = true;
    }

    if(command_option_exists(argv, argv + argc, "-neighborhood_optimization_off"))
    {
        result.neighborhood_optimization = false;

    }

    if(command_option_exists(argv, argv + argc, "-output_steps"))
    {
        result.output_steps = true;
    }

    if(command_option_exists(argv, argv + argc, "-store_tree"))
    {
        result.store_tree = true;
    }

    return result;
}
