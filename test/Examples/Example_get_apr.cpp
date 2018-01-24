//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Form the APR form image: Takes an uint16_t input tiff image and forms the APR and saves it as hdf5. The hdf5 output of this program
/// can be used with the other apr examples, and also viewed with HDFView.
///
/// Usage:
///
/// (minimal with auto-parameters)
///
/// Example_get_apr -i input_image_tiff -d input_directory [-o name_of_output]
///
/// Additional settings (High Level):
///
/// -I_th intensity_threshold  (will ignore areas of image below this threshold, useful for removing camera artifacts or auto-flouresence)
/// -SNR_min minimal_snr (minimal ratio of the signal to the standard deviation of the background, set by default to 6)
/// -u8bit_img (required flag for using uint8_t tif image as input)
/// -float_img (required flag for using float tif image as input)
///
/// Advanced (Direct) Settings:
///
/// -lambda lambda_value (directly set the value of the gradient smoothing parameter lambda, default: 3)
/// -min_signal min_signal_val (directly sets a minimum absolute signal size relative to the local background, also useful for removing background, otherwise set using noise estimate)
/// -mask_file mask_file_tiff (takes an input image uint16_t, assumes all zero regions should be ignored by the APR, useful for pre-processing of isolating desired content, or using another channel as a mask)
/// -rel_error rel_error_value (Reasonable ranges are from .08-.15), Default: 0.1
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "Example_get_apr.h"

#include "test/utils.h"
#include "src/io/TiffUtils.hpp"

int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options;

    options = read_command_line_options(argc,argv);

    //the apr datastructure
    APR<uint16_t> apr;

    APRConverter<uint16_t> apr_converter;

    //read in the command line options into the parameters file
    apr_converter.par.Ip_th = options.Ip_th;
    apr_converter.par.rel_error = options.rel_error;
    apr_converter.par.lambda = options.lambda;
    apr_converter.par.mask_file = options.mask_file;
    apr_converter.par.min_signal = options.min_signal;
    apr_converter.par.SNR_min = options.SNR_min;
    apr_converter.image_type = options.img_type;

    //where things are
    apr_converter.par.input_image_name = options.input;
    apr_converter.par.input_dir = options.directory;
    apr_converter.par.name = options.output;
    apr_converter.par.output_dir = options.output_dir;

    //Gets the APR
    if(apr_converter.get_apr(apr)){

        //output
        std::string save_loc = options.output_dir;
        std::string file_name = options.output;

        Part_timer timer;

        timer.verbose_flag = true;


        MeshData<uint16_t> level;

        apr.interp_depth_ds(level);

        std::string output_path = save_loc + file_name + "_level.tif";
        //write output as tiff
        TiffUtils::saveMeshAsTiff(output_path, level);


        timer.start_timer("writing output");

        //write the APR to hdf5 file
        apr.write_apr(save_loc,file_name);

        timer.stop_timer();


        } else {
        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
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
        std::cerr << "Usage: \"Example_get_apr -i inputfile [-t] [-s statsfile -d directory] [-o outputfile]\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-u8bit_img"))
    {
        result.img_type = "uint8";
    } else if(command_option_exists(argv, argv + argc, "-float_img")){
        result.img_type = "float";
    }


    return result;

}