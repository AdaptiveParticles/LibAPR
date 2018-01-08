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

int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options;

    //the apr datastructure
    APR<float> apr;

    //Gets the APR
    if(get_apr(argc,argv,apr,options)){

        //output
        std::string save_loc = options.output_dir;
        std::string file_name = options.output;

        Part_timer timer;

        timer.verbose_flag = true;

        timer.start_timer("writing output");

        //write the APR to hdf5 file
        apr.write_apr(save_loc,file_name);

        timer.stop_timer();

        Mesh_data<uint16_t> level;

        apr.interp_depth(level);

        std::string output_path = save_loc + file_name + "_level.tif";

        //write output as tiff
        level.write_image_tiff(output_path);

        APR_converter<float> apr_converter;



    } else {
        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
    }

}


