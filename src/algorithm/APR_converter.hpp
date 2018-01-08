////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// APR Converter class handles the methods and functions for creating an APR from an input image
///
////////////////////////////////

#ifndef PARTPLAY_APR_CONVERTER_HPP
#define PARTPLAY_APR_CONVERTER_HPP

#include "src/data_structures/Mesh/meshclass.h"
#include "src/data_structures/APR/APR.hpp"

template<typename ImageType>
class APR_converter {

public:

    APR_converter(){

    }


    /*
     * Declerations
     */


    bool get_apr(int argc, char **argv,APR<float>& apr,cmdLineOptions& options);












};

/*
 * Implimentations
 */
template<typename ImageType>
bool APR_converter<ImageType>::get_apr(int argc, char **argv,APR<float>& apr,cmdLineOptions& options) {

    // INPUT PARSING

//    options = read_command_line_options(argc, argv, apr.pars);

//    Part_timer timer;
//    timer.verbose_flag = true;
//
//    timer.start_timer("read tif input image");

    Mesh_data<ImageType> input_image;

    input_image.load_image_tiff(options.directory + options.input);


    //load_image_tiff(input_image, options.directory + options.input);

//    timer.stop_timer();

//        if (!options.stats_file) {
//            // defaults
//
//            apr.pars.dy = apr.pars.dx = apr.pars.dz = 1;
//            apr.pars.psfx = apr.pars.psfy = apr.pars.psfz = 2;
//            //apr.pars.rel_error = 0.1;
//            apr.pars.var_th = 0;
//            apr.pars.var_th_max = 0;
//
//            // setting the command line options
//            apr.pars.I_th = options.Ip_th;
//            apr.pars.noise_scale = options.SNR_min;
//            apr.pars.lambda = options.lambda;
//            apr.pars.var_th = options.min_signal;
//
//            if(options.mask_file != "") {
//                apr.pars.mask_file = options.directory + options.mask_file;
//            }
//
//            if(input_image.mesh.size() == 0){
//                std::cout << "Image Not Found" << std::endl;
//
//                return false;
//            }
//
//            timer.start_timer("calculate automatic parameters");
//            auto_parameters(input_image,apr.pars);
//            timer.stop_timer();
//
//            apr.pars.name = options.output;
//            apr.name = options.output;
//
//            //return false;
//        }
//
//        get_apr(input_image,apr);

    return true;
}


#endif //PARTPLAY_APR_CONVERTER_HPP
