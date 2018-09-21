//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_APR_PARAMETERS_HPP
#define PARTPLAY_APR_PARAMETERS_HPP

#include <string>

class APRParameters {

public:

    // pixel spacing
    float dx = 1;
    float dy = 1;
    float dz = 1;

    // window size set for Local Intensity Scale
    float psfx = 2;
    float psfy = 2;
    float psfz = 2;

    float Ip_th = 0;
    float SNR_min = 0;
    float lambda = 3.0;
    float min_signal = 0;
    float rel_error = 0.1;

    float sigma_th = 0;
    float sigma_th_max = 0;

    float noise_sd_estimate = 0;
    float background_intensity_estimate = 0;

    bool auto_parameters = true;

    bool normalized_input = false;

    bool neighborhood_optimization = true;

    bool output_steps = false;

    std::string name;
    std::string output_dir;
    std::string input_image_name;
    std::string input_dir;
    std::string mask_file;
};


#endif //PARTPLAY_APR_PARAMETERS_HPP
