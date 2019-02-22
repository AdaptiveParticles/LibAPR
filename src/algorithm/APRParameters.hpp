//
// Created by cheesema on 08.01.18.
//

#ifndef PARTPLAY_APR_PARAMETERS_HPP
#define PARTPLAY_APR_PARAMETERS_HPP

#include <string>
#include <iostream>

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

    friend std::ostream & operator<<(std::ostream &os, const APRParameters &obj) {
        os << "Ip_th=" << obj.Ip_th << "\n";
        os << "SNR_min=" << obj.SNR_min << "\n";
        os << "lambda=" << obj.lambda << "\n";
        os << "min_signal=" << obj.min_signal << "\n";
        os << "rel_error=" << obj.rel_error << "\n";
        os << "sigma_th=" << obj.sigma_th << "\n";
        os << "sigma_th_max=" << obj.sigma_th_max << "\n";
        os << "auto_parameters=" << (obj.auto_parameters ? "true" : "false") << "\n";
        os << "normalized_input=" << (obj.normalized_input ? "true" : "false") << "\n";
        os << "neighborhood_optimization=" << (obj.neighborhood_optimization ? "true" : "false") << "\n";
        os << "output_steps=" << (obj.output_steps ? "true" : "false") << "\n";

	return os;
    }
};


#endif //PARTPLAY_APR_PARAMETERS_HPP
