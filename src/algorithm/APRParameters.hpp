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

    // main pipeline parameters
    float Ip_th = 0;
    float lambda = 3.0;
    float rel_error = 0.1;
    float sigma_th = 0;
    float sigma_th_max = 0;
    float grad_th = 0;
    bool auto_parameters = false;

    // additional pipeline parameters
    bool reflect_bc_lis = true;
    bool check_input = false;
    bool swap_dimensions = false;
    bool neighborhood_optimization = true;
    bool constant_intensity_scale = false;
    bool output_steps = false;

    std::string name;
    std::string output_dir;
    std::string input_image_name;
    std::string input_dir;
    std::string mask_file;

    // compression
    float noise_sd_estimate = 0;
    float background_intensity_estimate = 0;

    friend std::ostream & operator<<(std::ostream &os, const APRParameters &obj) {
        os << "Ip_th=" << obj.Ip_th << "\n";
        os << "lambda=" << obj.lambda << "\n";
        os << "rel_error=" << obj.rel_error << "\n";
        os << "sigma_th=" << obj.sigma_th << "\n";
        os << "sigma_th_max=" << obj.sigma_th_max << "\n";
        os << "auto_parameters=" << (obj.auto_parameters ? "true" : "false") << "\n";
        os << "neighborhood_optimization=" << (obj.neighborhood_optimization ? "true" : "false") << "\n";
        os << "output_steps=" << (obj.output_steps ? "true" : "false") << "\n";

	    return os;
    }

    void validate_parameters(){
        if (sigma_th ==  0){
            std::cerr << "Warning: sigma_th is set to 0, this may result in unexpected results due to divide by zero errors. Consider setting this to a non-zero small value, if it is not needed."  << std::endl;
        }
    }
};


#endif //PARTPLAY_APR_PARAMETERS_HPP
