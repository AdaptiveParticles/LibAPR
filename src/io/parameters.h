/////////////////////////////////////////////////////
//
//  Header File for setting local file locations and parameter settings
//
//  Bevan Cheeseman 2015
//
//
///////////////////////////////////////////////////////

#ifndef PARTPLAY_PARAMETERS_H
#define PARTPLAY_PARAMETERS_H

#include <string>
#include <vector>

class Proc_par{
    //
    //  Bevan Cheeseman
    //
    //  Preproccessing parameter class
    //
    //

public:

    unsigned int window_mean;
    unsigned int window_smooth;
    unsigned int window_var;
    unsigned int num_p_grad;
    float grad_h;
    float E0;
    float I_th;
    float var_th;
    float var_th_max;
    float h_max;
    float aniso;
    float z_factor;
    unsigned int slices_per_read;
    float min_var;
    int bit_rate;
    std::string name;
    float comp_scale;
    int type_float;
    float max_I;
    float min_I;

    int k_method;
    int var_method;
    int grad_method;

    int noise_model;

    //new parameters

    float dy;
    float dx;
    float dz;

    float psfy;
    float psfx;
    float psfz;

    float ydim;
    float xdim;
    float zdim;

    float noise_sigma;
    float background;

    //length scale (L0, effective real size of full domain)
    float len_scale;

    std::string image_path;
    std::string utest_path;
    std::string output_path;
    std::string data_path;


    //
    int interp_type;

    //pipeline parameters

    //bspline smoothing
    float lambda;
    float tol;
    float var_scale;
    float mean_scale;
    float noise_scale;

    //padding
    int padd_flag;

    //Approximation parameter
    float rel_error;

    unsigned int part_config;

    unsigned int pull_scheme;

    // padd vector
    std::vector<int> padd_dims;

    Proc_par()
    :aniso(1),pull_scheme(2),interp_type(2),window_mean(3),window_smooth(1),window_var(10),num_p_grad(1),grad_h(1),E0(1),I_th(0),var_th(10),h_max(0.00999),z_factor(1),var_th_max(0),slices_per_read(1),min_var(1),bit_rate(16),comp_scale(20),type_float(0),max_I(0),min_I(0),lambda(1),tol(0.01),var_scale(2),mean_scale(1),grad_method(1),var_method(2),k_method(3),padd_flag(1),rel_error(0.1),part_config(1),noise_model(1),noise_scale(2),len_scale(0)
    {};




};

std::string get_path(std::string PATH_ENV);
void get_test_paths(std::string& image_path,std::string& utest_path,std::string& output_path);
bool get_image_stats(Proc_par& pars,std::string output_path,std::string image_name);
void get_image_parameters(Proc_par& pars,std::string output_path,std::string image_name);
void setup_output_folder(Proc_par& pars);
void get_file_names(std::vector<std::string>& file_names,std::string file_list_path);

#endif



