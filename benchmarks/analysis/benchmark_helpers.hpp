//
// Created by cheesema on 27/01/17.
//
//  These are just some base structs and functions for making writing of benchmarking easier :D
//
//  Bevan Cheeseman 2017
//
//


#ifndef PARTPLAY_BENCHMARK_HELPERS_HPP
#define PARTPLAY_BENCHMARK_HELPERS_HPP

#include "AnalysisData.hpp"
#include "MeshDataAF.h"
#include "SynImageClasses.hpp"
#include "GenerateTemplates.hpp"
#include "SynImagePar.hpp"


struct benchmark_settings{
    //benchmark settings and defaults
    int x_num = 128;
    int y_num = 128;
    int z_num = 128;

    float voxel_size = 0.1;
    float sampling_delta = 0.1;

    float dom_size_y = 0;
    float dom_size_x = 0;
    float dom_size_z = 0;

    std::string noise_type = "poisson";

    float shift = 1000;

    float linear_shift = 0;

    float desired_I = 0;
    float N_repeats = 1;
    float num_objects = 1;
    float sig = 1;

    float int_scale_min = 1;
    float int_scale_max = 10;

    float rel_error = 0.085;

};

void set_gaussian_psf(SynImage& syn_image_loc,benchmark_settings& bs);

void set_up_benchmark_defaults(SynImage& syn_image,benchmark_settings& bs){
    /////////////////////////////////////////
    //////////////////////////////////////////
    // SET UP THE DOMAIN SIZE

    int x_num = bs.x_num;
    int y_num = bs.y_num;
    int z_num = bs.z_num;

    ///////////////////////////////////////////////////////////////////
    //
    //  sampling properties

    //voxel size
    float voxel_size = bs.voxel_size;
    float sampling_delta = bs.sampling_delta;

    syn_image.sampling_properties.voxel_real_dims[0] = voxel_size;
    syn_image.sampling_properties.voxel_real_dims[1] = voxel_size;
    syn_image.sampling_properties.voxel_real_dims[2] = voxel_size;

    //sampling rate/delta
    syn_image.sampling_properties.sampling_delta[0] = sampling_delta;
    syn_image.sampling_properties.sampling_delta[1] = sampling_delta;
    syn_image.sampling_properties.sampling_delta[2] = sampling_delta;

    //real size of domain
    float dom_size_y = y_num*sampling_delta;
    float dom_size_x = x_num*sampling_delta;
    float dom_size_z = z_num*sampling_delta;
    syn_image.real_domain.set_domain_size(0, dom_size_y, 0, dom_size_x, 0, dom_size_z);

    bs.dom_size_y = dom_size_y;
    bs.dom_size_x = dom_size_x;
    bs.dom_size_z = dom_size_z;

    ///////////////////////////////////////////////////
    //Noise properties

    syn_image.noise_properties.gauss_var = 50;
    syn_image.noise_properties.noise_type = bs.noise_type;
    //syn_image.noise_properties.noise_type = "none";

    ////////////////////////////////////////////////////
    // Global Transforms

    float shift = bs.shift;
    syn_image.global_trans.const_shift = shift;
    float background = shift;

    float max_dim = std::max(dom_size_y,std::max(dom_size_y,dom_size_z));

    float min_grad = .5*shift/max_dim; //stop it going negative
    float max_grad = 1.5*shift/max_dim;

    Genrand_uni gen_rand;

    syn_image.global_trans.grad_y = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);
    syn_image.global_trans.grad_x = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);
    syn_image.global_trans.grad_z = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);


    bs.desired_I = sqrt(background)*10;

    set_gaussian_psf(syn_image,bs);

}
void update_domain(SynImage& syn_image,benchmark_settings& bs){

    //real size of domain
    bs.dom_size_y = bs.y_num*bs.sampling_delta;
    bs.dom_size_x = bs.x_num*bs.sampling_delta;
    bs.dom_size_z = bs.z_num*bs.sampling_delta;
    syn_image.real_domain.set_domain_size(0, bs.dom_size_y, 0, bs.dom_size_x, 0, bs.dom_size_z);

}


void set_gaussian_psf(SynImage& syn_image_loc,benchmark_settings& bs){
    ///////////////////////////////////////////////////////////////////
    //PSF properties
    syn_image_loc.PSF_properties.real_sigmas[0] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[0];
    syn_image_loc.PSF_properties.real_sigmas[1] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[1];
    syn_image_loc.PSF_properties.real_sigmas[2] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[2];

    syn_image_loc.PSF_properties.I0 = 1/(pow(2*3.14159265359,1.5)*syn_image_loc.PSF_properties.real_sigmas[0]*syn_image_loc.PSF_properties.real_sigmas[1]*syn_image_loc.PSF_properties.real_sigmas[2]);

    syn_image_loc.PSF_properties.cut_th = 0.01;

    syn_image_loc.PSF_properties.set_guassian_window_size();

    syn_image_loc.PSF_properties.type = "gauss";

}

void generate_objects(SynImage& syn_image_loc,benchmark_settings& bs){

    Genrand_uni gen_rand;

    //remove previous objects
    syn_image_loc.real_objects.resize(0);

    // loop over the objects
    for(int id = 0;id < syn_image_loc.object_templates.size();id++) {

        //loop over the different template objects

        Real_object temp_obj;

        //set the template id
        temp_obj.template_id = 0;

        for (int q = 0; q < bs.num_objects; q++) {

            // place them randomly in the image

            temp_obj.template_id = id;

            Object_template curr_obj = syn_image_loc.object_templates[temp_obj.template_id];

            temp_obj.location[0] = gen_rand.rand_num(0, bs.dom_size_y - curr_obj.real_size[0]);
            temp_obj.location[1] = gen_rand.rand_num(0, bs.dom_size_x - curr_obj.real_size[0]);
            temp_obj.location[2] = gen_rand.rand_num(0, bs.dom_size_z - curr_obj.real_size[0]);

            float obj_int = gen_rand.rand_num(bs.int_scale_min, bs.int_scale_max) * bs.desired_I;

            temp_obj.int_scale = (
                    ((curr_obj.real_deltas[0] * curr_obj.real_deltas[1] * curr_obj.real_deltas[2]) * obj_int) /
                    (curr_obj.max_sample * pow(bs.voxel_size, 3)));


            syn_image_loc.real_objects.push_back(temp_obj);
        }
    }


}

void set_up_part_rep(SynImage& syn_image_loc,Part_rep& p_rep,benchmark_settings& bs){

    std::string image_name = "benchmark_image";

    p_rep.initialize(bs.y_num,bs.x_num,bs.z_num);

    gen_parameter_pars(syn_image_loc,p_rep.pars,image_name);

    p_rep.pars.var_th = bs.desired_I;
    p_rep.pars.rel_error = bs.rel_error;
    p_rep.len_scale = p_rep.pars.dx*pow(2.0,p_rep.pl_map.k_max+1);
    p_rep.pars.noise_sigma = sqrt(bs.shift);
    p_rep.pars.interp_type = 0;

    get_test_paths(p_rep.pars.image_path,p_rep.pars.utest_path,p_rep.pars.output_path);
}

#endif //PARTPLAY_BENCHMARK_HELPERS_HPP
