//
// Created by bevanc on 25.01.17.
//

////////////////////////
//
//  Bevan Cheeseman 2016
//
//  APR Paper Generating Figure Increasing Information Metric
//
////////////////////////

#include <arrayfire.h>
#include "MeshDataAF.h"
#include "SynImageClasses.hpp"
#include "GenerateTemplates.hpp"
#include "SynImagePar.hpp"

#include "../src/io/writeimage.h"
#include "../src/data_structures/structure_parts.h"
#include "../src/data_structures/meshclass.h"
#include "analysis/AnalysisData.hpp"
#include "analysis/apr_analysis.h"

int main(int argc, const char * argv[]) {

    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////


    SynImage syn_image;

    std::string image_name = "paper_rel_error_test_noise";

    /////////////////////////////////////////
    //////////////////////////////////////////
    // SET UP THE DOMAIN SIZE

    int x_num = 128;
    int y_num = 128;
    int z_num = 128;

    ///////////////////////////////////////////////////////////////////
    //
    //  sampling properties


    //voxel size
    float voxel_size = .1;
    float sampling_delta = .1;

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


    ///////////////////////////////////////////////////
    //Noise properties

    syn_image.noise_properties.gauss_var = 50;
    syn_image.noise_properties.noise_type = "poisson";
    syn_image.noise_properties.noise_type = "none";

    ////////////////////////////////////////////////////
    // Global Transforms

    float shift = 1000;
    syn_image.global_trans.const_shift = shift;
    float background = shift;

    float max_dim = std::max(dom_size_y,std::max(dom_size_y,dom_size_z));

    float min_grad = .5*shift/max_dim; //stop it going negative
    float max_grad = 1.5*shift/max_dim;

    Genrand_uni gen_rand;

    syn_image.global_trans.grad_y = 0*gen_rand.rand_num(-min_grad,max_grad);
    syn_image.global_trans.grad_x = 0*gen_rand.rand_num(-min_grad,max_grad);
    syn_image.global_trans.grad_z = 0*gen_rand.rand_num(-min_grad,max_grad);


    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////

    AnalysisData analysis_data("change_rel_error_noise_free","Test");

    std::string analysis_type = "quality_metrics";

    analysis_data.create_float_dataset("num_objects",0);

    //////////////////////////////////////////////////////////
    //
    //
    //  Change rel_error and sigma
    //
    //
    /////////////////////////////////////////////////////////

    float rel_error = 0.1;

    std::vector<float> sig_vec;
    std::vector<float> rel_error_vec;


    //two linear sections

    //min mean
    float min_rel_error = .005;
    float max_rel_error = .1;
    float num_steps = 2.0;

    float del = (max_rel_error - min_rel_error)/num_steps;

    for(float i = min_rel_error;i <= max_rel_error; i = i + del ){
        rel_error_vec.push_back(i);
    }

    min_rel_error = .2;
    max_rel_error = 1;
    num_steps = 2;

    del = (max_rel_error - min_rel_error)/num_steps;

    for(float i = min_rel_error;i <= max_rel_error; i = i + del ){
        rel_error_vec.push_back(i);
    }

    //min mean
    float min_sig = 2;
    float max_sig = 4;
    num_steps = 2;

    del = (max_sig - min_sig)/num_steps;


    for(float i = min_sig;i <= max_sig; i = i + del ){
        sig_vec.push_back(i);
    }

    float desired_I = sqrt(background)*10;

    int N_repeats = 1; // so you have this many realisations at the parameter set
    int N_par1 = (int)rel_error_vec.size(); // this many different parameter values to be run
    int N_par2 = (int)sig_vec.size();

    float sig = 1;

    int num_objects = 10;


    for (int p = 0; p < N_par2;p++){

        sig = sig_vec[p];

        //generate a sphere to use
        Object_template basic_sphere;
        int sample_rate = 200;
        float obj_size = 4;

        float real_size = obj_size + 8*sig*syn_image.sampling_properties.sampling_delta[0];
        float rad_ratio = (obj_size/2)/real_size;

        float density = 1000000;

        generate_sphere_template(basic_sphere,sample_rate,real_size,density,rad_ratio);

        for (int j = 0;j < N_par1;j++){


            for(int i = 0; i < N_repeats; i++){

                //af::sync();
                af::deviceGC();

                ///////////////////////////////
                //
                //  Individual synthetic image parameters
                //
                ///////////////////////////////

                analysis_data.get_data_ref<float>("num_objects")->data.push_back(num_objects);
                analysis_data.part_data_list["num_objects"].print_flag = true;

                SynImage syn_image_loc = syn_image;

                //add the basic sphere as the standard template
                syn_image_loc.object_templates.push_back(basic_sphere);

                ///////////////////////////////////////////////////////////////////
                //PSF properties
                syn_image_loc.PSF_properties.real_sigmas[0] = sig*syn_image_loc.sampling_properties.sampling_delta[0];
                syn_image_loc.PSF_properties.real_sigmas[1] = sig*syn_image_loc.sampling_properties.sampling_delta[1];
                syn_image_loc.PSF_properties.real_sigmas[2] = sig*syn_image_loc.sampling_properties.sampling_delta[2];

                syn_image_loc.PSF_properties.I0 = 1/(pow(2*3.14159265359,1.5)*syn_image_loc.PSF_properties.real_sigmas[0]*syn_image_loc.PSF_properties.real_sigmas[1]*syn_image_loc.PSF_properties.real_sigmas[2]);

                syn_image_loc.PSF_properties.cut_th = 0.01;

                syn_image_loc.PSF_properties.set_guassian_window_size();

                syn_image_loc.PSF_properties.type = "gauss";


                std::cout << "Par1: " << j << " of " << N_par1 << " Par2: " << p << " of " << N_par2 << " Rep: " << i << " of " << N_repeats << std::endl;

                MeshDataAF<uint16_t> gen_image;

                //remove previous objects
                syn_image_loc.real_objects.resize(0);

                Real_object temp_obj;

                //set the template id
                temp_obj.template_id = 0;

                num_objects = 5;


                for (int q = 0; q < num_objects; q++) {

                    temp_obj.location[0] = gen_rand.rand_num(0,dom_size_y - real_size);
                    temp_obj.location[1] = gen_rand.rand_num(0,dom_size_x - real_size);
                    temp_obj.location[2] = gen_rand.rand_num(0,dom_size_z - real_size);

                    float obj_int = gen_rand.rand_num(1,10)*desired_I;

                    temp_obj.int_scale = (((syn_image_loc.object_templates[temp_obj.template_id].real_deltas[0]*syn_image_loc.object_templates[temp_obj.template_id].real_deltas[1]*syn_image_loc.object_templates[temp_obj.template_id].real_deltas[2])*obj_int)/(syn_image_loc.object_templates[temp_obj.template_id].max_sample*pow(voxel_size,3)));
                    // gen_rand.rand_num(.1,1)
                    syn_image_loc.real_objects.push_back(temp_obj);
                }

                //af::sync();
                af::deviceGC();

                ///////////////////////////////
                //
                //  Generate the image
                //
                ////////////////////////////////

                syn_image_loc.generate_syn_image(gen_image);

                Mesh_data<uint16_t> input_img;

                copy_mesh_data_structures(gen_image,input_img);

                //af::sync();
                af::deviceGC();

                ///////////////////////////////
                //
                //  Get the APR
                //
                //////////////////////////////

                rel_error = rel_error_vec[j];

                Part_rep p_rep(input_img.y_num,input_img.x_num,input_img.z_num);

                p_rep.timer.verbose_flag = false;

                gen_parameter_pars(syn_image_loc,p_rep.pars,image_name);

                p_rep.pars.var_th = desired_I;
                p_rep.pars.rel_error = rel_error;
                p_rep.len_scale = p_rep.pars.dx*pow(2.0,p_rep.pl_map.k_max+1);
                p_rep.pars.noise_sigma = sqrt(background);
                p_rep.pars.interp_type = 2;

                get_test_paths(p_rep.pars.image_path,p_rep.pars.utest_path,p_rep.pars.output_path);

                //af::sync();
                af::deviceGC();

                // Get the APR

                PartCellStructure<float,uint64_t> pc_struct;

                bench_get_apr(input_img,p_rep,pc_struct);

                ///////////////////////////////
                //
                //  Calculate analysis of the result
                //
                ///////////////////////////////

                produce_apr_analysis(input_img,analysis_data,pc_struct,syn_image_loc,p_rep.pars);


                af::sync();
                af::deviceGC();

            }
        }

    }


    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;
}

