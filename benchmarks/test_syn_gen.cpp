//
// Created by bevanc on 25.01.17.
//
#include <iostream>

#include <arrayfire.h>
#include "MeshDataAF.h"
#include "SynImageClasses.hpp"
#include "GenerateTemplates.hpp"
#include "SynImagePar.hpp"
#include "benchmarks/development/old_io/writeimage.h"
#include "benchmarks/development/old_structures/structure_parts.h"
#include "src/data_structures/Mesh/MeshData.hpp"

int main() {

    af::setBackend(AF_BACKEND_OPENCL);

    af::info();

    af::array test;

    std::cout << af::getDeviceCount() << std::endl;

    MeshDataAF<uint16_t> test_m;

    SynImage test_syn_image;

    Syn_gen_par syn_par;

    std::string image_name = "test_sphere1";

    syn_par.name = image_name;
    syn_par.template_name = "sphere";

    /////////////////////////////////////////
    //////////////////////////////////////////
    // SET UP THE DOMAIN SIZE

    int x_num = 1200;
    int y_num = 1200;
    int z_num = 1000;

    ///////////////////////////////////////////////////////////////////
    //
    //  sampling properties


    //voxel size
    float voxel_size = .1;
    float sampling_delta = .1;

    test_syn_image.sampling_properties.voxel_real_dims[0] = voxel_size;
    test_syn_image.sampling_properties.voxel_real_dims[1] = voxel_size;
    test_syn_image.sampling_properties.voxel_real_dims[2] = voxel_size;

    //sampling rate/delta
    test_syn_image.sampling_properties.sampling_delta[0] = sampling_delta;
    test_syn_image.sampling_properties.sampling_delta[1] = sampling_delta;
    test_syn_image.sampling_properties.sampling_delta[2] = sampling_delta;

    //real size of domain
    float dom_size_y = y_num*sampling_delta;
    float dom_size_x = x_num*sampling_delta;
    float dom_size_z = z_num*sampling_delta;
    test_syn_image.real_domain.set_domain_size(0, dom_size_y, 0, dom_size_x, 0, dom_size_z);

    ///////////////////////////////////////////////////////////////////
    //PSF properties
    test_syn_image.PSF_properties.real_sigmas[0] = .1;
    test_syn_image.PSF_properties.real_sigmas[1] = .1;
    test_syn_image.PSF_properties.real_sigmas[2] = .1;

    test_syn_image.PSF_properties.I0 = 1/(pow(2*3.14159265359,1.5)*test_syn_image.PSF_properties.real_sigmas[0]*test_syn_image.PSF_properties.real_sigmas[1]*test_syn_image.PSF_properties.real_sigmas[2]);

    test_syn_image.PSF_properties.cut_th = 0.01;

    test_syn_image.PSF_properties.set_guassian_window_size();

    test_syn_image.PSF_properties.type = "gauss";

    ///////////////////////////////////////////////////
    //Noise properties

    test_syn_image.noise_properties.gauss_var = 50;
    test_syn_image.noise_properties.noise_type = "poisson";

    ////////////////////////////////////////////////////
    // Global Transforms

    float shift = 1000;
    float background = shift;
    test_syn_image.global_trans.const_shift = shift;

    float max_dim = std::max(dom_size_y,std::max(dom_size_y,dom_size_z));

    float min_grad = .5*shift/max_dim; //stop it going negative
    float max_grad = 1.5*shift/max_dim;

    Genrand_uni gen_rand;

    test_syn_image.global_trans.grad_y = 0*gen_rand.rand_num(-min_grad,max_grad);
    test_syn_image.global_trans.grad_x = 0*gen_rand.rand_num(-min_grad,max_grad);
    test_syn_image.global_trans.grad_z = 0*gen_rand.rand_num(-min_grad,max_grad);

    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    //generate a sphere to use
    Object_template basic_sphere;
    int sample_rate = 100;
    float real_size = 5;

    float density = 1000000;

    generate_sphere_template(basic_sphere,sample_rate,real_size,density);

    //add the basic sphere as the standard template
    test_syn_image.object_templates.push_back(basic_sphere);


    Real_object temp_obj;

    //set the template id
    temp_obj.template_id = 0;

    int num_objects = 10;

    float desired_I = sqrt(background)*30;


    for (int q = 0; q < num_objects; q++) {

        temp_obj.location[0] = gen_rand.rand_num(0,test_syn_image.real_domain.size[0] - real_size);
        temp_obj.location[1] = gen_rand.rand_num(0,test_syn_image.real_domain.size[1] - real_size);
        temp_obj.location[2] = gen_rand.rand_num(0,test_syn_image.real_domain.size[2] - real_size);

        float obj_int = gen_rand.rand_num(1,10)*desired_I;

        temp_obj.int_scale = (((test_syn_image.object_templates[temp_obj.template_id].real_deltas[0]*test_syn_image.object_templates[temp_obj.template_id].real_deltas[1]*test_syn_image.object_templates[temp_obj.template_id].real_deltas[2])*obj_int)/(test_syn_image.object_templates[temp_obj.template_id].max_sample*pow(test_syn_image.sampling_properties.voxel_real_dims[0],3)));

        test_syn_image.real_objects.push_back(temp_obj);
    }

    ////////////////////////////////////////////////////////////////////////////
    //

    //generate image


    ////////////////////////////////////////////////////////////////////////////
    //

    MeshDataAF<uint16_t> test_gen_image;

    Part_timer timer;
    timer.verbose_flag = true;

    timer.start_timer("Generating Image");

    test_syn_image.generate_syn_image(test_gen_image);

    timer.stop_timer();

    MeshData<uint16_t> output_img;
    output_img.x_num = test_gen_image.x_num;
    output_img.y_num = test_gen_image.y_num;
    output_img.z_num = test_gen_image.z_num;

    //output_img.mesh = test_gen_image.mesh;

    debug_write(output_img,"gen_image");

    return 0;
}
