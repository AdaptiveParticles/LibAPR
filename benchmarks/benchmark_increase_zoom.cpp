//
// Created by cheesema on 28/01/17.
//

#include "benchmark_increase_zoom.hpp"


int main(int argc, char **argv) {

    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    std::cout << "Increase Zoom Benchmark" << std::endl;

    cmdLineOptionsBench options = read_command_line_options(argc,argv);

    SynImage syn_image;

    std::string image_name = options.template_name;

    benchmark_settings bs;

    set_up_benchmark_defaults(syn_image,bs);

    Genrand_uni gen_rand;

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////

    AnalysisData analysis_data(options.description,"Test",argc,argv);


    analysis_data.create_float_dataset("num_objects",0);

    process_input(options,syn_image,analysis_data,bs);


    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE



    //////////////////////////////////////////////////////////
    //
    //
    //  Change rel_error and sigma
    //
    //
    /////////////////////////////////////////////////////////

    float image_size_max = options.image_size;
    float image_size_min = 50;

    std::vector<float> sampling_rate;

    float obj_size = 3;

    float real_domain_size = obj_size*6;

   // float sampling_lower_b = sqrt(log(1/syn_image.PSF_properties.cut_th)*2*pow(bs.sig,2));
    float min_sampling = real_domain_size/image_size_max;
    float max_sampling = real_domain_size/image_size_min;
    float num_points = options.delta;
    float delta = (max_sampling - min_sampling)/num_points;

    for (float i = min_sampling; i < max_sampling; i = i + delta) {
        sampling_rate.push_back(i);
    }

    //sampling_rate.push_back(max_sampling);

    //sampling_rate.push_back(max_sampling);

//    //min mean
//    float min_sig = 1;
//    float max_sig = 10;
//    float num_steps = bs.N_repeats;
//
//    float del = (max_sig - min_sig) / num_steps;
//    std::vector<float> sig_vec;
//
//    for (float i = min_sig; i <= max_sig; i = i + del) {
//         sig_vec.push_back(i);
//    }
//
//    //syn_image.noise_properties.noise_type = "none";
//
    int N_par = (int)sampling_rate.size();
    set_up_benchmark_defaults(syn_image,bs);

    float sig =3;



    for (int j = 0;j < N_par;j++){

        ///////////////////////////////////////////////////////////////////
        //PSF properties



        for(int i = 0; i < bs.N_repeats; i++){

            SynImage syn_image_loc = syn_image;


            set_up_benchmark_defaults(syn_image_loc,bs);

            bs.desired_I = sqrt(bs.shift)*30;

            bs.voxel_size = sampling_rate[j];
            bs.sampling_delta = sampling_rate[j];

            bs.x_num = round(real_domain_size/bs.sampling_delta);
            bs.y_num = round(real_domain_size/bs.sampling_delta);
            bs.z_num = round(real_domain_size/bs.sampling_delta);

            update_domain(syn_image_loc,bs);

            bs.sig = sig*sampling_rate.back()/sampling_rate[j];

            set_gaussian_psf(syn_image_loc,bs);

            std::cout << "Generating Templates" << std::endl;

            obj_properties obj_prop(bs);

            obj_prop.sample_rate = std::max(bs.x_num,200);
            obj_prop.obj_size = obj_size;

            Object_template  basic_object = get_object_template(options,obj_prop);

            syn_image_loc.object_templates.push_back(basic_object);

            //syn_image_loc.noise_properties.noise_type = "gaussian";
            bs.int_scale_min = 1;
            bs.int_scale_max = 1;

            bs.rel_error = 0.12;

            //af::sync();
            af::deviceGC();

            ///////////////////////////////
            //
            //  Individual synthetic image parameters
            //
            ///////////////////////////////

            analysis_data.add_float_data("num_objects",bs.num_objects);



            //add the basic sphere as the standard template


            std::cout << "Par: " << j << " of " << N_par << " Rep: " << i << " of " << bs.N_repeats << std::endl;


            //generate one objects in the center
            generate_object_center(syn_image_loc,bs);


            ///////////////////////////////
            //
            //  Generate the image
            //
            ////////////////////////////////

            MeshDataAF<uint16_t> gen_image;

            syn_image_loc.generate_syn_image(gen_image);

            Mesh_data<uint16_t> input_img;

            copy_mesh_data_structures(gen_image,input_img);


            ///////////////////////////////
            //
            //  Get the APR
            //
            //////////////////////////////

            bs.lambda = 20*pow(bs.x_num/50,2);

            Part_rep p_rep;

            set_up_part_rep(syn_image_loc,p_rep,bs);

            // Get the APR

            p_rep.pars.var_th = 1;


            PartCellStructure<float,uint64_t> pc_struct;

            bench_get_apr(input_img,p_rep,pc_struct,analysis_data);

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




    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;
}