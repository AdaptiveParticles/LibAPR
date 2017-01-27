//
// Created by cheesema on 27/01/17.
//

#include "benchmark_incrase_info.hpp"



////////////////////////
//
//  Bevan Cheeseman 2017
//
//  Generating Increasing Domain Size
//
////////////////////////



int main(int argc, char **argv) {


    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    cmdLineOptionsBench options = read_command_line_options(argc,argv);

    SynImage syn_image;

    std::string image_name = options.template_name;


    ///////////////////////////////////////////////////////////////////
    //  PSF properties
    //////////////////////////////////////////////////////////////////

    benchmark_settings bs;


    bs.sig = 3;

    set_up_benchmark_defaults(syn_image,bs);

    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    float obj_size = 2;

    obj_properties obj_prop(obj_size,bs.sig,syn_image.sampling_properties.sampling_delta[0]);

    obj_prop.sample_rate = 200;

    Object_template  basic_object = get_object_template(options,obj_prop);

    syn_image.object_templates.push_back(basic_object);


    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////

    std::cout << "BENCHMARK INCREASE IMAGE SIZE" << std::endl;

    AnalysisData analysis_data("paper_increase_domain","Benchmark fixed number of spheres with increasing sized imaging domain");

    analysis_data.create_float_dataset("num_objects",0);

    analysis_data.create_float_dataset("mean_int",0);

    // In this case we are increasing the number of objects
    ///////////////////////////////////////////////////////
    //
    //  Increase information
    //
    ///////////////////////////////////////////////////


    int num_obj =30;
    std::vector<int> number_obj;
    int step = 2;

    //number_obj.push_back(num_obj);

    for(int i = 1; i <= num_obj ; i = i + step ){
        number_obj.push_back(i);
    }



    //////////////////////////////////////////////////////////
    //
    //
    //  Increase Singal to Noise Ratio
    //
    //
    /////////////////////////////////////////////////////////

    float rel_error = 0.1;

    std::vector<float> mean_int;

    //minimum intensity
    float min_int = sqrt(background);


    //two linear sections

    //min mean
    float min_mean = .25;
    float max_mean = 5;
    float num_steps = 20;

    float del = (max_mean - min_mean)/num_steps;

    for(float i = min_mean;i <= max_mean; i = i + del ){
        mean_int.push_back(i);
    }

    min_mean = 15;
    max_mean = 50;
    num_steps = 10;

    del = (max_mean - min_mean)/num_steps;

    for(float i = min_mean;i <= max_mean; i = i + del ){
        mean_int.push_back(i);
    }

    std::vector<float> sig_vec;


    //min mean
    float min_sig = 1;
    float max_sig = 5;
    num_steps = 5;

    del = (max_sig - min_sig)/num_steps;


    //for(float i = min_sig;i <= max_sig; i = i + del ){
    //    sig_vec.push_back(i);
    //}

    float sig_single = 5;

    sig_vec.push_back(sig_single);

    float desired_I;


    bs.N_repeats = 30; // so you have this many realisations at the parameter set
    int N_par1 = (int)number_obj.size(); // this many different parameter values to be run
    int N_par2 = (int)mean_int.size();
    int N_par3 = (int)sig_vec.size();

    float sig = 1;

    for(int m = 0; m < N_par3; m++){

        sig = sig_vec[m];

        //generate a sphere to use
        Object_template basic_sphere;
        int sample_rate = 200;
        float obj_size = 2;

        float real_size = obj_size + 8*sig*syn_image.sampling_properties.sampling_delta[0];
        float rad_ratio = (obj_size/2)/real_size;

        float density = 1000000;

        generate_sphere_template(basic_sphere,sample_rate,real_size,density,rad_ratio);

        //add the basic sphere as the standard template

        Part_timer b_time;
        //b_time.verbose_flag = true;


        for (int p = 0; p < N_par2;p++){

            for (int j = 0;j < N_par1;j++) {

                for (int i = 0; i < N_repeats; i++) {


                    b_timer.start_timer("one_it");

                    ///////////////////////////////
                    //
                    //  Individual synthetic image parameters
                    //
                    ///////////////////////////////

                    analysis_data.get_data_ref<float>("num_objects")->data.push_back(bs.num_objects);
                    analysis_data.part_data_list["num_objects"].print_flag = true;

                    SynImage syn_image_loc = syn_image;

                    std::cout << "Par: " << j << " of " << N_par << " Rep: " << i << " of " << bs.N_repeats
                              << std::endl;

                    Mesh_data<uint16_t> input_image;


                    /////////////////////////////////////////
                    //////////////////////////////////////////
                    // SET UP THE DOMAIN SIZE

                    bs.x_num = image_size[j];
                    bs.y_num = image_size[j];
                    bs.z_num = image_size[j];

                    update_domain(syn_image_loc, bs);

                    //Generate objects

                    generate_objects(syn_image_loc, bs);


                    ///////////////////////////////
                    //
                    //  Generate the image
                    //
                    ////////////////////////////////

                    MeshDataAF<uint16_t> gen_image;

                    syn_image_loc.generate_syn_image(gen_image);

                    Mesh_data<uint16_t> input_img;

                    copy_mesh_data_structures(gen_image, input_img);


                    ///////////////////////////////
                    //
                    //  Get the APR
                    //
                    //////////////////////////////

                    Part_rep p_rep;

                    set_up_part_rep(syn_image_loc, p_rep, bs);

                    // Get the APR

                    PartCellStructure<float, uint64_t> pc_struct;

                    bench_get_apr(input_img, p_rep, pc_struct, analysis_data);

                    ///////////////////////////////
                    //
                    //  Calculate analysis of the result
                    //
                    ///////////////////////////////

                    produce_apr_analysis(input_img, analysis_data, pc_struct, syn_image_loc, p_rep.pars);


                    af::sync();
                    af::deviceGC();

                    b_timer.stop_timer();
                }
            }
        }
    }

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;

}