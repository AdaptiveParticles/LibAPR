//
// Created by cheesema on 29.09.17.
//

#include "benchmark_local_scale.hpp"


int main(int argc, char **argv) {

    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    std::cout << "Increase Reconstruction Error Parameter Benchmark" << std::endl;

    cmdLineOptionsBench options = read_command_line_options(argc, argv);

    SynImage syn_image;

    std::string image_name = options.template_name;

    benchmark_settings bs;

    set_up_benchmark_defaults(syn_image, bs);

    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //////////////////////////////////////////////////////////////////

    AnalysisData analysis_data(options.description, "Test", argc, argv);

    analysis_data.create_float_dataset("num_objects", 0);
    analysis_data.create_float_dataset("rel_error", 0);

    process_input(options, syn_image, analysis_data, bs);

    //////////////////////////////////////////////////////////
    //
    //
    //  Change rel_error and sigma
    //
    //
    /////////////////////////////////////////////////////////

    std::vector<float> sig_vec;
    std::vector<float> rel_error_vec;

    //two linear sections

    //min mean
    float min_rel_error = .1;
    float max_rel_error = .1;
    float num_steps = options.delta;

//    float del = (max_rel_error - min_rel_error) / num_steps;
//
//    for (float i = min_rel_error; i <= max_rel_error; i = i + del) {
//        rel_error_vec.push_back(i);
//    }

    float del = 0;

    rel_error_vec = {50,200,1000};

    //min mean
    float min_sig = 1;
    float max_sig = 10;
    num_steps = 3;

    del = (max_sig - min_sig) / num_steps;


    for (float i = min_sig; i <= max_sig; i = i + del) {
        // sig_vec.push_back(i);
    }

    sig_vec = {1,2,3,4,5,6};

    //min mean

    std::vector<int> window_1;
    std::vector<int> window_2;

    window_1 = {1,2,3};
    window_2 = {1,2,3,4,5,6,7,8};

    int N_par1 = (int)rel_error_vec.size(); // this many different parameter values to be run
    int N_par2 = (int)sig_vec.size();
    int N_par3 = (int)window_1.size();
    int N_par4 = (int)window_2.size();

    bs.num_objects = 10;

    bs.obj_size = 2;

    analysis_data.add_float_data("obj_size",bs.obj_size);
    analysis_data.add_float_data("num_objects",bs.num_objects);

    Genrand_uni gen_rand;

    bs.int_scale_min = 1;
    bs.int_scale_max = 10;

    Part_timer b_timer;
    b_timer.verbose_flag = true;

    bs.shift = 1000;
    syn_image.global_trans.const_shift = 1000;


    for(int u = 0;u < N_par4;u++) {

        for (int q = 0; q < N_par3; q++) {

            for (int p = 0; p < N_par2; p++) {

                bs.sig = sig_vec[p];

                obj_properties obj_prop(bs);

                Object_template basic_object = get_object_template(options, obj_prop);

                SynImage syn_image_n = syn_image;

                syn_image_n.object_templates.push_back(basic_object);

                for (int j = 0; j < N_par1; j++) {

                    for (int i = 0; i < bs.N_repeats; i++) {

                        b_timer.start_timer("one it");

                        //af::sync();
                        af::deviceGC();

                        ///////////////////////////////
                        //
                        //  Individual synthetic image parameters
                        //
                        ///////////////////////////////

                        analysis_data.get_data_ref<float>("num_objects")->data.push_back(bs.num_objects);
                        analysis_data.part_data_list["num_objects"].print_flag = true;

                        SynImage syn_image_loc;

                        set_up_benchmark_defaults(syn_image_loc, bs);

                        bs.sig = sig_vec[p];

                        update_domain(syn_image_loc, bs);
                        syn_image_loc.object_templates.push_back(basic_object);

                        //add the basic sphere as the standard template

                        analysis_data.add_float_data("psf",sig_vec[p]);
                        analysis_data.add_float_data("rep",i);

                        ///////////////////////////////////////////////////////////////////
                        //PSF properties


                        bs.desired_I = rel_error_vec[j];

                        analysis_data.add_float_data("desired_I", bs.desired_I);

                        set_gaussian_psf(syn_image_loc, bs);

                        std::cout << "Par1: " << j << " of " << N_par1 << " Par2: " << p << " of " << N_par2 << " Par: "
                                  << q << " of " << N_par3 << std::endl;

                        std::cout << "Outer loop%: " << 100*u/N_par4 << std::endl;

                        generate_objects(syn_image_loc, bs);

                        //b_timer.stop_timer();

                        //b_timer.start_timer("generate_syn_image");

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

                        bs.rel_error = 0.1;

                        analysis_data.add_float_data("rel_error", bs.rel_error);

                        Part_rep p_rep;

                        set_up_part_rep(syn_image_loc, p_rep, bs);

                        p_rep.pars.padd_dims = {window_1[q],window_1[q],window_1[q],window_2[u],window_2[u],window_2[u]};

                        analysis_data.add_float_data("var_window_1",window_1[q]);
                        analysis_data.add_float_data("var_window_2",window_2[u]);

                        // Get the APRgit



                        //p_rep.pars.var_scale = 1.0;
                        //p_rep.pars.var_th = 0;

                       // compute_var_ratio_perfect(syn_image_loc,p_rep,input_img,analysis_data);


                        PartCellStructure<float, uint64_t> pc_struct;

                        //p_rep.pars.var_th = 1;
                        bench_get_apr(input_img, p_rep, pc_struct, analysis_data);

                        //b_timer.stop_timer();

                        ///////////////////////////////
                        //
                        //  Calculate analysis of the result
                        //
                        ///////////////////////////////

                       // b_timer.start_timer("analysis");

                        produce_apr_analysis(input_img, analysis_data, pc_struct, syn_image_loc, p_rep.pars);

                     //   std::cout << "Num Parts: " << pc_struct.get_number_parts() << std::endl;

                        af::sync();
                        af::deviceGC();

                        b_timer.stop_timer();

                    }
                }

            }
        }
    }
    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;
}
