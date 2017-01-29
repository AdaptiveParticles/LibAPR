//
// Created by cheesema on 28/01/17.
//

#include "benchmark_increase_rel_error.hpp"

int main(int argc, char **argv) {

    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    std::cout << "Increase Reconstruction Error Parameter Benchmark" << std::endl;

    cmdLineOptionsBench options = read_command_line_options(argc,argv);

    SynImage syn_image;

    std::string image_name = options.template_name;

    benchmark_settings bs;

    set_up_benchmark_defaults(syn_image,bs);

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

    AnalysisData analysis_data(options.description,"Test",argc,argv;

    std::string analysis_type = "quality_metrics";

    analysis_data.create_float_dataset("num_objects",0);

    analysis_data.quality_metrics_input = true;
    analysis_data.quality_metrics_gt = true;

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
    float min_rel_error = .005;
    float max_rel_error = .1;
    float num_steps = 2;

    float del = (max_rel_error - min_rel_error)/num_steps;

    for(float i = min_rel_error;i <= max_rel_error; i = i + del ){
        rel_error_vec.push_back(i);
    }

    min_rel_error = .12;
    max_rel_error = .4;
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
        //sig_vec.push_back(i);
    }

    sig_vec.push_back(2);


    int N_par1 = (int)rel_error_vec.size(); // this many different parameter values to be run
    int N_par2 = (int)sig_vec.size();


    Genrand_uni gen_rand;


    for (int p = 0; p < N_par2;p++){

        bs.sig = sig_vec[p];

        obj_properties obj_prop(bs.obj_size,bs.sig);

        Object_template  basic_object = get_object_template(options,obj_prop);

        syn_image.object_templates.push_back(basic_object);

        for (int j = 0;j < N_par1;j++){

            for(int i = 0; i < bs.N_repeats; i++){

                //af::sync();
                af::deviceGC();

                ///////////////////////////////
                //
                //  Individual synthetic image parameters
                //
                ///////////////////////////////

                analysis_data.get_data_ref<float>("num_objects")->data.push_back(bs.num_objects);
                analysis_data.part_data_list["num_objects"].print_flag = true;

                SynImage syn_image_loc = syn_image;

                //add the basic sphere as the standard template

                ///////////////////////////////////////////////////////////////////
                //PSF properties

                set_gaussian_psf(syn_image_loc,bs);

                std::cout << "Par1: " << j << " of " << N_par1 << " Par2: " << p << " of " << N_par2 << " Rep: " << i << " of " << bs.N_repeats << std::endl;

                generate_objects(syn_image_loc,bs);


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

                bs.rel_error = rel_error_vec[j];

                Part_rep p_rep;

                set_up_part_rep(syn_image_loc,p_rep,bs);

                // Get the APR

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

    }

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;
}