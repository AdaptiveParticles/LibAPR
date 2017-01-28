//
// Created by cheesema on 27/01/17.
//

#include "benchmark_increase_info.hpp"



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


    benchmark_settings bs;


    set_up_benchmark_defaults(syn_image,bs);

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////

    std::cout << "BENCHMARK INCREASE INFO CONTENT" << std::endl;

    AnalysisData analysis_data(options.description,"Benchmark fixed number of spheres with increasing sized imaging domain");

    analysis_data.debug = false;

    // In this case we are increasing the number of objects
    ///////////////////////////////////////////////////////
    //
    //  Increase information
    //
    //////////////////////////////////////////////////////

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
    float min_int = sqrt(bs.shift);


    //two linear sections

    //min mean
    float min_mean = .25;
    float max_mean = 5;
    float num_steps = 1;

    float del = (max_mean - min_mean)/num_steps;

    for(float i = min_mean;i <= max_mean; i = i + del ){
        mean_int.push_back(i);
    }

    min_mean = 15;
    max_mean = 50;
    num_steps = 1;

    del = (max_mean - min_mean)/num_steps;

    for(float i = min_mean;i <= max_mean; i = i + del ){
        mean_int.push_back(i);
    }

    std::vector<float> sig_vec;


    //min mean
    float min_sig = 1;
    float max_sig = 5;
    num_steps = 1;

    del = (max_sig - min_sig)/num_steps;


    //for(float i = min_sig;i <= max_sig; i = i + del ){
    //    sig_vec.push_back(i);
    //}

    float sig_single = 2;

    sig_vec.push_back(sig_single);


    bs.N_repeats = 1; // so you have this many realisations at the parameter set
    int N_par1 = (int)number_obj.size(); // this many different parameter values to be run
    int N_par2 = (int)mean_int.size();
    int N_par3 = (int)sig_vec.size();


    for(int m = 0; m < N_par3; m++){


        ///////////////////////////////////////////////////////////////////
        //  PSF properties
        //////////////////////////////////////////////////////////////////
        bs.sig = sig_vec[m];
        set_gaussian_psf(syn_image,bs);

        /////////////////////////////////////////////
        // GENERATE THE OBJECT TEMPLATE
        //////////////////////////////////////////////

        obj_properties obj_prop(bs.obj_size,bs.sig,syn_image.sampling_properties.sampling_delta[0]);

        Object_template  basic_object = get_object_template(options,obj_prop);

        syn_image.object_templates.push_back(basic_object);

        //add the basic sphere as the standard template

        Part_timer b_time;
        //b_time.verbose_flag = true;


        for (int p = 0; p < N_par2;p++){

            for (int j = 0;j < N_par1;j++) {

                for (int i = 0; i < bs.N_repeats; i++) {


                    ///////////////////////////////
                    //
                    //  Individual synthetic image parameters
                    //
                    ///////////////////////////////




                    SynImage syn_image_loc = syn_image;

                    std::cout << "Par1: " << j << " of " << N_par1 << " Par2: " << p << " of " << N_par2 << " Rep: " << i << " of " << bs.N_repeats << "iteration % done: " <<  round((j*p*i*m)/(N_par1*N_par2*N_par3*bs.N_repeats)*100) <<  std::endl;

                    Mesh_data<uint16_t> input_image;


                    /////////////////////////////////////////
                    //////////////////////////////////////////
                    // SET UP THE DOMAIN SIZE


                    bs.num_objects = number_obj[j];

                    analysis_data.add_float_data("num_objects",bs.num_objects);

                    bs.desired_I = mean_int[p]*sqrt(bs.shift);

                    analysis_data.add_float_data("desired_I",bs.desired_I);

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

                }
            }
        }
    }

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;

}