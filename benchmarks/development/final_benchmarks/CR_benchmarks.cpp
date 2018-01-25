//
// Created by cheesema on 25.01.18.
//

#include <functional>
#include <string>

#include "src/data_structures/APR/APR.hpp"

#include "benchmarks/development/final_benchmarks/APRBenchmark.hpp"

#include <arrayfire.h>
#include "MeshDataAF.h"
#include "SynImageClasses.hpp"
#include "GenerateTemplates.hpp"
#include "SynImagePar.hpp"
//#include "benchmarks/analysis/syn_templates.h"

#include "benchmarks/development/final_benchmarks/BenchHelper.hpp"



bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    float CR = 0;
    uint64_t number_reps = 1;
    bool stats_file = false;
};



////////////////////////
//
//  Bevan Cheeseman 2017
//
//  Generating Increasing Domain Size
//
////////////////////////


cmdLineOptions read_command_line_options_this(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"CR_benchmarks -CR val\"" << std::endl;
        exit(1);
    }


    if(command_option_exists(argv, argv + argc, "-CR"))
    {
        result.CR = std::stof(std::string(get_command_option(argv, argv + argc, "-CR")));
    }

    if(command_option_exists(argv, argv + argc, "-number_reps"))
    {
        result.number_reps = std::stof(std::string(get_command_option(argv, argv + argc, "-number_reps")));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    return result;

}



int main(int argc, char **argv) {


    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    af::info();

    cmdLineOptions options = read_command_line_options_this(argc,argv);

    SynImage syn_image;

    std::string image_name = "sphere";

    ///////////////////////////////////////////////////////////////////
    //  PSF properties
    //////////////////////////////////////////////////////////////////

    BenchHelper benchHelper;

    BenchHelper::benchmark_settings bs;

    bs.sig = 3;

    benchHelper.set_up_benchmark_defaults(syn_image,bs);

    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    bs.obj_size = 3;



    BenchHelper::obj_properties obj_prop(bs);

    Object_template  basic_object;

    generate_sphere_template(basic_object, obj_prop.sample_rate, obj_prop.real_size, obj_prop.density,
                             obj_prop.rad_ratio);

    syn_image.object_templates.push_back(basic_object);

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////



    std::cout << "BENCHMARK  KEEPING COMP RATIO FIXED" << std::endl;

    std::vector<int> image_size;

    image_size = {200,400,600,800};

    float ratio = options.CR;
    bs.N_repeats = options.number_reps;

    int N_par = (int)image_size.size(); // this many different parameter values to be run

    APRTimer b_timer;
    b_timer.verbose_flag = true;

    APRBenchmark apr_benchmarks;

    apr_benchmarks.analysis_data.add_float_data("FixedCR",ratio);
    apr_benchmarks.analysis_data.add_float_data("obj_size",bs.obj_size);
    apr_benchmarks.analysis_data.add_float_data("sig",bs.sig);

    for (int j = 0;j < N_par;j++){

        /////////////////////////////////////////
        //////////////////////////////////////////
        // SET UP THE DOMAIN SIZE

        bs.x_num = image_size[j];
        bs.y_num = image_size[j];
        bs.z_num = image_size[j];

        bs.num_objects = pow(bs.x_num,3)/(33400*ratio);

        bs.rel_error = 0.1;

        for(int i = 0; i < bs.N_repeats; i++){

            b_timer.start_timer("one_it");

            apr_benchmarks.analysis_data.add_float_data("width",bs.x_num);

            apr_benchmarks.analysis_data.add_float_data("number_objects",bs.num_objects);

            SynImage syn_image_loc = syn_image;

            benchHelper.update_domain(syn_image_loc,bs);

            //Generate objects

            benchHelper.generate_objects(syn_image_loc,bs);

            ///////////////////////////////
            //
            //  Generate the image
            //
            ////////////////////////////////

            MeshDataAF<uint16_t> gen_image;
            b_timer.start_timer("generating image");
            syn_image_loc.generate_syn_image(gen_image);
            b_timer.stop_timer();

            std::cout << "Image Generated" << std::endl;

            MeshData<uint16_t> input_img;

            benchHelper.copy_mesh_data_structures(gen_image,input_img);

            APRConverter<uint16_t> apr_converter;

            apr_converter.par.Ip_th = 1100;
            apr_converter.par.sigma_th = 500;
            apr_converter.par.sigma_th_max = 250;
            apr_converter.par.rel_error = 0.1;
            apr_converter.par.lambda = 3;

            apr_converter.par.input_dir = options.directory;
            apr_converter.par.input_image_name = "cr_" + std::to_string(j) + "_" + std::to_string(i) ;

            apr_converter.fine_grained_timer.verbose_flag = false;
            apr_converter.method_timer.verbose_flag = false;
            apr_converter.allocation_timer.verbose_flag = false;
            apr_converter.computation_timer.verbose_flag = false;
            apr_converter.total_timer.verbose_flag = true;

            apr_benchmarks.analysis_data.name = "cr_final_benchmarking";



            apr_benchmarks.benchmark_dataset_synthetic(apr_converter,input_img);

            af::sync();
            af::deviceGC();

        }
    }
    apr_benchmarks.analysis_data.file_name = options.directory + "analysis_data/CR" + std::to_string((int)ratio) + apr_benchmarks.analysis_data.file_name;
    apr_benchmarks.analysis_data.write_analysis_data_hdf5();
    //write the analysis output

    return 0;


}
