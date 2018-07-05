//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///

const char* usage = R"(
Form the APR form image: Takes an uint16_t input tiff image and forms the APR and saves it as hdf5. The hdf5 output of this program
can be used with the other apr examples, and also viewed with HDFView.

Usage:

(minimal with auto-parameters)

Example_get_apr -i input_image_tiff -d input_directory [-o name_of_output]

Additional settings (High Level):

-I_th intensity_threshold  (will ignore areas of image below this threshold, useful for removing camera artifacts or auto-flouresence)
-SNR_min minimal_snr (minimal ratio of the signal to the standard deviation of the background, set by default to 6)

Advanced (Direct) Settings:

-lambda lambda_value (directly set the value of the gradient smoothing parameter lambda (reasonable range 0.1-10, default: 3)
-min_signal min_signal_val (directly sets a minimum absolute signal size relative to the local background, also useful for removing background, otherwise set using estimated background noise estimate and minimal SNR of 6)
-mask_file mask_file_tiff (takes an input image uint16_t, assumes all zero regions should be ignored by the APR, useful for pre-processing of isolating desired content, or using another channel as a mask)
-rel_error rel_error_value (Reasonable ranges are from .08-.15), Default: 0.1
-normalize_input (flag that will rescale the input from the input data range to 80% of the output data type range, useful for float scaled datasets)
-compress_level (the IO uses BLOSC for lossless compression of the APR, this can be set from 1-9, where higher increases the compression level. Note, this can come at a significant time increase.)
-compress_type (Default: 0, loss-less compression of partilce intensities, (1,2) WNL (Balázs et al. 2017) - approach compression applied to particles (1 = without prediction, 2 = with)

-neighborhood_optimization_off turns off the neighborhood opetimization (This results in boundary Particle Cells also being increased in resolution after the Pulling Scheme step)
-output_steps Writes tiff images of the individual steps (gradient magnitude, local intensity scale, and final level of the APR calculation).

)";

#include <algorithm>
#include <iostream>
#include "ConfigAPR.h"
#include "Example_get_apr.h"

#include "algorithm/PullingSchemeSparse.hpp"

int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options;

    options = read_command_line_options(argc,argv);

    //the apr datastructure
    APR<uint16_t> apr;

    //read in the command line options into the parameters file
    apr.parameters.Ip_th = options.Ip_th;
    apr.parameters.rel_error = options.rel_error;
    apr.parameters.lambda = options.lambda;
    apr.parameters.mask_file = options.mask_file;
    apr.parameters.min_signal = options.min_signal;
    apr.parameters.SNR_min = options.SNR_min;
    apr.parameters.normalized_input = options.normalize_input;
    apr.parameters.neighborhood_optimization = options.neighborhood_optimization;
    apr.parameters.output_steps = options.output_steps;

    //where things are
    apr.parameters.input_image_name = options.input;
    apr.parameters.input_dir = options.directory;
    apr.parameters.name = options.output;
    apr.parameters.output_dir = options.output_dir;

    apr.apr_converter.fine_grained_timer.verbose_flag = false;
    apr.apr_converter.method_timer.verbose_flag = false;
    apr.apr_converter.computation_timer.verbose_flag = false;
    apr.apr_converter.allocation_timer.verbose_flag = false;
    apr.apr_converter.total_timer.verbose_flag = true;


    uint64_t level_min = 2;
    uint64_t level_max = 10;

    apr.apr_access.level_max = level_max;
    apr.apr_access.level_min = level_min;

    apr.apr_access.org_dims[0] = pow(2,level_max);
    apr.apr_access.org_dims[1] = pow(2,level_max);
    apr.apr_access.org_dims[2] = pow(2,level_max);

    std::vector<PixelData<uint8_t>> input;

    uint64_t l_max = apr.level_max() - 1;
    uint64_t l_min = apr.level_min();
    //make so you can reference the array as l
    input.resize(l_max + 1);

    for (unsigned int l = l_min; l < (l_max + 1) ;l ++){
        input[l].init(ceil((1.0 * apr.apr_access.org_dims[0]) / pow(2.0, 1.0 * l_max - l + 1)),
                                   ceil((1.0 * apr.apr_access.org_dims[1]) / pow(2.0, 1.0 * l_max - l + 1)),
                                   ceil((1.0 * apr.apr_access.org_dims[2]) / pow(2.0, 1.0 * l_max - l + 1)), EMPTY);
    }

    unsigned int number_pcs = 10000;

    for (int j = 0; j < number_pcs; ++j) {
        unsigned int index = std::rand()%input[level_max-1].mesh.size();
        input[level_max-1].mesh[index]=level_max;
    }



    APRTimer timer;
    timer.verbose_flag = true;


    timer.start_timer("Original");

    PullingScheme pullingScheme;
    pullingScheme.initialize_particle_cell_tree(apr);

    for (int i = level_min; i < level_max; ++i) {
        pullingScheme.fill(i,input[i]);
    }

    pullingScheme.pulling_scheme_main();

    timer.stop_timer();

    timer.start_timer("Sparse");

    PullingSchemeSparse pullingSchemeSparse;
    pullingSchemeSparse.initialize_particle_cell_tree(apr);

    for (int i = level_min; i < level_max; ++i) {
        pullingSchemeSparse.fill(i,input[i]);
    }

    pullingSchemeSparse.pulling_scheme_main();

    timer.stop_timer();

    uint64_t counter = 0;
    uint64_t counter_wrong = 0;


    for (int i = level_min; i < level_max; ++i) {
        for (int z = 0; z < input[i].z_num; ++z) {
            for (int x = 0; x < input[i].x_num; ++x) {
                const size_t offset_pc = input[i].x_num * z + x;
                auto mesh = pullingSchemeSparse.particle_cell_tree.data[i][offset_pc][0].mesh;
                for (int y = 0; y < input[i].y_num; ++y) {
                    const size_t offset_part_map = x * input[i].y_num + z * input[i].y_num * input[i].x_num;

                    uint64_t s_old = pullingScheme.particle_cell_tree[i].mesh[offset_part_map + y];
                    uint64_t s_sparse =mesh[y];

                    if(s_old==s_sparse){
                        if(s_old > 0) {
                            counter++;
                        }
                    } else {
                        counter_wrong++;
                    }

                }
            }
        }
    }

    //check solution
    std::cout << " wrong " <<  counter_wrong << " correct " << counter <<  std::endl;



//    //Gets the APR
//    if(apr.get_apr()){
//
//
//
//
//    } else {
//        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
//    }

}


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

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << argv[0] << std::endl;
        std::cerr << "APR version " << ConfigAPR::APR_VERSION << std::endl;
        std::cerr << "Short usage: \"" << argv[0] << " -i inputfile [-d directory] [-o outputfile]\"" << std::endl;

        std::cerr << usage << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-od"))
    {
        result.output_dir = std::string(get_command_option(argv, argv + argc, "-od"));
    } else {
        result.output_dir = result.directory;
    }

    if(command_option_exists(argv, argv + argc, "-gt"))
    {
        result.gt_input = std::string(get_command_option(argv, argv + argc, "-gt"));
    } else {
        result.gt_input = "";
    }

    if(command_option_exists(argv, argv + argc, "-lambda"))
    {
        result.lambda = std::stof(std::string(get_command_option(argv, argv + argc, "-lambda")));
    }

    if(command_option_exists(argv, argv + argc, "-I_th"))
    {
        result.Ip_th = std::stof(std::string(get_command_option(argv, argv + argc, "-I_th")));
    }

    if(command_option_exists(argv, argv + argc, "-SNR_min"))
    {
        result.SNR_min = std::stof(std::string(get_command_option(argv, argv + argc, "-SNR_min")));
    }

    if(command_option_exists(argv, argv + argc, "-min_signal"))
    {
        result.min_signal = std::stof(std::string(get_command_option(argv, argv + argc, "-min_signal")));
    }

    if(command_option_exists(argv, argv + argc, "-rel_error"))
    {
        result.rel_error = std::stof(std::string(get_command_option(argv, argv + argc, "-rel_error")));
    }

    if(command_option_exists(argv, argv + argc, "-mask_file"))
    {
        result.mask_file = std::string(get_command_option(argv, argv + argc, "-mask_file"));
    }

    if(command_option_exists(argv, argv + argc, "-compress_level"))
    {
        result.compress_level = (unsigned int)std::stoi(std::string(get_command_option(argv, argv + argc, "-compress_level")));
    }

    if(command_option_exists(argv, argv + argc, "-compress_type"))
    {
        result.compress_type = (unsigned int)std::stoi(std::string(get_command_option(argv, argv + argc, "-compress_type")));
    }

    if(command_option_exists(argv, argv + argc, "-normalize_input"))
    {
        result.normalize_input = true;
    }

    if(command_option_exists(argv, argv + argc, "-neighborhood_optimization_off"))
    {
        result.neighborhood_optimization = false;

    }

    if(command_option_exists(argv, argv + argc, "-output_steps"))
    {
        result.output_steps = true;
    }

    return result;
}