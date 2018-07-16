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

#include "algorithm/APRConverterBatch.hpp"

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

    apr.parameters.Ip_th = 900;
    apr.parameters.min_signal = 400;
    apr.parameters.sigma_th = 400;
    apr.parameters.sigma_th_max = 100;
    apr.parameters.lambda = 1;
    apr.parameters.rel_error = 0.15;

    //where things are
    apr.parameters.input_image_name = options.input;
    apr.parameters.input_dir = options.directory;
    apr.parameters.name = options.output;
    apr.parameters.output_dir = options.output_dir;

    apr.apr_converter.fine_grained_timer.verbose_flag = false;
    apr.apr_converter.method_timer.verbose_flag = true;
    apr.apr_converter.computation_timer.verbose_flag = false;
    apr.apr_converter.allocation_timer.verbose_flag = false;
    apr.apr_converter.total_timer.verbose_flag = true;

    //output
    std::string save_loc = options.output_dir;
    std::string file_name = options.output;

    APR<uint16_t> aprB;

    APRConverterBatch<uint16_t> aprConverterBatch;

    aprConverterBatch.method_timer.verbose_flag = true;

    apr.apr_compress.set_quantization_factor(2);
    apr.apr_compress.set_compression_type(1);

    aprConverterBatch.par = apr.parameters;

    APRTimer timer;
    timer.verbose_flag = true;

    timer.start_timer("get APR patch");

    aprConverterBatch.get_apr(aprB);

    timer.stop_timer();

    timer.start_timer("get APR classic");

    apr.parameters.output_steps = false;

    //apr.get_apr();

    timer.stop_timer();


    std::cout << "Total number of particles patch: " << aprB.total_number_particles() << " original: " << apr.total_number_particles() << std::endl;
//
//    PixelData<uint16_t> pc_img;
//
//    aprB.interp_level(pc_img);
//
//
//    std::string output_path = save_loc + file_name + "_pc.tif";
//    //write output as tiff
//    TiffUtils::saveMeshAsTiff(output_path, pc_img);
//
//    apr.interp_level(pc_img);
//
//    output_path = save_loc + file_name + "_org_pc.tif";
//    //write output as tiff
//    TiffUtils::saveMeshAsTiff(output_path, pc_img);

    aprB.write_apr(save_loc,file_name,6,9,2,true);

    //FileSizeInfo fileSizeInfo2 = apr.write_apr(save_loc,file_name,6,9,2,false);

//    APR<uint16_t> apr1;
//
//    uint64_t level_min = 2;
//    uint64_t level_max = 7;
//
//    apr.apr_access.level_max = level_max;
//    apr.apr_access.level_min = level_min;
//
//    apr.apr_access.org_dims[0] = pow(2,level_max);
//    apr.apr_access.org_dims[1] = pow(2,level_max);
//    apr.apr_access.org_dims[2] = pow(2,level_max);
//
//    apr1.apr_access.level_max = level_max;
//    apr1.apr_access.level_min = level_min;
//
//    apr1.apr_access.org_dims[0] = pow(2,level_max);
//    apr1.apr_access.org_dims[1] = pow(2,level_max);
//    apr1.apr_access.org_dims[2] = pow(2,level_max);
//
//    std::vector<PixelData<uint8_t>> input;
//
//    uint64_t l_max = apr.level_max() - 1;
//    uint64_t l_min = apr.level_min();
//    //make so you can reference the array as l
//    input.resize(l_max + 1);
//
//    for (unsigned int l = l_min; l < (l_max + 1) ;l ++){
//        input[l].init(ceil((1.0 * apr.apr_access.org_dims[0]) / pow(2.0, 1.0 * l_max - l + 1)),
//                                   ceil((1.0 * apr.apr_access.org_dims[1]) / pow(2.0, 1.0 * l_max - l + 1)),
//                                   ceil((1.0 * apr.apr_access.org_dims[2]) / pow(2.0, 1.0 * l_max - l + 1)), EMPTY);
//    }
//
//    unsigned int number_pcs = input[l_max].mesh.size()*.0001;
//
//    std::cout << "full " <<  input[l_max].mesh.size() << std::endl;
//
//    for (int j = 0; j < number_pcs; ++j) {
//        unsigned int index = std::rand()%input[level_max-1].mesh.size();
//        input[level_max-1].mesh[index]=level_max;
//    }
//
//
//    APRTimer timer;
//    timer.verbose_flag = true;
//
//
//    timer.start_timer("Original");
//
//    PullingScheme pullingScheme;
//    pullingScheme.initialize_particle_cell_tree(apr);
//
//    for (int i = level_min; i < level_max; ++i) {
//        pullingScheme.fill(i,input[i]);
//    }
//
//    pullingScheme.pulling_scheme_main();
//
//    timer.stop_timer();
//
//    timer.start_timer("Sparse init");
//
//    PullingSchemeSparse pullingSchemeSparse;
//    pullingSchemeSparse.initialize_particle_cell_tree(apr);
//
//    imagePatch patch;
//
//    patch.x_begin = 0;
//    patch.x_end = apr.apr_access.org_dims[1];
//    patch.x_offset = 0;
//
//    patch.y_begin = 0;
//    patch.y_end = apr.apr_access.org_dims[0];
//    patch.y_offset = 0;
//
//    patch.z_begin = 0;
//    patch.z_end = apr.apr_access.org_dims[2];
//    patch.z_offset = 0;
//
//    for (int i = level_min; i < level_max; ++i) {
//        pullingSchemeSparse.fill(i,input[i],patch);
//    }
//    timer.stop_timer();
//
//    timer.start_timer("Pulling Scheme");
//
//    pullingSchemeSparse.pulling_scheme_main();
//
//    timer.stop_timer();
//
//    uint64_t counter = 0;
//    uint64_t counter_wrong = 0;
//
//
//    apr1.apr_access.initialize_structure_from_particle_cell_tree(apr1,pullingScheme.particle_cell_tree);
//
//    apr.apr_access.initialize_structure_from_particle_cell_tree_sparse(apr,pullingSchemeSparse.particle_cell_tree);
//
//    std::cout << apr1.total_number_particles() <<  " " << apr.total_number_particles() << std::endl;
//
//    std::cout << apr1.apr_access.total_number_gaps <<  " " << apr.apr_access.total_number_gaps << std::endl;
//
//    std::cout << apr1.apr_access.total_number_non_empty_rows <<  " " << apr.apr_access.total_number_non_empty_rows << std::endl;
//
//
//    for (int i = level_min; i < level_max; ++i) {
//        for (int z = 0; z < input[i].z_num; ++z) {
//            for (int x = 0; x < input[i].x_num; ++x) {
//                const size_t offset_pc = input[i].x_num * z + x;
//
//                if(apr1.apr_access.gap_map.data[i][offset_pc].size() > 0) {
//
//                    auto org = apr1.apr_access.gap_map.data[i][offset_pc][0].map;
//                    auto sp = apr.apr_access.gap_map.data[i][offset_pc][0].map;
//
//                    auto it_sp = sp.begin();
//
//                    for (auto it=org.begin(); it!=org.end(); ++it) {
//                        size_t y = it->first;
//                        size_t ysp = it_sp->first;
//
//                        YGap_map ygorg = it->second;
//                        YGap_map ygsp = it_sp->second;
//
//                        if(y!=ysp){
//
//                            counter_wrong++;
//                        }
//
//                        if(ygorg.global_index_begin_offset!=ygsp.global_index_begin_offset){
//
//                            counter_wrong++;
//                        }
//
//                        if(ygorg.y_end!=ygsp.y_end){
//
//                            counter_wrong++;
//                        }
//
//
//                        it_sp++;
//
//                    }
//
//                    if (org.size() == sp.size()) {
//
//                    } else {
//                        counter_wrong++;
//                    }
//                }
//
//
//            }
//        }
//    }
//
//    PixelData<uint16_t> level;
//
//    apr.interp_level(level);
//
//    std::cout << std::endl;
//
//    std::cout << "Saving Particle Cell level as tiff image" << std::endl;
//
//    output_path = save_loc + file_name + "test_level.tif";
//    //write output as tiff
//    TiffUtils::saveMeshAsTiff(output_path, level);
//
//
//
////    for (int i = level_min; i < level_max; ++i) {
////        for (int z = 0; z < input[i].z_num; ++z) {
////            for (int x = 0; x < input[i].x_num; ++x) {
////                const size_t offset_pc = input[i].x_num * z + x;
////                auto& mesh = pullingSchemeSparse.particle_cell_tree.data[i][offset_pc][0].mesh;
////                for (int y = 0; y < input[i].y_num; ++y) {
////                    const size_t offset_part_map = x * input[i].y_num + z * input[i].y_num * input[i].x_num;
////
////                    uint64_t s_old = pullingScheme.particle_cell_tree[i].mesh[offset_part_map + y];
////                    uint64_t s_sparse =mesh[y];
////
////                    if(s_old==s_sparse){
////                        if(s_old > 0) {
////                            counter++;
////                        }
////                    } else {
////                        counter_wrong++;
////                    }
////
////                }
////            }
////        }
////    }
//
//    //check solution
//    std::cout << " wrong " <<  counter_wrong << " correct " << counter <<  std::endl;



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