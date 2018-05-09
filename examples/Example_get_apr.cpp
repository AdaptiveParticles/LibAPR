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
-store_delta (stores the delta between an APR reconstruction and the original image as an additional hdf5 file)
-compress_level (the IO uses BLOSC for lossless compression of the APR, this can be set from 1-9, where higher increases the compression level. Note, this can come at a significant time increase.)
)";

#include <algorithm>
#include <iostream>
#include "ConfigAPR.h"
#include "Example_get_apr.h"



int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options;

    options = read_command_line_options(argc,argv);

    //the apr datastructure
    APR<uint16_t> apr;

    APRConverter<uint16_t> apr_converter;

    //read in the command line options into the parameters file
    apr_converter.par.Ip_th = options.Ip_th;
    apr_converter.par.rel_error = options.rel_error;
    apr_converter.par.lambda = options.lambda;
    apr_converter.par.mask_file = options.mask_file;
    apr_converter.par.min_signal = options.min_signal;
    apr_converter.par.SNR_min = options.SNR_min;
    apr_converter.par.normalized_input = options.normalize_input;

    //where things are
    apr_converter.par.input_image_name = options.input;
    apr_converter.par.input_dir = options.directory;
    apr_converter.par.name = options.output;
    apr_converter.par.output_dir = options.output_dir;

    apr_converter.fine_grained_timer.verbose_flag = false;
    apr_converter.method_timer.verbose_flag = false;
    apr_converter.computation_timer.verbose_flag = false;
    apr_converter.allocation_timer.verbose_flag = false;
    apr_converter.total_timer.verbose_flag = true;

    //Gets the APR
    if(apr_converter.get_apr(apr)){

        //Below is IO and outputting of the Implied Resolution Function through the Particle Cell level.

        //output
        std::string save_loc = options.output_dir;
        std::string file_name = options.output;

        APRTimer timer;

        timer.verbose_flag = true;

        PixelData<uint16_t> level;

        apr.interp_depth_ds(level);

        std::cout << std::endl;

        std::cout << "Saving down-sampled Particle Cell level as tiff image" << std::endl;

        std::string output_path = save_loc + file_name + "_level.tif";
        //write output as tiff
        TiffUtils::saveMeshAsTiff(output_path, level);

        std::cout << std::endl;
        float original_pixel_image_size = (2.0f*apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2))/(1000000.0);
        std::cout << "Original image size: " << original_pixel_image_size << " MB" << std::endl;

        timer.start_timer("writing output");

        std::cout << "Writing the APR to hdf5..." << std::endl;

        //feel free to change
        unsigned int blosc_comp_type = BLOSC_ZSTD;
        unsigned int blosc_comp_level = options.compress_level;
        unsigned int blosc_shuffle = 1;

        //write the APR to hdf5 file
        float apr_file_size = apr.write_apr(save_loc,file_name,blosc_comp_type,blosc_comp_level,blosc_shuffle);

        timer.stop_timer();

        float computational_ratio = (1.0f*apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2))/(1.0f*apr.total_number_particles());

        std::cout << std::endl;
        std::cout << "Computational Ratio (Pixels/Particles): " << computational_ratio << std::endl;
        std::cout << "Lossy Compression Ratio: " << original_pixel_image_size/apr_file_size << std::endl;
        std::cout << std::endl;

        if(options.store_delta){
            //feel free to change
            unsigned int blosc_comp_type = BLOSC_ZSTD;
            unsigned int blosc_comp_level = options.compress_level;
            unsigned int blosc_shuffle = 1;

            PixelData<uint16_t> recon_image;

            apr.interp_img(recon_image, apr.particles_intensities);

            TiffUtils::TiffInfo inputTiff(options.directory + options.input);
            PixelData<uint16_t> inputImage = TiffUtils::getMesh<uint16_t>(inputTiff);

            PixelData<int16_t> diff_image(inputImage.y_num,inputImage.x_num,inputImage.z_num,0);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < inputImage.mesh.size(); ++i) {

                diff_image.mesh[i] = 2 * abs(recon_image.mesh[i] - inputImage.mesh[i]) +
                                         ((recon_image.mesh[i] - inputImage.mesh[i]) > 0);
            }

            std::cout << "Storing diff image for lossless reconstruction" << std::endl;
            APRWriter aprWriter;
            float file_size = aprWriter.write_mesh_to_hdf5(diff_image,save_loc,file_name,blosc_comp_type,blosc_comp_level,blosc_shuffle);
            std::cout << "Size of the image diff: " << file_size << " MB" << std::endl;

            std::cout << std::endl;
            std::cout << "Lossless Compression Ratio (APR + diff): " << original_pixel_image_size/(file_size + apr_file_size) << std::endl;
            std::cout << std::endl;

            float file_size_org = aprWriter.write_mesh_to_hdf5(inputImage,save_loc,file_name,blosc_comp_type,blosc_comp_level,blosc_shuffle);
            std::cout << "Size of the pixel image compressed: " << file_size_org << " MB" << std::endl;

            std::cout << std::endl;
            std::cout << "Lossless Compression Ratio (Pixel Image): " << original_pixel_image_size/(file_size_org) << std::endl;
            std::cout << std::endl;

        }


        } else {
        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
    }

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

    if(command_option_exists(argv, argv + argc, "-normalize_input"))
    {
        result.normalize_input = true;
    }

    if(command_option_exists(argv, argv + argc, "-store_delta"))
    {
        result.store_delta = true;
    }

    return result;
}