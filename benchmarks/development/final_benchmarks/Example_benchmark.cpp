//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Examples of simple iteration an access to Particle Cell, and particle information. (See Example_neigh, for neighbor access)
///
/// Usage:
///
/// (using output of Example_get_apr)
///
/// Example_apr_iterate -i input_image_tiff -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <src/algorithm/APRConverter.hpp>

#include "Example_benchmark.h"

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
        std::cerr << "Usage: \"Example_apr_iterate -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;

}


int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = false;

    // APR datastructure
    APR<uint16_t> apr;

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-4,name.end());

    APRConverter<uint16_t> apr_converter;

    apr_converter.par.Ip_th = 1032;
    apr_converter.par.sigma_th = 242.822;
    apr_converter.par.sigma_th_max = 95.9877;
    apr_converter.par.rel_error = 0.1;
    apr_converter.par.lambda = 3;

    apr_converter.par.input_dir = options.directory;
    apr_converter.par.input_image_name  = options.input;

    apr_converter.fine_grained_timer.verbose_flag = false;
    apr_converter.method_timer.verbose_flag = false;
    apr_converter.allocation_timer.verbose_flag = false;
    apr_converter.computation_timer.verbose_flag = true;

    TiffUtils::TiffInfo inputTiff(apr_converter.par.input_dir + apr_converter.par.input_image_name);
    MeshData<uint16_t> input_image = TiffUtils::getMesh<uint16_t>(inputTiff);

    apr_converter.get_apr_method(apr, input_image);

    APRBenchmark apr_benchmarks;

//    float num_repeats = 1;
//
//    apr_benchmarks.pixels_linear_neighbour_access<uint16_t,float>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),num_repeats);
//    apr_benchmarks.apr_linear_neighbour_access<uint16_t,float>(apr,num_repeats);
//
//    float num_repeats_random = 10000000;
//
//    apr_benchmarks.pixel_neighbour_random<uint16_t,float>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2), num_repeats_random);
//    apr_benchmarks.apr_random_access<uint16_t,float>(apr,num_repeats_random);

    APRCompress<uint16_t> apr_compress;

    APRWriter apr_writer;

    ExtraParticleData<uint16_t> intensities;
    intensities.copy_parts(apr,apr.particles_intensities);

    apr_compress.set_compression_type(1);

    timer.verbose_flag = true;

    timer.start_timer("compress");
    float size = apr_writer.write_apr(apr,options.directory ,name + "_compress",apr_compress,BLOSC_ZSTD,3,2);
    timer.stop_timer();


    apr.particles_intensities.copy_parts(apr,intensities);
    apr_compress.set_compression_type(2);

    timer.start_timer("compress1");
    float size2 = apr_writer.write_apr(apr,options.directory ,name + "_compress1",apr_compress,BLOSC_ZSTD,3,2);
    timer.stop_timer();


    apr.particles_intensities.copy_parts(apr,intensities);
    apr_compress.set_compression_type(0);

    timer.start_timer("compress2");
    float size3 = apr_writer.write_apr(apr,options.directory ,name + "_compress2",apr_compress,BLOSC_ZSTD,3,2);
    timer.stop_timer();

    float size4 = apr_writer.write_particles_only(options.directory ,name + "_parts_only",intensities);


    std::cout << (size3 - size4)/size3  << std::endl;

}
