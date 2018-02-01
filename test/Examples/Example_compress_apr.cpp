//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Example of performing additional compression on the APR intensities (APR adaptation of the within noise lossy compression from Balázs et al. 2017, A real-time compression library for
/// microscopy images)
///
/// Usage:
///
/// (using output of Example_compress_apr)
///
/// Example_compress_apr -i input_image_tiff -d input_directory
/// #TODO add compression type options and make it all output on entry
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "Example_compress_apr.h"
#include "src/io/TiffUtils.hpp"

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
        std::cerr << "Usage: \"Example_compress_apr -i input_apr_file -d directory\"" << std::endl;
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

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    apr.parameters.input_dir = options.directory;

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    APRCompress<uint16_t> comp;
    ExtraParticleData<uint16_t> symbols;

    comp.set_quantization_factor(1);
    comp.set_compression_type(2);

    timer.start_timer("compress");
    apr.write_apr(options.directory ,name + "_compress",comp,BLOSC_ZSTD,1,2);
    timer.stop_timer();

    timer.start_timer("decompress");
    apr.read_apr(options.directory + name + "_compress_apr.h5");
    timer.stop_timer();

    MeshData<uint16_t> img;
    apr.interp_img(img,apr.particles_intensities);
    std::string output = options.directory + name + "_compress.tif";
    TiffUtils::saveMeshAsTiff(output, img);
}


