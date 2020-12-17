//
// Created by joel on 30.11.20.
//

const char* usage = R"(

 Example applying a convolution operation (Gaussian blur) to an APR

 Usage:

 (using *.apr output of Example_get_apr)

 Example_apr_filter -i input_apr_hdf5 -d directory -o output_tiff_file

 Note: input file will be read from 'directory + input_apr_hdf5'
       if -o is given, a reconstructed TIFF image will be written to: 'directory + output_tiff_file + ".tif"'

 Options:

 -use_cuda      if this flag is given, the convolution is performed on the GPU (requires library to be built with CUDA enabled)

)";


#include "Example_apr_filter.hpp"


int main(int argc, char **argv) {

    // INPUT PARSING
    cmdLineOptions options = read_command_line_options(argc, argv);

    std::string file_name = options.directory + options.input;

    APRTimer timer(true);

    timer.start_timer("Read APR and particles from file");

    APR apr;
    ParticleData<uint16_t> parts;

    //read file
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);
    aprFile.read_particles(apr,"particles",parts);

    timer.stop_timer();

    auto stencil = APRStencil::create_gaussian_filter<float>(/*sigma*/{1, 1, 1}, /*stencil size*/{5, 5, 5}, /*normalize*/true);
    ParticleData<float> output;

    bool done = false;

    // GPU convolution
    if(options.use_cuda) {
#ifdef APR_USE_CUDA
        timer.start_timer("APR Convolution CUDA");
        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();
        VectorData<float> stencil_vd;
        stencil_vd.resize(125); // stencil must be 5x5x5!
        std::copy(stencil.mesh.begin(), stencil.mesh.end(), stencil_vd.begin());
        ParticleData<float> tree_data;
        isotropic_convolve_555(access, tree_access, parts.data, output.data, stencil_vd, tree_data.data,
                               /*reflect boundary*/true, /*downsample stencil*/true, /*normalize stencils*/true);

        done = true;
        timer.stop_timer();
#else
        std::cout << "Option -use_cuda was given, but LibAPR was not built with CUDA enabled. Using CPU implementation." << std::endl;
#endif
    }

    // CPU convolution (this works for stencils of any size in 1-3 dimensions)
    if(!done) {
        timer.start_timer("APR Convolution CPU");
        APRFilter::convolve_pencil(apr, stencil, parts, output, /*reflect boundary*/true,
                                   /*downsample stencil*/true, /*normalize stencils*/true);
        timer.stop_timer();
    }

    // If output option given, reconstruct pixel image from output and write to file
    if(options.output.length() > 0) {
        // reconstruct pixel image from gradient
        timer.start_timer("reconstruct pixel image");
        PixelData<float> output_image;
        APRReconstruction::interp_img(apr, output_image, output);
        timer.stop_timer();

        timer.start_timer("write pixel image to file");
        std::string image_file_name = options.directory + options.output + ".tif";
        TiffUtils::saveMeshAsTiff(image_file_name, output_image);
        timer.stop_timer();
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
        std::cerr << "Short Usage: \"Example_compute_gradient -i input_apr_file -d directory\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-use_cuda"))
    {
        result.use_cuda = true;
    }


    return result;

}