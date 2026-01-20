const char* usage = R"(
Converts images to APR format: Takes input directory with uint16_t input tiff images and generates the APRs and saves it as hdf5.
The hdf5 output of this program can be used with the other apr examples, and also viewed with HDFView.

Usage:
======
Example_get_multiapr -d input_directory [-od output_directory]

Additional settings (High Level):
=================================
-I_th       intensity_threshold (will ignore areas of image below this threshold, useful for removing camera artifacts or auto-fluorescence)
-sigma_th   lower threshold for the local intensity scale
-grad_th    ignore areas in the image where the gradient magnitude is lower than this value

-skipOutputMessages     produce less output/debug messages
-doNotSaveAPRs          do not save output APR files (good for benchmarking)
-r                      number of repetitions - default value = 1 means that all input files are processed only once
                        for higher number input files are processed multiple times (good for benchmarking)

Advanced (Direct) Settings:
===========================
-lambda lambda_value (directly set the value of the gradient smoothing parameter lambda (reasonable range 0.1-10, default: 3)
-rel_error rel_error_value (Reasonable ranges are from .08-.15), Default: 0.1
-neighborhood_optimization_off turns off the neighborhood optimization (This results in boundary Particle Cells also being increased in resolution after the Pulling Scheme step)
)";


#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include "ConfigAPR.h"
#include "io/APRFile.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/APR/APR.hpp"
#include "algorithm/APRConverter.hpp"


struct cmdLineOptions {
    std::string directory;
    std::string output_dir;

    float lambda = 3.0;
    float Ip_th = 0;
    float grad_th = 1;
    float sigma_th = 0;
    float rel_error = 0.1;

    bool neighborhood_optimization = true;

    int numOfRepetitions = 1;
    int numOfStreams = 3;
    bool doNotSaveAPRs = false;
    bool skipOutputMessages = false;
};

bool command_option_exists(const char **begin, const char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

const char* get_command_option(const char **begin, const char **end, const std::string &option)
{
    if (const char** itr = std::find(begin, end, option); itr != end && ++itr != end) {
        return *itr;
    }
    return nullptr;
}

void printUsage() {
    std::cerr << "APR version " << ConfigAPR::APR_VERSION << std::endl;
    std::cerr << usage << std::endl;
    exit(1);
}

cmdLineOptions read_command_line_options(const int argc, const char **argv) {

    cmdLineOptions options;

    // --------- Print usage if no args provided
    if (argc == 1) printUsage();

    // --------- Read params

    // Input Directory
    if (command_option_exists(argv, argv + argc, "-d")) {
        options.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    } else {
        std::cout << "Input directory required" << std::endl;
        exit(2);
    }

    // Output Directory
    if (command_option_exists(argv, argv + argc, "-od")) {
        options.output_dir = std::string(get_command_option(argv, argv + argc, "-od"));
    } else {
        options.output_dir = options.directory;
    }

    if (command_option_exists(argv, argv + argc, "-lambda")) {
        options.lambda = std::stof(std::string(get_command_option(argv, argv + argc, "-lambda")));
    }

    if (command_option_exists(argv, argv + argc, "-I_th")) {
        options.Ip_th = std::stof(std::string(get_command_option(argv, argv + argc, "-I_th")));
    }

    if (command_option_exists(argv, argv + argc, "-grad_th")) {
        options.grad_th = std::stof(std::string(get_command_option(argv, argv + argc, "-grad_th")));
    }

    if (command_option_exists(argv, argv + argc, "-sigma_th")) {
        options.sigma_th = std::stof(std::string(get_command_option(argv, argv + argc, "-sigma_th")));
    }

    if (command_option_exists(argv, argv + argc, "-rel_error")) {
        options.rel_error = std::stof(std::string(get_command_option(argv, argv + argc, "-rel_error")));
    }

    if (command_option_exists(argv, argv + argc, "-neighborhood_optimization_off")) {
        options.neighborhood_optimization = false;
    }

    if (command_option_exists(argv, argv + argc, "-skipOutputMessages")) {
        options.skipOutputMessages = true;
    }

    if (command_option_exists(argv, argv + argc, "-doNotSaveAPRs")) {
        options.doNotSaveAPRs = true;
    }

    if (command_option_exists(argv, argv + argc, "-r")) {
        options.numOfRepetitions = std::stoi(std::string(get_command_option(argv, argv + argc, "-r")));
    }

    if (command_option_exists(argv, argv + argc, "-numOfStreams")) {
        options.numOfStreams = std::stoi(std::string(get_command_option(argv, argv + argc, "-numOfStreams")));
    }

    return options;
}

/* Finds all tiff files (with possible different extensions) in provided directory */
auto getTiffFilesFromDir(const std::string &directory_path) {
    namespace fs = std::filesystem;

    std::vector<fs::path> tif_files;

    try {
        for (const auto& entry : fs::directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                if (auto ext = entry.path().extension().string(); ext == ".tif" || ext == ".tiff" || ext == ".TIF" || ext == ".TIFF") {
                    tif_files.push_back(entry.path());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << '\n';
        exit(2);
    }

    return tif_files;
}

int runAPR(const cmdLineOptions &options) {

    using ImgType = uint16_t;
    using ImgContainer = PixelData<ImgType>;

    APRConverter<ImgType> aprConverter;

    // read in the command line options into the parameters file
    aprConverter.par.input_dir = options.directory;
    aprConverter.par.output_dir = options.output_dir;

    aprConverter.par.lambda = options.lambda;
    aprConverter.par.Ip_th = options.Ip_th;
    aprConverter.par.grad_th = options.grad_th;
    aprConverter.par.sigma_th = options.sigma_th;
    aprConverter.par.rel_error = options.rel_error;

    aprConverter.par.neighborhood_optimization = options.neighborhood_optimization;

    // TODO: read here all input files instead of options.input
    auto tifFiles = getTiffFilesFromDir(options.directory);
    std::vector<std::unique_ptr<ImgContainer>> input_images;
    std::vector<ImgContainer *> input_images_raw;
    std::vector<std::unique_ptr<APR>> APRs;
    std::vector<APR*> APRs_raw;
    std::vector<std::unique_ptr<VectorData<ImgType>>> partIntensities;
    std::vector<VectorData<ImgType> *> partIntensities_raw;

    // Load all images from input directory, check if they have same resolution
    // Also create APR and intensities objects to be filled by pipeline later
    int firstOne = true;
    PixelDataDim sizeOfInput;
    for (const auto &file : tifFiles) {
        // Read a file and store it, also keep a vector of raw pointers to read images since this is needed by APRConverter
        input_images.push_back(std::make_unique<ImgContainer>(TiffUtils::getMesh<ImgType>(file)));
        input_images_raw.push_back(input_images.back().get());
        if (firstOne) {
            firstOne = false;
            sizeOfInput = input_images.back().get()->getDimension();
        }
        else if (input_images.back().get()->getDimension() != sizeOfInput) {
                std::cerr << "Input images must have the same dimension." << std::endl;
                exit(2);
        }

        // We need as many APR objects as input images, and also raw pointer for APRConverter
        APRs.push_back(std::make_unique<APR>(APR{}));
        APRs_raw.push_back(APRs.back().get());

        // And same for particle intensities...
        partIntensities.push_back(std::make_unique<VectorData<ImgType>>(VectorData<ImgType>{}));
        partIntensities_raw.push_back(partIntensities.back().get());
    }
    // To proces input image multiple times (for benchmarking etc.) we 'multiply input data' by adding extra APR and particle intensity objects and
    // by copying input raw pointer to images to 'pretend' that we have a lot of input images.
    size_t numOfInputImages = input_images_raw.size();
    for (int m = 1; m < options.numOfRepetitions; m++) {
        for (size_t n = 0; n < numOfInputImages; n++) {
            input_images_raw.push_back(input_images[n].get());
            APRs.push_back(std::make_unique<APR>(APR{}));
            APRs_raw.push_back(APRs.back().get());
            partIntensities.push_back(std::make_unique<VectorData<ImgType>>(VectorData<ImgType>{}));
            partIntensities_raw.push_back(partIntensities.back().get());
        }
    }

    std::cout << std::endl;

    APRTimer timer(false);
    timer.start_timer("GPU pipeline (mem allocation, processing, sampling) ");
    if (aprConverter.get_apr_cuda_multistreams(APRs_raw, input_images_raw, partIntensities_raw, options.numOfStreams)) {
        timer.stop_timer();
        size_t numOfImagesToProcess = input_images_raw.size(); // might be 'processInputMultipleTimes' times bigger than num of input images
        if (!options.skipOutputMessages) std::cout << std::endl;

        for (size_t i = 0; i < numOfImagesToProcess && !options.doNotSaveAPRs; i++) {
            if (!options.skipOutputMessages) std::cout << "Postprocessing " << i+1 << "/" << numOfImagesToProcess << " image...\n";
            auto &apr = *APRs[i].get(); // currently process APR
            auto &particle_intensities = *partIntensities[i].get(); // intensities sampled for current APR


            // ------------ TODO: remove me later, this is quick test for Cpu vs Gpu before real test is written
            // std::cout << apr.linearAccess.y_vec.size() << " particles in APR" << std::endl;
            // std::cout << particle_intensities.size() << " intensities in CPU in APR" << std::endl;
            if (apr.linearAccess.y_vec.size() != particle_intensities.size()) {std::cerr << "CPU vs GPU number of particles differ!" << std::endl;}
            ParticleData<ImgType> particle_intensities_cpu;
            particle_intensities_cpu.sample_image(apr, *input_images_raw[i]); // sample your particles from your image
            int errorCnt = 0;
            for (size_t j = 0 ; j < particle_intensities.size(); ++j) {
                if (particle_intensities_cpu[j]  != particle_intensities[j]) {
                    errorCnt++;
                    // std::cout << "Mismatch at " << j << " CPU: " << particle_intensities_cpu[j] << " GPU: " << particle_intensities[j] << std::endl;
                }
            }
            if (errorCnt > 0) std::cout << errorCnt << " errors for index=" << i << std::endl;
            // ---------------------------------------------------------------------------------------------------

            // Output name is like base of input filename + extension ".apr"
            // Extra number is added for 'multiplied' input images
            auto outputDir = std::filesystem::path(options.output_dir);
            const std::filesystem::path& p(tifFiles[i % numOfInputImages]);
            std::string num = (i >= numOfInputImages) ? std::to_string(i) : "";
            std::string outputFileName = p.stem().string() + num + ".apr";

            //write the APR to hdf5 file
            timer.start_timer("writing output");
            APRFile aprFile;
            aprFile.open(outputDir / outputFileName);
            aprFile.write_apr(apr, 0, "t", false);
            ParticleData<ImgType> pd;
            pd.data = std::move(particle_intensities);
            aprFile.write_particles("particles",pd);
            timer.stop_timer();

            // Print some output statistics
            float aprImageSizeInMB = aprFile.current_file_size_MB();
            double originalImageSizeInMB = sizeof(ImgType) * static_cast<double>(apr.org_dims(0) * apr.org_dims(1) * apr.org_dims(2)) / 1'000'000.0;

            if (!options.skipOutputMessages) {
                std::cout << "Save filename: [" << outputFileName << "]" << std::endl;
                std::cout << "Computational Ratio (Pixels/Particles): " << apr.computational_ratio() << std::endl;
                std::cout << "Original / APR image size:              " << originalImageSizeInMB << " / " << aprImageSizeInMB <<" MB" << std::endl;
                std::cout << "Lossy Compression Ratio:                " << originalImageSizeInMB/aprImageSizeInMB << std::endl;
                std::cout << std::endl;
            }
        }
    }
    else {
        std::cout << "Oops, something went wrong. APR not computed :(" << std::endl;
    }

    std::cout << "DONE!\n";

    return 0;
}


int main(const int argc, const char **argv) {
    const cmdLineOptions options = read_command_line_options(argc, argv);
    const auto result = runAPR(options);

    return result;
}
