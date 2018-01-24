//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Example calculating the gradient, and gradient magnitude of the APR.
///
/// Produces *_paraview.h5 file and *_paraview.xmf and tiff images of the gradient
///
/// To use load the xmf file in Paraview, and select Xdmf Reader. Then click the small eye, to visualize the dataset. (Enable opacity mapping for surfaces, option can be useful)
///
/// Usage:
///
/// (using output of Example_compute_gradient)
///
/// Example_compute_gradient -i input_apr_hdf5 -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "Example_compute_gradient.hpp"
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
        std::cerr << "Usage: \"Example_compute_gradient -i input_apr_file -d directory\"" << std::endl;
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

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    //Calculate the gradient of the APR

    ExtraParticleData<std::vector<float>> gradient; //vector for holding the derivative in the three directions, initialized to have the same number of elements as particles.

    std::vector<float> init_val = {0,0,0};

    gradient.data.resize(apr.total_number_particles(),init_val);

    APRIterator<uint16_t> apr_iterator(apr);
    APRIterator<uint16_t> neighbour_iterator(apr);

    uint64_t particle_number;

    const std::vector<std::vector<uint8_t>> group_directions = {{0,1},{2,3},{4,5}}; // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
    const std::vector<float> sign = {1.0,-1.0};

    timer.start_timer("Calculate the gradient in each direction for the APR");

    //
    //  Calculates an estimate of the gradient in each direciton, using an average of two one sided FD of the gradient using the average of particles for children.
    //

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,neighbour_iterator)
    for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
        //needed step for any parallel loop (update to the next part)

        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        float current_intensity = apr_iterator(apr.particles_intensities);

        //loop over all the neighbours and set the neighbour iterator to it
        for (int dimension = 0; dimension < 3; ++dimension) {
            float gradient_estimate= 0;

            float counter_dir = 0;

            for (int i = 0; i < 2; ++i) {
                float intensity_sum = 0;
                float count_neighbours = 0;

                const uint8_t direction = group_directions[dimension][i];

                apr_iterator.find_neighbours_in_direction(direction);

                const float distance_between_particles = 0.5*pow(2,apr_iterator.level_max() - apr_iterator.level())+0.5*pow(2,apr_iterator.level_max()-neighbour_iterator.level()); //in pixels

                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                    if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                        intensity_sum += neighbour_iterator(apr.particles_intensities);
                        count_neighbours++;
                    }
                }
                if(count_neighbours > 0) {
                    gradient_estimate += sign[i] * (current_intensity - intensity_sum / count_neighbours) /
                                         distance_between_particles; //calculates the one sided finite difference in each direction using the average of particles
                    counter_dir++;
                }
            }
            //store the estimate of the gradient
            apr_iterator(gradient)[dimension] = gradient_estimate/counter_dir;
        }

    }

    timer.stop_timer();

    ExtraParticleData<float> gradient_magnitude(apr);
    //compute the magnitude of the gradient, scale it by 5 for visualization when writing as uint16 int
    gradient.map(apr,gradient_magnitude,[](const std::vector<float> &a) { return 5*sqrt(pow(a[0], 2.0) + pow(a[1], 2.0) + pow(a[2], 2.0)); });

    // write result to image
    MeshData<float> gradient_magnitude_image;
    apr.interp_img(gradient_magnitude_image,gradient_magnitude);
    std::string image_file_name = options.directory + name + "_gradient.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, gradient_magnitude_image);

    // also write to a paraview viewable file
    apr.write_apr_paraview(options.directory,name + "_gradient",gradient_magnitude);

    ExtraParticleData<float> gradient_y(apr);
    gradient.map(apr,gradient_y,[](const std::vector<float> &a) { return 5*abs(a[0]); });

    apr.interp_img(gradient_magnitude_image,gradient_y);
    image_file_name = options.directory + name + "_gradient_y.tif";
    TiffUtils::saveMeshAsTiffUint16(image_file_name, gradient_magnitude_image);
}