#include <algorithm>
#include <vector>
#include <array>
#include <iostream>

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/ExtraParticleData.hpp"


struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

bool command_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv) {
    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }
    if(command_option_exists(argv, argv + argc, "-i")) {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    if(command_option_exists(argv, argv + argc, "-d")) {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-o")) {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);
    std::string fileName = options.directory + options.input;
    APR<uint16_t> apr;
    apr.read_apr(fileName);

    // Get dense representation of APR
    APRIterator<uint16_t> aprIt(apr);

    ExtraParticleData<uint16_t> yRow;
    ExtraParticleData<uint16_t> xRow;
    ExtraParticleData<uint16_t> zRow;
    ExtraParticleData<uint16_t> levelRow;
    ExtraParticleData<uint64_t> globalIndexRow;


    uint64_t particle_number;
    //Basic serial iteration over all particles
    for (particle_number = 0; particle_number < aprIt.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);

        //you can then also use it to access any particle properties stored as ExtraParticleData

        yRow.data.push_back(aprIt.y());
        xRow.data.push_back(aprIt.x());
        zRow.data.push_back(aprIt.z());
        levelRow.data.push_back(aprIt.level());
        globalIndexRow.data.push_back(aprIt.global_index());

    }


    ///////////////////////////
    ///
    /// Sparse Data for GPU
    ///
    ///////////////////////////

    std::vector<std::array<std::size_t,2>> level_zx_index_start;//size = number of rows on all levels
    std::vector<std::uint16_t> y_explicit;y_explicit.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::uint16_t> particle_values;particle_values.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::size_t> level_offset;//size = number of levels

    std::size_t x = 0;
    std::size_t z = 0;

    std::size_t zx_counter = 0;
    

    level_offset.resize(aprIt.level_max()+1,UINT64_MAX);


    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
        level_offset[level] = zx_counter;

        for (z = 0; z < aprIt.spatial_index_z_max(level); ++z) {
            for (x = 0; x < aprIt.spatial_index_x_max(level); ++x) {

                zx_counter++;
                if (aprIt.set_new_lzx(level, z, x) < UINT64_MAX) {
                    level_zx_index_start.emplace_back(std::array<uint64_t,2>{aprIt.global_index(),
                                aprIt.particles_zx_end(level,z,x)-1}); //This stores the begining and end global index for each level_xz_row
                } else {
                    level_zx_index_start.emplace_back(std::array<uint64_t,2>{UINT64_MAX, 0}); //This stores the begining and end global index for each level_
                }

                for (aprIt.set_new_lzx(level, z, x);
                     aprIt.global_index() < aprIt.particles_zx_end(level, z,
                                                                   x); aprIt.set_iterator_to_particle_next_particle()) {
                    y_explicit.emplace_back(aprIt.y());
                    particle_values.emplace_back(apr.particles_intensities[aprIt]);

                }
            }

        }
    }


    ////////////////////
    ///
    /// Example of doing our level,z,x access using the GPU data structure
    ///
    /////////////////////

    std::vector<uint16_t> test_access_data;


    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {

        const int x_num = aprIt.spatial_index_x_max(level);
        //const int z_num = aprIt.spatial_index_z_max(level);

        for (z = 0; z < aprIt.spatial_index_z_max(level); ++z) {
            for (x = 0; x < aprIt.spatial_index_x_max(level); ++x) {
                if(level_offset[level]<UINT64_MAX) {
                    uint64_t level_xz_offset = level_offset[level] + x_num * z + x;
                    if (level_zx_index_start[level_xz_offset].size() > 0) {
                        uint64_t particle_index_begin = level_zx_index_start[level_xz_offset][0];
                        uint64_t particle_index_end = level_zx_index_start[level_xz_offset][1];

                        for (uint64_t global_index = particle_index_begin;
                             global_index <= particle_index_end; ++global_index) {

                            uint16_t current_particle_value = particle_values[global_index];

                            test_access_data.push_back(current_particle_value);

                        }
                    }
                }

            }
        }
    }


    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////

    bool success = true;

    for (std::size_t i = 0; i < test_access_data.size(); ++i) {
        if(apr.particles_intensities.data[i]!=test_access_data[i]){
            success = false;
        }
    }

    if(success){
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }


}
