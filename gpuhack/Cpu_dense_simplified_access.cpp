//
// Created by cheesema on 05.03.18.
//

//
// Created by cheesema on 05.03.18.
//
//
// Created by cheesema on 28.02.18.
//

//
// Created by cheesema on 28.02.18.
//

//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(


)";


#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"

#include "algorithm/APRConverter.hpp"
#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include <numerics/APRNumerics.hpp>
#include <numerics/APRComputeHelper.hpp>

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);


template<typename T,typename ParticleDataType>
void update_dense_array(const uint64_t level,const uint64_t z,APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,MeshData<T>& temp_vec,ExtraParticleData<ParticleDataType>& particleData){

    uint64_t x;

    const uint64_t x_num_m = temp_vec.x_num;
    const uint64_t y_num_m =  temp_vec.y_num;


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
    for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

        //
        //  This loop recreates particles at the current level, using a simple copy
        //

        uint64_t mesh_offset = (x+1)*y_num_m + x_num_m*y_num_m*(z % 3);

        apr_iterator.set_new_lzx(level, z, x);
        for (unsigned long gap = 0;
             gap < apr_iterator.number_gaps(); apr_iterator.move_gap(gap)) {

            uint64_t y_begin = apr_iterator.current_gap_y_begin()+1;
            uint64_t y_end =apr_iterator.current_gap_y_end()+1;
            uint64_t index =apr_iterator.current_gap_index();

            std::copy(particleData.data.begin() + index,particleData.data.begin() + index + (y_end - y_begin) + 1,temp_vec.mesh.begin() + mesh_offset + y_begin );


        }

    }

    if(level > apr_iterator.level_min()) {
        const int y_num = apr_iterator.spatial_index_y_max(level);

        //
        //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
        //

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
        for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

            for (apr_iterator.set_new_lzx(level - 1, z / 2, x / 2);
                 apr_iterator.global_index() < apr_iterator.particles_zx_end(level - 1, z / 2,
                                                                             x /
                                                                             2); apr_iterator.set_iterator_to_particle_next_particle()) {

                int y_m = std::min(2 * apr_iterator.y() + 2,y_num);

                temp_vec.at(2 * apr_iterator.y()+1, x + 1, z % 3) = particleData[apr_iterator];

                temp_vec.at(y_m, x + 1, z % 3) = particleData[apr_iterator];


            }

        }
    }

    if(level < apr_iterator.level_max()) {

        //
        //  This is an interpolating from higher resolution particles to lower, if it was full this would be the simply average
        //  of the 8 children particle cells.
        //
        //  However, there are not always 8 children nodes, therefore we need to keep track of the number of children.
        //

        const uint64_t z_num_us =  apr_iterator.spatial_index_z_max(level+1);
        const uint64_t x_num_us =  apr_iterator.spatial_index_x_max(level+1);

        int x_ds=0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_ds) firstprivate(apr_iterator)
#endif
        for (x_ds = 0; x_ds < apr.spatial_index_x_max(level); ++x_ds) {

            std::vector<uint8_t> curr_counter;
            curr_counter.resize(y_num_m,0);

            for (int i = 0; i < 2; ++i) {

                int x = 2*x_ds + i;

                if ((x) < x_num_us) {

                    for (apr_iterator.set_new_lzx(level + 1, 2 * z, x);
                         apr_iterator.global_index() < apr_iterator.particles_zx_end(level + 1, 2 * z,
                                                                                     x); apr_iterator.set_iterator_to_particle_next_particle()) {

                        temp_vec.at(apr_iterator.y() / 2 + 1, x / 2 + 1, z % 3) =
                                (1.0f * curr_counter[apr_iterator.y() / 2] *
                                 temp_vec.at(apr_iterator.y() / 2 + 1, x / 2 + 1, z % 3) +
                                 particleData[apr_iterator]) /
                                (1.0f * (curr_counter[apr_iterator.y() / 2]) + 1.0f);

                        curr_counter[apr_iterator.y() / 2]++;
                    }
                    if ((2 * z + 1) < z_num_us) {
                        for (apr_iterator.set_new_lzx(level + 1, 2 * z + 1, x);
                             apr_iterator.global_index() < apr_iterator.particles_zx_end(level + 1, 2 * z + 1,
                                                                                         x); apr_iterator.set_iterator_to_particle_next_particle()) {

                            temp_vec.at(apr_iterator.y() / 2 + 1, x / 2 + 1, z % 3) =
                                    (1.0f * curr_counter[apr_iterator.y() / 2] *
                                     temp_vec.at(apr_iterator.y() / 2 + 1, x / 2 + 1, z % 3) +
                                     particleData[apr_iterator]) /
                                    (1.0f * curr_counter[apr_iterator.y() / 2] +
                                     1.0f);
                            curr_counter[apr_iterator.y() / 2]++;

                        }
                    }
                }
            }

        }

    }


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

    ///////////////////////////
    ///
    /// Serial Neighbour Iteration (Only Von Neumann (Face) neighbours)
    ///
    /////////////////////////////////

    APRIterator<uint16_t> neighbour_iterator(apr);
    APRIterator<uint16_t> apr_iterator(apr);

    int num_rep = 1;


    //Basic serial iteration over all particles
    uint64_t particle_number;
    //Basic serial iteration over all particles


    ExtraPartCellData<uint16_t> y_row;
    ExtraPartCellData<uint64_t> global_index_row_begin;

    y_row.data.resize(apr_iterator.level_max() + 1);
    global_index_row_begin.data.resize(apr_iterator.level_max() + 1);

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

        unsigned int z = 0;
        unsigned int x = 0;

        const int y_num = apr_iterator.spatial_index_y_max(level);
        const int x_num = apr_iterator.spatial_index_x_max(level);
        const int z_num = apr_iterator.spatial_index_z_max(level);

        y_row.data[level].resize(x_num*z_num);
        global_index_row_begin.data[level].resize(x_num*z_num);

        for (z = 0; z < apr.spatial_index_z_max(level); ++z) {


            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                if(apr_iterator.set_new_lzx(level, z, x)<UINT64_MAX){
                    global_index_row_begin.data[level][apr_iterator.z()*x_num + apr_iterator.x()].push_back(apr_iterator.global_index());
                }


                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.particles_zx_end(level, z,
                                                                                 x); apr_iterator.set_iterator_to_particle_next_particle()) {

                    y_row.data[level][apr_iterator.z()*x_num + apr_iterator.x()].push_back(apr_iterator.y());


                }
            }
        }
    }











                    //check the result

    ExtraParticleData<float> part_sum_dense2(apr);

    timer.start_timer("Dense neighbour access");

    for (int j = 0; j < num_rep; ++j) {

        for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

            unsigned int z = 0;
            unsigned int x = 0;

            const int y_num = apr_iterator.spatial_index_y_max(level);
            const int x_num = apr_iterator.spatial_index_x_max(level);
            const int z_num = apr_iterator.spatial_index_z_max(level);

            MeshData<float> temp_vec;
            temp_vec.init(apr_iterator.spatial_index_y_max(level) + 2, apr_iterator.spatial_index_x_max(level) + 2, 3,
                          0); //padded boundaries

            z = 0;

            //initial condition
            update_dense_array(level, z, apr, apr_iterator, temp_vec,apr.particles_intensities);

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - 1)) {
                    //update the next z plane for the access
                    update_dense_array(level, z + 1, apr, apr_iterator, temp_vec,apr.particles_intensities);
                } else {
                    // need to set (z+1)%3 to zero, zero boundary condition

                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + 1) % 3);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                        std::fill(temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num ,
                                  temp_vec.mesh.begin() + index + (x + 2) * temp_vec.y_num , 0);
                    }
                }


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    uint64_t global_index;
                    if(global_index_row_begin.data[level][x_num * z + x].size() > 0) {
                        global_index = global_index_row_begin.data[level][x_num * z + x][0];
                    }
                    for (int m = 0; m < y_row.data[level][x_num*z + x].size(); ++m) {


                        float neigh_sum = 0;
                        float counter = 0;

                        const int k = y_row.data[level][x_num*z + x][m] + 1; // offset to allow for boundary padding
                        const int i = x + 1;

                        for (int l = -1; l < 2; ++l) {
                            for (int q = -1; q < 2; ++q) {
                                for (int w = -1; w < 2; ++w) {
                                    neigh_sum += temp_vec.at(k+w, i+q, (z+3+w)%3);
                                }
                            }
                        }


                        part_sum_dense2.data[global_index] = temp_vec.at(k, i, z%3);

                        global_index++;

                    }
                }


            }
        }
    }


    timer.stop_timer();



    bool success = true;
    uint64_t f_c=0;


    //Basic serial iteration over all particles
    for (particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        if(part_sum_dense2.data[particle_number]!=apr.particles_intensities.data[particle_number]){

            success = false;
            f_c++;
        }

    }

    if(success){
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << " " << f_c <<  std::endl;
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
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
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

    return result;

}


