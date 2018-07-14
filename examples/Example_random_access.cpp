//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
const char* usage = R"(
Example setting the APR iterator using random access

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_random_accesss -i input_apr_hdf5 -d input_directory

Note: There is no output, this file is best utilized by looking at the source code for example (test/Examples/Example_random_access.cpp) of how to code different
random access strategies on the APR.

)";


#include <algorithm>
#include <iostream>

#include "Example_random_access.hpp"



int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR <uint16_t> apr;

    timer.start_timer("full read");
    //read file
    apr.read_apr(file_name);

    timer.stop_timer();

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end() - 3, name.end());

//    APRIterator<uint16_t> apr_iterator(apr);
//
//    ///////////////////////
//    ///
//    /// Set the iterator using random access by particle cell spatial index (x,y,z) and particle cell level
//    ///
//    ////////////////////////
//
//    std::cout << "Search for a Particle Cell that may not exist at (x,y,z,l) = (10,10,10,4)" << std::endl;
//    std::cout << "--------------------" << std::endl;
//
//    ParticleCell random_particle_cell;
//    random_particle_cell.x = 1;
//    random_particle_cell.y = 2;
//    random_particle_cell.z = 5;
//    random_particle_cell.level = apr.level_max() - 1;
//
//    bool found = apr_iterator.set_iterator_by_particle_cell(random_particle_cell);
//
//    if(!found){
//        std::cout << "Particle Cell doesn't exist!" << std::endl;
//    } else {
//        std::cout << "Particle Cell exists with global index (particle number): " << random_particle_cell.global_index << " and has intensity value: " << apr.particles_intensities[apr_iterator] <<  std::endl;
//    }
//
//
//    ///////////////////////
//    ///
//    /// Set the iterator using random access by using a global co-ordinate (in original pixels), and setting the iterator, to the Particle Cell that contains the point in its spatial domain.
//    ///
//    ////////////////////////
//
//    srand (time(NULL));
//
//    float x = apr.orginal_dimensions(1)*((rand() % 10000)/10000.0f);
//    float y = apr.orginal_dimensions(0)*((rand() % 10000)/10000.0f);
//    float z = apr.orginal_dimensions(2)*((rand() % 10000)/10000.0f);
//
//    found = apr_iterator.set_iterator_by_global_coordinate(x,y,z);
//
//    std::cout << std::endl;
//    std::cout << "Searching for Particle Cell thats spatial domain contains (x,y,z)=(" << x << "," << y << "," << z << ") " << std::endl;
//    std::cout << "--------------------" << std::endl;
//
//    if(!found){
//        std::cout << "out of bounds" << std::endl;
//    } else {
//        std::cout << "Particle Cell found is at level: " << apr_iterator.level() << " with x: " << apr_iterator.x() << " y: " << apr_iterator.y() << " z: " << apr_iterator.z() << std::endl;
//        std::cout << " with global index: " << apr_iterator.global_index() << " and intensity " << apr.particles_intensities[apr_iterator] << std::endl;
//    }
//
//
//    ///
//    //  Testing of the random access against some alternative strategies
//    //
//    //
//    //
//
//    ExtraPartCellData<uint16_t> y_b;
//
//    y_b.initialize_structure_parts_empty(apr);
//
//
//    ExtraPartCellData<uint16_t> hint;
//
//    hint.initialize_structure_parts_empty(apr);
//
//
//
//    timer.start_timer("APR serial iterator loop");
//
//
//    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
//        int z = 0;
//        int x = 0;
//
//        y_b.data[level].resize(apr.apr_access.gap_map.x_num[level]*apr.apr_access.gap_map.z_num[level]);
//        hint.data[level].resize(apr.apr_access.gap_map.x_num[level]*apr.apr_access.gap_map.z_num[level]);
//
//        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
//            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
//
//                size_t offset = apr.apr_access.gap_map.x_num[level]*z + x;
//
//                if(level == apr.level_max()){
//                    offset = apr.apr_access.gap_map.x_num[level]*(z/2) + (x/2);
//                }
//
//                if(apr.apr_access.gap_map.data[level][offset].size()>0){
//                    for(auto it = apr.apr_access.gap_map.data[level][offset][0].map.begin(); it != apr.apr_access.gap_map.data[level][offset][0].map.end();++it){
//                        y_b.data[level][offset].push_back(it->first);
//                    }
//
//                }
//            }
//        }
//    }
//
//    timer.stop_timer();
//
//    unsigned int rep = 4;
//
//    timer.start_timer("loop map");
//    for (int k = 0; k < rep; ++k) {
//
//        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
//            int z = 0;
//            int x = 0;
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
//#endif
//            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
//                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
//
//                    size_t offset = apr.apr_access.gap_map.x_num[level] * z + x;
//
//                    if (level == apr.level_max()) {
//                        offset = apr.apr_access.gap_map.x_num[level] * (z / 2) + (x / 2);
//                    }
//
//                    size_t max_y = apr.apr_access.y_num[level];
//                    ParticleCell pc;
//
//                    pc.x = x;
//                    pc.z = z;
//                    pc.level = level;
//                    pc.pc_offset = offset;
//
//                    for (int i = 0; i < ceil(0.2 * max_y); ++i) {
//                        pc.y = (uint16_t) (rand() % max_y);
//
//                        bool found = apr_iterator.set_iterator_by_particle_cell_test(pc);
//                    }
//                }
//            }
//        }
//    }
//
//    timer.stop_timer();
//
//    float t1 = timer.timings.back();
//
//
//    timer.start_timer("loop map");
//    for (int j = 0; j < rep; ++j) {
//        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
//            int z = 0;
//            int x = 0;
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
//#endif
//            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
//                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
//
//                    size_t offset = apr.apr_access.gap_map.x_num[level] * z + x;
//
//                    if (level == apr.level_max()) {
//                        offset = apr.apr_access.gap_map.x_num[level] * (z / 2) + (x / 2);
//                    }
//
//                    size_t max_y = apr.apr_access.y_num[level];
//
//                    auto curr_g = y_b.data[level][offset];
//
//                    for (int i = 0; i < ceil(0.2 * max_y); ++i) {
//                        uint16_t y_val = (uint16_t) (rand() % max_y);
//                        auto it = curr_g.begin();
//                        auto it_end = curr_g.rbegin();
//                        uint16_t counter = 0;
//
//                        if(it!= curr_g.end()) {
//
//                            if ((y_val >= *it) && ((y_val <= *it_end))) {
//
//                                if((y_val-*it) > (*it_end - y_val)) {
//
//                                    while ((it_end != curr_g.rend()) && (*it_end > y_val)) {
//                                        --it_end;
//                                        counter++;
//                                    }
//                                } else {
//                                    while ((it != curr_g.end()) && (*it < y_val)) {
//                                        ++it;
//                                        counter++;
//                                    }
//
//                                }
//                            }
//                        }
//
//                        uint16_t found = *it;
//                    }
//                }
//            }
//
//        }
//    }
//
//    timer.stop_timer();
//
//    float t2 = timer.timings.back();
//
//    float normt1 = t1/(1.0f*apr.total_number_particles());
//    float normt2 = t2/(1.0f*apr.total_number_particles());
//
//    std::cout << "Normed time: " << normt1 << std::endl;
//    std::cout << "Normed time2: " << normt2 << std::endl;
//    std::cout << "Ratio: " << t1/t2 << std::endl;




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
        std::cerr << "Usage: \"Example_random_access -i input_apr_file -d directory\"" << std::endl;
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
