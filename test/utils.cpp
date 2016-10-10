////////////////////////
//
//  Mateusz Susik 2016
//
//  Utility functions for the tests
//
////////////////////////

#include "utils.h"

#include <algorithm>
#include <functional>
#include <iostream>

#include "../src/io/readimage.h"
#include "../src/io/write_parts.h"
#include "../src/io/read_parts.h"

bool compare_two_images(const Mesh_data<uint16_t>& in_memory, std::string filename) {

    /* Returns true iff two images are the same with tolerance of 1 per pixel. */

    Mesh_data<uint16_t > input_image;

    load_image_tiff(input_image, filename);

    auto it2 = input_image.mesh.begin();
    for(auto it1 = in_memory.mesh.begin(); it1 != in_memory.mesh.end(); it1++, it2++)
    {
        // 60000 is the threshold introduced in variance computations. When a value reaches zero when mean is
        // computed, it is changed to 60000 afterwards. It is caused by the set of parameters in test case.
        if(std::abs(*it1 - *it2) > 1 && std::abs(*it1 - *it2) != 60000) {
            std::cout << std::distance(it1, in_memory.mesh.begin()) << " " << *it1 << " " << *it2 << std::endl;
            return false;
        }
    }
    return true;

}

bool compare_two_ks(const Particle_map<float>& in_memory, std::string filename) {

    for (int k = in_memory.k_min;k <= in_memory.k_max;k++) {

        Mesh_data<uint8_t > to_compare;

        // in_memory.layers[k]
        load_image_tiff(to_compare, filename + "_" + std::to_string(k) + ".tif");

        auto it2 = to_compare.mesh.begin();
        for(auto it1 = in_memory.layers[k].mesh.begin();
                 it1 != in_memory.layers[k].mesh.end(); it1++, it2++)
        {
            if(*it1 != *it2) {
                std::cout << std::distance(it1, in_memory.layers[k].mesh.begin()) <<
                             " " << (int)*it1 << " " << (int)*it2 << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool compare_part_rep_with_particle_map(const Particle_map<float>& in_memory, std::string filename) {
    Part_rep p_rep;
    read_parts_from_full_hdf5(p_rep, filename);


    // Check

    for(int i = 0; i < p_rep.status.data.size(); i++) {
        //count Intensity as well

        if(true) {

            if (p_rep.status.data[i] != EMPTY) {


                int x = p_rep.pl_map.cells[i].x;
                int y = p_rep.pl_map.cells[i].y;
                int z = p_rep.pl_map.cells[i].z;
                int k = p_rep.pl_map.cells[i].k;

                int x_num = in_memory.layers[k].x_num;
                int y_num = in_memory.layers[k].y_num;

                if (x <= p_rep.org_dims[1] / 2 &&
                    y <= p_rep.org_dims[0] / 2 &&
                    z <= p_rep.org_dims[2] / 2) {
                    // add if it is in domain
                    if (p_rep.status.data[i] == 2) {

                        if(in_memory.layers[k].mesh[(z-1) * x_num * y_num + (x-1) * y_num + y - 1] != TAKENSTATUS)
                        {
                            std::cout << "Different status: INITIALIZED" << std::endl;
                            return false;
                        }




                    } else if (p_rep.status.data[i] >= 4) {
                        //check if has the same status

                        if(p_rep.status.data[i] == 4 &&
                            in_memory.layers[k].mesh[(z-1) * x_num * y_num + (x-1) * y_num + y - 1] != NEIGHBOURSTATUS)
                        {
                            std::cout << "Different status: NEIGHBOUR " << std::endl;
                            return false;
                        }

                        if(p_rep.status.data[i] == 5 &&
                           in_memory.layers[k].mesh[(z-1) * x_num * y_num + (x-1) * y_num + y - 1] != SLOPESTATUS)
                        {
                            std::cout << "Different status: SLOPE" << (int)z << " " <<
                                         (int)x << " " << (int)y << std::endl;
                            //return false;
                        }

                    }

                }
            }
        }
    }


    return true;
}

Mesh_data<uint16_t> create_random_test_example(unsigned int size_y, unsigned int size_x,
                                                unsigned int size_z, unsigned int seed) {
    // creates the input image of a given size with given seed
    // uses ranlux48 random number generator
    // the seed used in 2016 for generation was 5489u

    std::ranlux48 generator(seed);
    std::normal_distribution<float> distribution(1000, 250);

    Mesh_data<uint16_t> test_example(size_y, size_x, size_z);

    std::generate(test_example.mesh.begin(), test_example.mesh.end(),
                  // partial application of generator and distribution to get_random_number function
                  std::bind(get_random_number, generator, distribution));

    return test_example;

}

Mesh_data<uint16_t> generate_random_ktest_example(unsigned int size_y, unsigned int size_x,
                                                  unsigned int size_z, unsigned int seed,
                                                  float mean_fraction, float sd_fraction) {

    // creates the input image of a given size with given seed
    // the image should be used as a source of benchmarking for the get_k step
    // dx, dy and dz should all be set to 1, rel_error to 1000
    // the seed used in 2016 for generation was 5489u

    std::ranlux48 generator(seed);

    int max_dim = std::max(size_x, std::max(size_y, size_z));
    float k_max = ceil(M_LOG2E*log(max_dim)) - 1;

    std::normal_distribution<float> distribution(k_max * mean_fraction, k_max * sd_fraction);

    Mesh_data<uint16_t> test_example(size_y, size_x, size_z);

#pragma omp parallel for default(shared)
    for(int i = 0; i < test_example.mesh.size(); i++){
        test_example.mesh[i] = get_random_number_k(generator, distribution, k_max);
    }

    std::generate(test_example.mesh.begin(), test_example.mesh.end(),
                  // partial application of generator and distribution to get_random_number function
                  std::bind(get_random_number_k, generator, distribution, k_max));

    return test_example;

}

uint16_t get_random_number(std::ranlux48& generator, std::normal_distribution<float>& distribution){

    float val = distribution(generator);
    //there should be no values below zero.
    return val < 0 ? 1 : val;

}

uint16_t get_random_number_k(std::ranlux48& generator,
                             std::normal_distribution<float>& distribution, float k_max){

    float val = distribution(generator);
    //there should be no values below zero.
    return std::max(K_BENCHMARK_REL_ERROR * pow(2, val - k_max), 0.01);

}

std::string get_source_directory(){
    // returns path to the directory where utils.cpp is stored

    std::string tests_directory = std::string(__FILE__);
    tests_directory = tests_directory.substr(0, tests_directory.find_last_of("\\/") + 1);

    return tests_directory;
}