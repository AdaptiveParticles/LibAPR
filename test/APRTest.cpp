//
// Created by cheesema on 21.01.18.
//

#include <gtest/gtest.h>
#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRConverter.hpp"
#include <utility>
#include <cmath>

struct TestData{

    APR<uint16_t> apr;
    PixelData<uint16_t> img_level;
    PixelData<uint16_t> img_type;
    PixelData<uint16_t> img_original;
    PixelData<uint16_t> img_pc;
    PixelData<uint16_t> img_x;
    PixelData<uint16_t> img_y;
    PixelData<uint16_t> img_z;

    std::string filename;
    std::string output_name;

};

class CreateAPRTest : public ::testing::Test {
public:

    TestData test_data;

protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

};

class CreateSmallSphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};


class Create210SphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

bool check_neighbours(APR<uint16_t>& apr,APRIterator<uint16_t>& current,APRIterator<uint16_t>& neigh){


    bool success = true;

    if (std::abs((float)neigh.level() - (float)current.level()) > 1.0f) {
        success = false;
    }

    float delta_x = current.x_global() - neigh.x_global();
    float delta_y = current.y_global() - neigh.y_global();
    float delta_z = current.z_global() - neigh.z_global();

    float resolution_max = 1.11*(0.5*pow(2,current.level_max()-current.level()) + 0.5*pow(2,neigh.level_max()-neigh.level()));

    float distance = sqrt(pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2));

    if(distance > resolution_max){
        success = false;
    }

    return success;
}
bool check_neighbour_out_of_bounds(APRIterator<uint16_t>& current,uint8_t face){


    uint64_t num_neigh = current.number_neighbours_in_direction(face);

    if(num_neigh ==0){
        ParticleCell neigh = current.get_neigh_particle_cell();

        if( (neigh.x >= current.spatial_index_x_max(neigh.level) ) | (neigh.y >= current.spatial_index_y_max(neigh.level) ) | (neigh.z >= current.spatial_index_z_max(neigh.level) )  ){
            return true;
        } else {
            return false;
        }
    }

    return true;
}

bool test_apr_input_output(TestData& test_data){

    bool success = true;



    APRIterator<uint16_t> apr_iterator(test_data.apr);

    std::string save_loc = "";
    std::string file_name = "read_write_test";

    //write the APR
    test_data.apr.write_apr(save_loc,file_name);

    APR<uint16_t> apr_read;

    apr_read.read_apr(save_loc + file_name + "_apr.h5");
    APRIterator<uint16_t> apr_iterator_read(apr_read);

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                apr_iterator_read.set_new_lzx(level, z, x);
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    //check the functionality
                    if (test_data.apr.particles_intensities[apr_iterator] !=
                        apr_read.particles_intensities[apr_iterator_read]) {
                        success = false;
                    }

                    if (apr_iterator.level() != apr_iterator_read.level()) {
                        success = false;
                    }

                    if (apr_iterator.x() != apr_iterator_read.x()) {
                        success = false;
                    }

                    if (apr_iterator.y() != apr_iterator_read.y()) {
                        success = false;
                    }

                    if (apr_iterator.z() != apr_iterator_read.z()) {
                        success = false;
                    }

                    if (apr_iterator.type() != apr_iterator_read.type()) {
                        success = false;
                    }
                    if(apr_iterator_read.global_index() < apr_iterator_read.end_index) {
                        apr_iterator_read.set_iterator_to_particle_next_particle();
                    }

                }
            }
        }

    }


    //
    // Now check the Extra Part Cell Data
    //

    APRIterator<uint16_t> neighbour_iterator(apr_read);
    APRIterator<uint16_t> apr_iterator_read2(apr_read);

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                apr_iterator_read2.set_new_lzx(level, z, x);
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    //counter++;
                    //apr_iterator_read2.set_iterator_to_particle_next_particle();


                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        apr_iterator_read2.find_neighbours_in_direction(direction);

                        success = check_neighbour_out_of_bounds(apr_iterator_read2, direction);

                        for (int index = 0;
                             index < apr_iterator_read2.number_neighbours_in_direction(direction); ++index) {

                            // on each face, there can be 0-4 neighbours accessed by index
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator_read2, direction, index)) {
                                //will return true if there is a neighbour defined
                                uint16_t apr_intensity = test_data.apr.particles_intensities[neighbour_iterator];
                                uint16_t check_intensity = test_data.img_pc(neighbour_iterator.y_nearest_pixel(),
                                                                            neighbour_iterator.x_nearest_pixel(),
                                                                            neighbour_iterator.z_nearest_pixel());

                                if (check_intensity != apr_intensity) {
                                    success = false;
                                }

                                if (!check_neighbours(apr_read, apr_iterator_read2, neighbour_iterator)) {
                                    success = false;
                                }
                            }
                        }
                    }

                    if(apr_iterator_read2.global_index() < apr_iterator_read2.end_index) {
                        apr_iterator_read2.set_iterator_to_particle_next_particle();
                    }
                }
            }
        }
    }


    ExtraParticleData<float> extra_data(test_data.apr);

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

                for (apr_iterator_read.set_new_lzx(level, z, x);
                     apr_iterator_read.global_index() < apr_iterator_read.end_index;
                     apr_iterator_read.set_iterator_to_particle_next_particle()) {

                    extra_data[apr_iterator_read] = apr_iterator_read.level();
                }
            }
        }

    }

    //write one of the above results to file
    test_data.apr.write_particles_only(save_loc,"example_output",extra_data);

    std::string extra_file_name = save_loc + "example_output" + "_apr_extra_parts.h5";

    ExtraParticleData<float> extra_data_read;

    //you need the same apr used to write it to load it (doesn't save location data)
    test_data.apr.read_parts_only(extra_file_name,extra_data_read);

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

                for (apr_iterator_read.set_new_lzx(level, z, x);
                     apr_iterator_read.global_index() < apr_iterator_read.end_index;
                     apr_iterator_read.set_iterator_to_particle_next_particle()) {

                    extra_data[apr_iterator_read] = apr_iterator_read.level();
                    if ((extra_data[apr_iterator_read]) != (extra_data_read[apr_iterator_read])) {

                        success = false;
                    }
                }
            }
        }
    }


    return success;
}



bool test_apr_neighbour_access(TestData& test_data){

    bool success = true;

    APRIterator<uint16_t> neighbour_iterator(test_data.apr);
    APRIterator<uint16_t> apr_iterator(test_data.apr);

    ExtraParticleData<uint16_t> x_p(test_data.apr);
    ExtraParticleData<uint16_t> y_p(test_data.apr);
    ExtraParticleData<uint16_t> z_p(test_data.apr);

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;


        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    x_p[apr_iterator] = apr_iterator.x();
                    y_p[apr_iterator] = apr_iterator.y();
                    z_p[apr_iterator] = apr_iterator.z();

                }
            }
        }
    }





    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;


        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        apr_iterator.find_neighbours_in_direction(direction);

                        success = check_neighbour_out_of_bounds(apr_iterator, direction);

                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                            // on each face, there can be 0-4 neighbours accessed by index
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //will return true if there is a neighbour defined
                                uint16_t apr_intensity = test_data.apr.particles_intensities[neighbour_iterator];
                                uint16_t check_intensity = test_data.img_pc(neighbour_iterator.y_nearest_pixel(),
                                                                            neighbour_iterator.x_nearest_pixel(),
                                                                            neighbour_iterator.z_nearest_pixel());

                                uint16_t x_n = x_p[neighbour_iterator];
                                uint16_t y_n = y_p[neighbour_iterator];
                                uint16_t z_n = z_p[neighbour_iterator];

                                if (check_intensity != apr_intensity) {
                                    success = false;
                                }


                                uint16_t apr_level = neighbour_iterator.level();
                                uint16_t check_level = test_data.img_level(neighbour_iterator.y_nearest_pixel(),
                                                                           neighbour_iterator.x_nearest_pixel(),
                                                                           neighbour_iterator.z_nearest_pixel());

                                if (check_level != apr_level) {
                                    success = false;
                                }


                                uint16_t apr_x = neighbour_iterator.x();
                                uint16_t check_x = test_data.img_x(neighbour_iterator.y_nearest_pixel(),
                                                                   neighbour_iterator.x_nearest_pixel(),
                                                                   neighbour_iterator.z_nearest_pixel());

                                if (check_x != apr_x) {
                                    success = false;
                                }

                                uint16_t apr_y = neighbour_iterator.y();
                                uint16_t check_y = test_data.img_y(neighbour_iterator.y_nearest_pixel(),
                                                                   neighbour_iterator.x_nearest_pixel(),
                                                                   neighbour_iterator.z_nearest_pixel());

                                if (check_y != apr_y) {
                                    success = false;
                                }

                                uint16_t apr_z = neighbour_iterator.z();
                                uint16_t check_z = test_data.img_z(neighbour_iterator.y_nearest_pixel(),
                                                                   neighbour_iterator.x_nearest_pixel(),
                                                                   neighbour_iterator.z_nearest_pixel());

                                if (check_z != apr_z) {
                                    success = false;
                                }

                                if (!check_neighbours(test_data.apr, apr_iterator, neighbour_iterator)) {
                                    success = false;
                                }

                            }

                        }

                    }
                }
            }
        }
    }

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator,neighbour_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        apr_iterator.find_neighbours_in_direction(direction);

                        success = check_neighbour_out_of_bounds(apr_iterator, direction);

                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                            // on each face, there can be 0-4 neighbours accessed by index
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //will return true if there is a neighbour defined
                                uint16_t apr_intensity = (test_data.apr.particles_intensities[neighbour_iterator]);
                                uint16_t check_intensity = test_data.img_pc(neighbour_iterator.y_nearest_pixel(),
                                                                            neighbour_iterator.x_nearest_pixel(),
                                                                            neighbour_iterator.z_nearest_pixel());

                                if (check_intensity != apr_intensity) {
                                    success = false;
                                }

                                uint16_t apr_level = neighbour_iterator.level();
                                uint16_t check_level = test_data.img_level(neighbour_iterator.y_nearest_pixel(),
                                                                           neighbour_iterator.x_nearest_pixel(),
                                                                           neighbour_iterator.z_nearest_pixel());

                                if (check_level != apr_level) {
                                    success = false;
                                }


                                uint16_t apr_x = neighbour_iterator.x();
                                uint16_t check_x = test_data.img_x(neighbour_iterator.y_nearest_pixel(),
                                                                   neighbour_iterator.x_nearest_pixel(),
                                                                   neighbour_iterator.z_nearest_pixel());

                                if (check_x != apr_x) {
                                    success = false;
                                }

                                uint16_t apr_y = neighbour_iterator.y();
                                uint16_t check_y = test_data.img_y(neighbour_iterator.y_nearest_pixel(),
                                                                   neighbour_iterator.x_nearest_pixel(),
                                                                   neighbour_iterator.z_nearest_pixel());

                                if (check_y != apr_y) {
                                    success = false;
                                }

                                uint16_t apr_z = neighbour_iterator.z();
                                uint16_t check_z = test_data.img_z(neighbour_iterator.y_nearest_pixel(),
                                                                   neighbour_iterator.x_nearest_pixel(),
                                                                   neighbour_iterator.z_nearest_pixel());

                                if (check_z != apr_z) {
                                    success = false;
                                }

                            }
                        }
                    }
                }
            }
        }
    }




    return success;


}



bool test_apr_iterate(TestData& test_data){
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator
    //

    bool success = true;

    APRIterator<uint16_t> apr_iterator(test_data.apr);
    uint64_t particle_number = 0;

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    uint16_t apr_intensity = (test_data.apr.particles_intensities[apr_iterator]);
                    uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(),
                                                                apr_iterator.x_nearest_pixel(),
                                                                apr_iterator.z_nearest_pixel());

                    if (check_intensity != apr_intensity) {
                        success = false;
                    }

                    uint16_t apr_level = apr_iterator.level();
                    uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(),
                                                               apr_iterator.x_nearest_pixel(),
                                                               apr_iterator.z_nearest_pixel());

                    if (check_level != apr_level) {
                        success = false;
                    }

                    uint16_t apr_type = apr_iterator.type();
                    uint16_t check_type = test_data.img_type(apr_iterator.y_nearest_pixel(),
                                                             apr_iterator.x_nearest_pixel(),
                                                             apr_iterator.z_nearest_pixel());


                    uint16_t apr_x = apr_iterator.x();
                    uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(), apr_iterator.x_nearest_pixel(),
                                                       apr_iterator.z_nearest_pixel());

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = apr_iterator.y();
                    uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(), apr_iterator.x_nearest_pixel(),
                                                       apr_iterator.z_nearest_pixel());

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = apr_iterator.z();
                    uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(), apr_iterator.x_nearest_pixel(),
                                                       apr_iterator.z_nearest_pixel());

                    if (check_z != apr_z) {
                        success = false;
                    }
                }
            }
        }

    }

    //Test parallel loop

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    uint16_t apr_intensity = (test_data.apr.particles_intensities[apr_iterator]);
                    uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(),
                                                                apr_iterator.x_nearest_pixel(),
                                                                apr_iterator.z_nearest_pixel());

                    if (check_intensity != apr_intensity) {
                        success = false;
                    }

                    uint16_t apr_level = apr_iterator.level();
                    uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(),
                                                               apr_iterator.x_nearest_pixel(),
                                                               apr_iterator.z_nearest_pixel());

                    if (check_level != apr_level) {
                        success = false;
                    }



                    uint16_t apr_x = apr_iterator.x();
                    uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(), apr_iterator.x_nearest_pixel(),
                                                       apr_iterator.z_nearest_pixel());

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = apr_iterator.y();
                    uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(), apr_iterator.x_nearest_pixel(),
                                                       apr_iterator.z_nearest_pixel());

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = apr_iterator.z();
                    uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(), apr_iterator.x_nearest_pixel(),
                                                       apr_iterator.z_nearest_pixel());

                    if (check_z != apr_z) {
                        success = false;
                    }
                }
            }
        }

    }

    uint64_t counter = 0;
    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    counter++;

                    if (apr_iterator.level() != level) {
                        //set all particles in calc_ex with an particle intensity greater then 100 to 0.
                        success = false;
                    }
                }
            }
        }
    }

    if(counter != apr_iterator.total_number_particles()){
        success = false;
    }



    return success;
}


bool test_apr_pipeline(TestData& test_data){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    bool success = true;

    //the apr datastructure
    APR<uint16_t> apr;

    APRConverter<uint16_t> apr_converter;

    //read in the command line options into the parameters file
    apr_converter.par.Ip_th = test_data.apr.parameters.Ip_th;
    apr_converter.par.rel_error = test_data.apr.parameters.rel_error;
    apr_converter.par.lambda = test_data.apr.parameters.lambda;
    apr_converter.par.mask_file = "";
    apr_converter.par.min_signal = -1;

    apr_converter.par.sigma_th_max = test_data.apr.parameters.sigma_th_max;
    apr_converter.par.sigma_th = test_data.apr.parameters.sigma_th;

    apr_converter.par.SNR_min = -1;

    //where things are
    apr_converter.par.input_image_name =test_data.filename;
    apr_converter.par.input_dir = "";
    apr_converter.par.name = test_data.output_name;
    apr_converter.par.output_dir = "";

    //Gets the APR
    if(apr_converter.get_apr(apr)){
        APRIterator<uint16_t> apr_iterator(apr);
        uint64_t particle_number = 0;
        std::cout << "NUM OF PARTICLES: " << apr_iterator.total_number_particles() << " vs " << test_data.apr.total_number_particles() << std::endl;
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            uint16_t apr_intensity = (apr.particles_intensities[apr_iterator]);
            uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

            if(check_intensity!=apr_intensity){
                success = false;
            }

            uint16_t apr_level = apr_iterator.level();
            uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

            if(check_level!=apr_level){
                success = false;
            }

            uint16_t apr_type = apr_iterator.type();
            uint16_t check_type = test_data.img_type(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

            if(check_type!=apr_type){
                success = false;
            }

            uint16_t apr_x = apr_iterator.x();
            uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

            if(check_x!=apr_x){
                success = false;
            }

            uint16_t apr_y = apr_iterator.y();
            uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

            if(check_y!=apr_y){
                success = false;
            }

            uint16_t apr_z = apr_iterator.z();
            uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

            if(check_z!=apr_z){
                success = false;
            }

        }

    } else {

        success = false;
    }


    return success;
}

std::string get_source_directory_apr(){
    // returns path to the directory where utils.cpp is stored

    std::string tests_directory = std::string(__FILE__);
    tests_directory = tests_directory.substr(0, tests_directory.find_last_of("\\/") + 1);

    return tests_directory;
}


void CreateSmallSphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_apr.h5";
    test_data.apr.read_apr(file_name);

    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_type.tif";
    test_data.img_type = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_120/sphere_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_120/sphere_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
    test_data.output_name = "sphere_small";
}

void Create210SphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_apr.h5";
    test_data.apr.read_apr(file_name);

    file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_type.tif";
    test_data.img_type = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name);
    file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_210/sphere_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_210/sphere_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_210/sphere_original.tif";
    test_data.output_name = "sphere_210";
}

TEST_F(CreateSmallSphereTest, APR_ITERATION) {

//test iteration
ASSERT_TRUE(test_apr_iterate(test_data));

}

TEST_F(CreateSmallSphereTest, APR_NEIGHBOUR_ACCESS) {

//test iteration
ASSERT_TRUE(test_apr_neighbour_access(test_data));

}

TEST_F(CreateSmallSphereTest, APR_INPUT_OUTPUT) {

//test iteration
    ASSERT_TRUE(test_apr_input_output(test_data));

}

TEST_F(CreateSmallSphereTest, APR_PIPELINE) {

//test iteration
//    ASSERT_TRUE(test_apr_pipeline(test_data));

}

TEST_F(Create210SphereTest, APR_ITERATION) {

//test iteration
    ASSERT_TRUE(test_apr_iterate(test_data));

}

TEST_F(Create210SphereTest, APR_NEIGHBOUR_ACCESS) {

//test iteration
    ASSERT_TRUE(test_apr_neighbour_access(test_data));

}

TEST_F(Create210SphereTest, APR_INPUT_OUTPUT) {

//test iteration
    ASSERT_TRUE(test_apr_input_output(test_data));

}

TEST_F(Create210SphereTest, APR_PIPELINE) {

//test iteration
// TODO: FIXME please!
//    ASSERT_TRUE(test_apr_pipeline(test_data));

}


int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
