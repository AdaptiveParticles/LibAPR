//
// Created by cheesema on 21.01.18.
//

#include <gtest/gtest.h>
#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRConverter.hpp"
#include <utility>
#include <cmath>
#include "TestTools.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include "io/APRWriter.hpp"

#include "io/APRFile.hpp"

struct TestData{

    APR apr;
    PixelData<uint16_t> img_level;
    PixelData<uint16_t> img_type;
    PixelData<uint16_t> img_original;
    PixelData<uint16_t> img_pc;
    PixelData<uint16_t> img_x;
    PixelData<uint16_t> img_y;
    PixelData<uint16_t> img_z;

    ParticleData<uint16_t> particles_intensities;

    std::string filename;
    std::string apr_filename;
    std::string output_name;
    std::string output_dir;

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

class CreateGTSmallTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmall2DTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmall1DTest : public CreateAPRTest
{
public:
    void SetUp() override;
};


bool check_neighbours(APR& apr,APRIterator &current, APRIterator &neigh){


    bool success = true;

    if (std::abs((float)neigh.level() - (float)current.level()) > 1.0f) {
        success = false;
    }

    float delta_x = current.x_global(current.level(),current.x()) - neigh.x_global(neigh.level(),neigh.x());
    float delta_y = current.y_global(current.level(),current.y()) - neigh.y_global(neigh.level(),neigh.y());
    float delta_z = current.z_global(current.level(),current.z()) - neigh.z_global(neigh.level(),neigh.z());

    float resolution_max = 1.11*(0.5*pow(2,current.level_max()-current.level()) + 0.5*pow(2,neigh.level_max()-neigh.level()));

    float distance = sqrt(pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2));

    if(distance > resolution_max){
        success = false;
    }

    return success;
}
bool check_neighbour_out_of_bounds(APRIterator &current,uint8_t face){


    uint64_t num_neigh = current.number_neighbours_in_direction(face);

    if(num_neigh ==0){
        ParticleCell neigh = current.get_neigh_particle_cell();

        if( (neigh.x >= current.x_num(neigh.level) ) | (neigh.y >= current.y_num(neigh.level) ) | (neigh.z >= current.z_num(neigh.level) )  ){
            return true;
        } else {
            return false;
        }
    }

    return true;
}

bool test_apr_tree(TestData& test_data) {

    bool success = true;

    std::string save_loc = "";
    std::string file_name = "read_write_test";


    test_data.apr.init_tree();

    ParticleData<float> tree_data;

    APRTreeIterator apr_tree_iterator = test_data.apr.tree_iterator();

//    aprTree.fill_tree_mean(test_data.apr,aprTree,test_data.particles_intensities,tree_data);
//
//    aprTree.fill_tree_mean_downsample(test_data.particles_intensities);

    APRTreeNumerics::fill_tree_mean(test_data.apr,test_data.particles_intensities,tree_data);



    //generate tree test data
    PixelData<float> pc_image;
    APRReconstruction::interp_img(test_data.apr,pc_image,test_data.particles_intensities);


    std::vector<PixelData<float>> downsampled_img;
    //Down-sample the image for particle intensity estimation
    downsamplePyrmaid(pc_image, downsampled_img, test_data.apr.level_max(), test_data.apr.level_min()-1);

    for (unsigned int level = (apr_tree_iterator.level_max()); level >= apr_tree_iterator.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_tree_iterator.z_num(level); z++) {
            for (x = 0; x < apr_tree_iterator.x_num(level); ++x) {
                for (apr_tree_iterator.set_new_lzx(level, z, x); apr_tree_iterator < apr_tree_iterator.end();
                     apr_tree_iterator.set_iterator_to_particle_next_particle()) {

                    uint16_t current_int = (uint16_t)std::round(downsampled_img[apr_tree_iterator.level()].at(apr_tree_iterator.y(),apr_tree_iterator.x(),apr_tree_iterator.z()));
                    //uint16_t parts_int = aprTree.particles_ds_tree[apr_tree_iterator];
                    uint16_t parts2 = (uint16_t)std::round(tree_data[apr_tree_iterator]);

                    // uint16_t y = apr_tree_iterator.y();

                    if(abs(parts2 - current_int) > 1){
                        success = false;
                    }

                }
            }
        }
    }

    //Also check the sparse data-structure generated tree

    ParticleData<float> treedata_2;


    APRTreeNumerics::fill_tree_mean(test_data.apr,test_data.particles_intensities,treedata_2);

//    tree2.fill_tree_mean(test_data.apr,tree2,test_data.particles_intensities,treedata_2);
    auto apr_tree_iterator_s = test_data.apr.tree_iterator();

    for (unsigned int level = (apr_tree_iterator_s.level_max()); level >= apr_tree_iterator_s.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_tree_iterator_s.z_num(level); z++) {
            for (x = 0; x < apr_tree_iterator_s.x_num(level); ++x) {
                for (apr_tree_iterator_s.set_new_lzx(level, z, x); apr_tree_iterator_s < apr_tree_iterator_s.end();
                     apr_tree_iterator_s.set_iterator_to_particle_next_particle()) {

                    uint16_t current_int = (uint16_t)std::round(downsampled_img[apr_tree_iterator_s.level()].at(apr_tree_iterator_s.y(),apr_tree_iterator_s.x(),apr_tree_iterator_s.z()));
                    //uint16_t parts_int = aprTree.particles_ds_tree[apr_tree_iterator];
                    uint16_t parts2 = (uint16_t)std::round(treedata_2[apr_tree_iterator_s]);

                    // uint16_t y = apr_tree_iterator.y();

                    if(abs(parts2 - current_int) > 1){
                        success = false;
                    }

                }
            }
        }
    }


    APRTreeIterator neigh_tree_iterator = test_data.apr.tree_iterator();


    for (unsigned int level = apr_tree_iterator.level_min(); level <= apr_tree_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_tree_iterator.z_num(level); z++) {
            for (x = 0; x < apr_tree_iterator.x_num(level); ++x) {
                for (apr_tree_iterator.set_new_lzx(level, z, x); apr_tree_iterator < apr_tree_iterator.end();
                     apr_tree_iterator.set_iterator_to_particle_next_particle()) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        apr_tree_iterator.find_neighbours_same_level(direction);
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        for (int index = 0; index < apr_tree_iterator.number_neighbours_in_direction(direction); ++index) {

                            if (neigh_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, index)) {
                                //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)

                                uint16_t current_int = (uint16_t)std::round(downsampled_img[neigh_tree_iterator.level()].at(neigh_tree_iterator.y(),neigh_tree_iterator.x(),neigh_tree_iterator.z()));
                                //uint16_t parts_int = aprTree.particles_ds_tree[apr_tree_iterator];
                                uint16_t parts2 = (uint16_t)std::round(tree_data[neigh_tree_iterator]);

                                //uint16_t y = apr_tree_iterator.y();

                                if(abs(parts2 - current_int) > 1){
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

bool test_apr_file(TestData& test_data){


    //Test Basic IO
    std::string file_name = "read_write_test.apr";

    //First write a file
    APRFile writeFile;
    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(test_data.apr);

    writeFile.write_particles(test_data.apr,"parts",test_data.particles_intensities);

    ParticleData<float> parts2;
    parts2.init(test_data.apr.total_number_particles());

    auto apr_it = test_data.apr.iterator();

    for (int i = 0; i < apr_it.total_number_particles(); ++i) {
        parts2[i] = test_data.particles_intensities[i]*3 - 1;
    }

    writeFile.write_particles(test_data.apr,"parts2",parts2);

    writeFile.close();

    //First read a file
    APRFile readFile;

    bool success = true;

    readFile.open(file_name,"READ");

    readFile.set_read_write_tree(false);

    APR aprRead;

    readFile.read_apr(aprRead);

    ParticleData<uint16_t> parts_read;

    readFile.read_particles(aprRead,"parts",parts_read);

    ParticleData<float> parts2_read;

    readFile.read_particles(aprRead,"parts2",parts2_read);

    readFile.close();

    auto apr_iterator = test_data.apr.iterator();
    auto apr_iterator_read = aprRead.iterator();

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                apr_iterator_read.begin(level, z, x);
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator++) {

                    //check the functionality
                    if (test_data.particles_intensities[apr_iterator] !=
                        parts_read[apr_iterator_read]) {
                        success = false;
                    }

                    //check the functionality
                    if (parts2[apr_iterator] !=
                        parts2_read[apr_iterator_read]) {
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

                    if(apr_iterator_read < apr_iterator_read.end()) {
                        apr_iterator_read++;
                    }

                }
            }
        }
    }


    //Test Tree IO and RW and channel
    APRFile TreeFile;
    file_name = "read_write_test_tree.apr";
    TreeFile.open(file_name,"WRITE");

    TreeFile.write_apr(test_data.apr,0,"mem");

    test_data.apr.init_tree();

    ParticleData<float> treeMean;

    APRTreeNumerics::fill_tree_mean(test_data.apr,test_data.particles_intensities,treeMean);

    TreeFile.write_particles(test_data.apr,"tree_parts",treeMean,0,false,"mem");
    TreeFile.write_particles(test_data.apr,"tree_parts1",treeMean,0,false,"mem");
    TreeFile.write_particles(test_data.apr,"tree_parts2",treeMean,0,false,"mem");

    TreeFile.close();

    TreeFile.open(file_name,"READWRITE");
    TreeFile.write_apr(test_data.apr,1,"mem");
    TreeFile.write_particles(test_data.apr,"tree_parts",treeMean,1,false,"mem");

    TreeFile.write_particles(test_data.apr,"particle_demo",test_data.particles_intensities,1,true,"mem");

    TreeFile.write_apr(test_data.apr,10,"ch1_");

    TreeFile.close();

    TreeFile.open(file_name,"READ");

    APR aprRead2;
    TreeFile.read_apr(aprRead2,1,"mem");

    ParticleData<float> treeMeanRead;

    TreeFile.read_particles(aprRead2,"tree_parts",treeMeanRead,1,false,"mem");


    auto tree_it = aprRead2.tree_iterator();
    auto tree_it_org = test_data.apr.tree_iterator();

    for (int l = tree_it.level_max(); l >= tree_it.level_min() ; --l) {
        for (int z = 0; z < tree_it.z_num(l); ++z) {
            for (int x = 0; x < tree_it.x_num(l); ++x) {

                tree_it_org.begin(l, z, x);
                for (tree_it.begin(l,z,x); tree_it < tree_it.end(); tree_it++) {

                    //check the functionality
                    if (treeMean[tree_it_org] !=
                        treeMeanRead[tree_it]) {
                        success = false;
                    }

                    if (tree_it.level() != tree_it_org.level()) {
                        success = false;
                    }

                    if (tree_it.x() != tree_it_org.x()) {
                        success = false;
                    }

                    if (tree_it.y() != tree_it_org.y()) {
                        success = false;
                    }

                    if (tree_it.z() != tree_it_org.z()) {
                        success = false;
                    }

                    if(tree_it_org < tree_it_org.end()) {
                        tree_it_org++;
                    }

                }
            }

        }

    }

    //Test file list
    std::vector<std::string> correct_names = {"tree_parts","tree_parts1","tree_parts2"};

    std::vector<std::string> dataset_names = TreeFile.get_particles_names(0,false,"mem");

    if(correct_names.size() == dataset_names.size()){

        for(int i = 0; i < correct_names.size(); ++i) {
            bool found = false;
            for(int j = 0; j < dataset_names.size(); ++j) {
                if(correct_names[i] == dataset_names[j]){
                    found = true;
                }
            }
            if(!found){
                success = false;
            }
        }

    } else{
        success = false;
    }

    std::vector<std::string> channel_names;
    channel_names = TreeFile.get_channel_names();


    //Test file list
    std::vector<std::string> channel_names_c = {"mem","mem1","ch1_10"};


    if(channel_names_c.size() == channel_names.size()){

        for (int i = 0; i < channel_names_c.size(); ++i) {
            bool found = false;
            for (int j = 0; j < channel_names.size(); ++j) {
                if(channel_names_c[i] == channel_names[j]){
                    found = true;
                }
            }
            if(!found){
                success = false;
            }
        }

    } else{
        success = false;
    }


    uint64_t time_steps = TreeFile.get_number_time_steps("mem");

    if(time_steps != 2){
        success = false;
    }

    TreeFile.close();

    return success;

}






bool test_apr_neighbour_access(TestData& test_data){

    bool success = true;

    APRIterator neighbour_iterator = test_data.apr.iterator();
    APRIterator apr_iterator = test_data.apr.iterator();

    ParticleData<uint16_t> x_p(test_data.apr.total_number_particles());
    ParticleData<uint16_t> y_p(test_data.apr.total_number_particles());
    ParticleData<uint16_t> z_p(test_data.apr.total_number_particles());

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;


        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
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

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
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
                                uint16_t apr_intensity = test_data.particles_intensities[neighbour_iterator];
                                uint16_t check_intensity = test_data.img_pc(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                            neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                            neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

//                                uint16_t x_n = x_p[neighbour_iterator];
//                                uint16_t y_n = y_p[neighbour_iterator];
//                                uint16_t z_n = z_p[neighbour_iterator];

                                if (check_intensity != apr_intensity) {
                                    success = false;
                                }

                                uint16_t apr_level = neighbour_iterator.level();
                                uint16_t check_level = test_data.img_level(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                           neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                           neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_level != apr_level) {
                                    success = false;
                                }

                                uint16_t apr_x = neighbour_iterator.x();
                                uint16_t check_x = test_data.img_x(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_x != apr_x) {
                                    success = false;
                                }

                                uint16_t apr_y = neighbour_iterator.y();
                                uint16_t check_y = test_data.img_y(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_y != apr_y) {
                                    success = false;
                                }

                                uint16_t apr_z = neighbour_iterator.z();
                                uint16_t check_z = test_data.img_z(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

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
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
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
                                uint16_t apr_intensity = (test_data.particles_intensities[neighbour_iterator]);
                                uint16_t check_intensity = test_data.img_pc(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                            neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                            neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_intensity != apr_intensity) {
                                    success = false;
                                }

                                uint16_t apr_level = neighbour_iterator.level();
                                uint16_t check_level = test_data.img_level(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                           neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                           neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_level != apr_level) {
                                    success = false;
                                }


                                uint16_t apr_x = neighbour_iterator.x();
                                uint16_t check_x = test_data.img_x(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_x != apr_x) {
                                    success = false;
                                }

                                uint16_t apr_y = neighbour_iterator.y();
                                uint16_t check_y = test_data.img_y(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_y != apr_y) {
                                    success = false;
                                }

                                uint16_t apr_z = neighbour_iterator.z();
                                uint16_t check_z = test_data.img_z(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

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


bool test_particle_structures(TestData& test_data) {
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator
    //

    bool success = true;

    auto it = test_data.apr.iterator();

    ParticleData<uint16_t> parts;

    parts.data.resize(it.total_number_particles(),0);

    APRTimer timer(true);


    test_data.apr.init_linear();
    auto lin_it = test_data.apr.linear_iterator();

    timer.start_timer("LinearIteration - normal - OpenMP");

    auto counter_p = 0;

    for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
        int z = 0;

        for (z = 0; z < lin_it.z_num(level); z++) {
            for (int x = 0; x < lin_it.x_num(level); ++x) {
                for (lin_it.begin(level, z, x); lin_it < lin_it.end();
                     lin_it++) {
                    //need to add the ability to get y, and x,z but as possible should be lazy.
                    parts[lin_it] += 1;
                    counter_p++;

                }
            }
        }
    }


    timer.stop_timer();

    PartCellData<uint16_t> partCellData;
    partCellData.initialize_structure_parts(test_data.apr);

    timer.start_timer("LinearIteration - PartCell - OpenMP");

    auto counter_pc = 0;

    for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
        int z = 0;

        for (z = 0; z < lin_it.z_num(level); z++) {
            for (int x = 0; x < lin_it.x_num(level); ++x) {
                for (auto begin = lin_it.begin(level, z, x); lin_it < lin_it.end();
                     lin_it++) {
                    //need to add the ability to get y, and x,z but as possible should be lazy.
                    auto off = z*lin_it.x_num(level) + x;
                    auto indx = lin_it - begin;
                    partCellData.data[level][off][indx] += 1;
                    if(partCellData.data[level][off][indx]!=parts[lin_it]){
                        success = false;
                    }
                    counter_pc++;
                }
            }
        }
    }

    timer.stop_timer();

    if(counter_pc != counter_p){
        success = false;
    }

    return success;

}


bool test_linear_iterate(TestData& test_data) {
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator
    //

    bool success = true;

    test_data.apr.init_linear();
    //auto it = test_data.apr.linear_iterator();

    auto it = test_data.apr.linear_iterator();

    uint64_t particle_number = 0;

    uint64_t counter = 0;

    auto it_c = test_data.apr.iterator();

    //need to transfer the particles across


    ParticleData<uint16_t> parts;
    parts.init(test_data.apr.total_number_particles());

    uint64_t c_t = 0;
    for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {
                for (it_c.begin(level, z, x); it_c < it_c.end();
                     it_c++) {

                    parts[c_t] = test_data.particles_intensities[it_c];
                    c_t++;
                }
            }
        }
    }



    for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                it_c.begin(level, z, x);

                for (it.begin(level, z, x); it < it.end();
                     it++) {

                    uint16_t apr_intensity = (parts[it]);
                    uint16_t check_intensity = test_data.img_pc(it.y_nearest_pixel(level,it.y()),
                                                                it.x_nearest_pixel(level,x),
                                                                it.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                        particle_number++;
                    }

                    uint16_t apr_level = level;
                    uint16_t check_level = test_data.img_level(it.y_nearest_pixel(level,it.y()),
                                                               it.x_nearest_pixel(level,x),
                                                               it.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }


                    uint16_t apr_x = x;
                    uint16_t check_x = test_data.img_x(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = it.y();
                    uint16_t check_y = test_data.img_y(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = z;
                    uint16_t check_z = test_data.img_z(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_z != apr_z) {
                        success = false;
                    }

                    counter++;

                    if(it_c < it_c.end()){
                        it_c++;
                    }

                }
            }
        }

    }

    std::cout << particle_number << std::endl;

    //Test parallel loop

    for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(it)
#endif
        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level, z, x); it < it.end();
                     it++) {

                    uint16_t apr_intensity = (parts[it]);
                    uint16_t check_intensity = test_data.img_pc(it.y_nearest_pixel(level,it.y()),
                                                                it.x_nearest_pixel(level,x),
                                                                it.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                    }

                    uint16_t apr_level = level;
                    uint16_t check_level = test_data.img_level(it.y_nearest_pixel(level,it.y()),
                                                               it.x_nearest_pixel(level,x),
                                                               it.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }



                    uint16_t apr_x = x;
                    uint16_t check_x = test_data.img_x(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = it.y();
                    uint16_t check_y = test_data.img_y(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = z;
                    uint16_t check_z = test_data.img_z(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_z != apr_z) {
                        success = false;
                    }
                }
            }
        }

    }

    test_data.apr.init_linear();
    auto it_l = test_data.apr.linear_iterator();

    for (unsigned int level = it_l.level_min(); level <= it_l.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it_l.z_num(level); z++) {
            for (x = 0; x < it_l.x_num(level); ++x) {

                for (it_l.begin(level, z, x); it_l < it_l.end();
                     it_l++) {

                    uint16_t apr_intensity = (parts[it_l]);
                    uint16_t check_intensity = test_data.img_pc(it_l.y_nearest_pixel(level, it_l.y()),
                                                                it_l.x_nearest_pixel(level, x),
                                                                it_l.z_nearest_pixel(level, z));

                    if(apr_intensity != check_intensity){
                        success = false;
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

    auto apr_iterator = test_data.apr.iterator();

    uint64_t particle_number = 0;

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator++) {

                    uint16_t apr_intensity = (test_data.particles_intensities[apr_iterator]);
                    uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                                apr_iterator.x_nearest_pixel(level,x),
                                                                apr_iterator.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                        particle_number++;
                    }

                    uint16_t apr_level = apr_iterator.level();
                    uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                               apr_iterator.x_nearest_pixel(level,x),
                                                               apr_iterator.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }



                    uint16_t apr_x = apr_iterator.x();
                    uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = apr_iterator.y();
                    uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = apr_iterator.z();
                    uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_z != apr_z) {
                        success = false;
                    }
                }
            }
        }

    }

    std::cout << particle_number << std::endl;

    //Test parallel loop

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    uint16_t apr_intensity = (test_data.particles_intensities[apr_iterator]);
                    uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                                apr_iterator.x_nearest_pixel(level,x),
                                                                apr_iterator.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                    }

                    uint16_t apr_level = apr_iterator.level();
                    uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                               apr_iterator.x_nearest_pixel(level,x),
                                                               apr_iterator.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }



                    uint16_t apr_x = apr_iterator.x();
                    uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = apr_iterator.y();
                    uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = apr_iterator.z();
                    uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

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

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
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

    APRConverter<uint16_t> aprConverter;

    //the apr datastructure
    APR apr;

    //read in the command line options into the parameters file
   aprConverter.par.Ip_th = test_data.apr.parameters.Ip_th;
   aprConverter.par.rel_error = test_data.apr.parameters.rel_error;
   aprConverter.par.lambda = test_data.apr.parameters.lambda;
   aprConverter.par.mask_file = "";
   aprConverter.par.min_signal = -1;

   aprConverter.par.sigma_th_max = test_data.apr.parameters.sigma_th_max;
   aprConverter.par.sigma_th = test_data.apr.parameters.sigma_th;

   aprConverter.par.SNR_min = -1;

    //where things are
   aprConverter.par.input_image_name = test_data.filename;
   aprConverter.par.input_dir = "";
   aprConverter.par.name = test_data.output_name;
   aprConverter.par.output_dir = "";

    //Gets the APR
    ParticleData<uint16_t> particles_intensities;

    // #TODO: Need to remove the by file name get APR method.


    if(aprConverter.get_apr(apr,test_data.img_original)){

        particles_intensities.sample_parts_from_img_downsampled(apr,test_data.img_original);

        auto apr_iterator = apr.iterator();

        std::cout << "NUM OF PARTICLES: " << apr_iterator.total_number_particles() << " vs " << test_data.apr.total_number_particles() << std::endl;

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

            for (z = 0; z < apr_iterator.z_num(level); z++) {
                for (x = 0; x < apr_iterator.x_num(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        uint16_t apr_intensity = (particles_intensities[apr_iterator]);
                        uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                                    apr_iterator.x_nearest_pixel(level,x),
                                                                    apr_iterator.z_nearest_pixel(level,z));

                        if (check_intensity != apr_intensity) {
                            success = false;
                        }

                        uint16_t apr_level = apr_iterator.level();
                        uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                                   apr_iterator.x_nearest_pixel(level,x),
                                                                   apr_iterator.z_nearest_pixel(level,z));

                        if (check_level != apr_level) {
                            success = false;
                        }

                        uint16_t apr_x = apr_iterator.x();
                        uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                           apr_iterator.x_nearest_pixel(level,x),
                                                           apr_iterator.z_nearest_pixel(level,z));

                        if (check_x != apr_x) {
                            success = false;
                        }

                        uint16_t apr_y = apr_iterator.y();
                        uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                           apr_iterator.x_nearest_pixel(level,x),
                                                           apr_iterator.z_nearest_pixel(level,z));

                        if (check_y != apr_y) {
                            success = false;
                        }

                        uint16_t apr_z = apr_iterator.z();
                        uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                           apr_iterator.x_nearest_pixel(level,x),
                                                           apr_iterator.z_nearest_pixel(level,z));

                        if (check_z != apr_z) {
                            success = false;
                        }

                    }
                }
            }
        }

    } else {

        success = false;
    }


    return success;
}

bool test_pipeline_bound(TestData& test_data,float rel_error){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    bool success = true;

    //the apr datastructure
    APR apr;

    //read in the command line options into the parameters file
    apr.parameters.Ip_th = 0;
    apr.parameters.rel_error = rel_error;
    apr.parameters.lambda = 0;
    apr.parameters.mask_file = "";
    apr.parameters.min_signal = -1;

    apr.parameters.sigma_th_max = 50;
    apr.parameters.sigma_th = 100;

    apr.parameters.SNR_min = -1;

    apr.parameters.auto_parameters = false;

    apr.parameters.output_steps = true;

    //where things are
    apr.parameters.input_image_name = test_data.filename;
    apr.parameters.input_dir = "";
    apr.parameters.name = test_data.output_name;
    apr.parameters.output_dir = test_data.output_dir;

    //Gets the APR
    APRConverter<uint16_t> aprConverter;
    aprConverter.par = apr.parameters;

    ParticleData<uint16_t> particles_intensities;

    aprConverter.get_apr(apr,test_data.img_original);

    particles_intensities.sample_parts_from_img_downsampled(apr,test_data.img_original);

    PixelData<uint16_t> pc_recon;
    APRReconstruction::interp_img(apr,pc_recon,particles_intensities);

    //read in used scale

    PixelData<float> scale = TiffUtils::getMesh<float>(test_data.output_dir + "local_intensity_scale_step.tif");

    TestBenchStats benchStats;

    benchStats =  compare_gt(test_data.img_original,pc_recon,scale);

    std::cout << "Inf norm: " << benchStats.inf_norm << " Bound: " << rel_error << std::endl;

    if(benchStats.inf_norm > rel_error){
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

void CreateGTSmallTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_small/";
}

void CreateGTSmall2DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_2D/";
}

void CreateGTSmall1DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_1D/";
}




void CreateSmallSphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_apr.h5";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

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
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

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
ASSERT_TRUE(test_linear_iterate(test_data));



}

TEST_F(CreateSmallSphereTest, APR_TREE) {

//test iteration
ASSERT_TRUE(test_apr_tree(test_data));

}

TEST_F(CreateSmallSphereTest, APR_NEIGHBOUR_ACCESS) {

//test iteration
ASSERT_TRUE(test_apr_neighbour_access(test_data));

}

TEST_F(CreateSmallSphereTest, APR_INPUT_OUTPUT) {

//test iteration
   // ASSERT_TRUE(test_apr_input_output(test_data));

    ASSERT_TRUE(test_apr_file(test_data));

}

TEST_F(CreateGTSmallTest, APR_PIPELINE_3D) {

//test pipeline
    ASSERT_TRUE(test_pipeline_bound(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.001));

}

TEST_F(CreateGTSmall2DTest, APR_PIPELINE_2D) {

//test pipeline
    ASSERT_TRUE(test_pipeline_bound(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.001));

}

TEST_F(CreateGTSmall1DTest, APR_PIPELINE_1D) {

//test pipeline
    ASSERT_TRUE(test_pipeline_bound(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.001));

}

TEST_F(Create210SphereTest, APR_ITERATION) {

//test iteration
    ASSERT_TRUE(test_apr_iterate(test_data));
    ASSERT_TRUE(test_linear_iterate(test_data));

}

TEST_F(Create210SphereTest, APR_TREE) {

//test iteration
    ASSERT_TRUE(test_apr_tree(test_data));

}

TEST_F(Create210SphereTest, APR_NEIGHBOUR_ACCESS) {

//test iteration
    ASSERT_TRUE(test_apr_neighbour_access(test_data));

}


TEST_F(Create210SphereTest, APR_PARTICLES) {

    ASSERT_TRUE(test_particle_structures(test_data));
}

TEST_F(CreateSmallSphereTest, APR_PARTICLES) {

    ASSERT_TRUE(test_particle_structures(test_data));
}


TEST_F(Create210SphereTest, APR_INPUT_OUTPUT) {

//test iteration
    //ASSERT_TRUE(test_apr_input_output(test_data));
    ASSERT_TRUE(test_apr_file(test_data));

}

//TEST_F(Create210SphereTest, APR_PIPELINE) {
//
////test iteration
//// TODO: FIXME please! I'm not sure the difference arises regarding the fastmath optimization resulting in small float changes in the solution
////    ASSERT_TRUE(test_apr_pipeline(test_data)); I have replaced these tests with newer set of tests.
//
//}


int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
