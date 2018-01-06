//
// Created by ulrik on 27.10.16.
//

#include "raytrace_test.h"
#include "../utils.h"

#include "../../src/io/readimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"

void CreateResultTest::SetUp(){
    tests_directory = get_source_directory();
}

Particle_map<float> RaytraceTest::create_result(std::string tiff_path, std::string stats_path, std::string image_name) {

    std::cout << "Hello from RaytraceTest!" << std::endl;

    Part_rep part_rep;


    // COMPUTATIONS

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    {
        Mesh_data<uint16_t> input_image;

        load_image_tiff(input_image, tiff_path);

        gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

        input_image_float = input_image.to_type<float>();

        // After this block, input_image will be freed.
    }

    get_image_stats(part_rep.pars, stats_path.substr(0, stats_path.find_last_of("/")+1), image_name.substr(image_name.find_last_of("/")+1));

    Part_timer t;
    t.verbose_flag = true;

    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<float>> down_sampled_images;

    // variables for tree
    std::vector<uint64_t> tree_mem(gradient.y_num * gradient.x_num * gradient.z_num * 1.25, 0);
    std::vector<Content> contents(gradient.y_num * gradient.x_num * gradient.z_num, {0});

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");

    part_map.downsample(input_image_float);

    std::cout << "all parts: " << part_map.all_parts << std::endl;


    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);

    part_rep.timer.start_timer("get_gradient_3D");
    get_gradient_3D(part_rep, input_image_float, gradient);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("get_variance_3D");
    get_variance_3D(part_rep, input_image_float, variance);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("get_level_3D");
    get_level_3D(variance, gradient, part_rep, part_map, temp);
    part_rep.timer.stop_timer();


    // free memory (not used anymore)
    std::vector<float>().swap( gradient.mesh );
    std::vector<float>().swap( variance.mesh );


    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();

    part_rep.timer.start_timer("estimate_part_intensity");

    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);

    Tree<float> tree(part_map, tree_mem, contents);
    part_rep.timer.stop_timer();
    std::cout << "all parts: " << part_map.all_parts << std::endl;

    t.stop_timer();

    std::cout << "Tree size=" << tree.get_content_size() << std::endl;

#define MAX_NUM_OF_CHILDREN 15
#define MAX_LEVELS 8

    for(unsigned int level = 0; level <= MAX_LEVELS; level++) {
        LevelIterator<float> it(tree, level);
        std::vector<uint64_t> neighbours(MAX_NUM_OF_CHILDREN);
        std::vector<coords3d> particles;

        std::cout << "coords=" << it.get_current_coords() << ", " << it.level_multiplier << std::endl;

        it.get_current_particle_coords(particles);

        std::cout << "Particle list:\n------------------" << std::endl;

        for(auto part: particles) {
            std::cout << part << std::endl;

        }

        std::cout << "------------------" << std::endl;

        tree.get_neighbours(*it, it.get_current_coords(), it.level_multiplier,
                            it.child_index, neighbours);

        std::cout << "Level " << level << " neighbours=" << neighbours.size() << std::endl;
    }

    return part_map;
}


TEST_P(RaytraceTest, RESULT_FROMFILE_IMAGE_PARAM){

std::string tiff_path = tests_directory + std::get<0>(GetParam());
std::string stats_path = tests_directory + std::get<1>(GetParam());
std::string image_name = tests_directory + std::get<2>(GetParam());

Particle_map<float> to_compare = create_result(tiff_path, stats_path, image_name);

/*timer.start_timer("writing");
write_apr_to_hdf5(to_compare, tests_directory + "hdf5/", std::get<1>(GetParam()));
timer.stop_timer();*/
}


#if PROFILING == 0


INSTANTIATE_TEST_CASE_P(RESULT_RAYTRACE,
        RaytraceTest,
        ::testing::Values(
        grad_and_var_paths("../APR_tests/test_sphere1.tif",
                           "../APR_tests/", "test_sphere1")
));

#elif PROFILING == 2

INSTANTIATE_TEST_CASE_P(RESULT_PROFILING,
                        CreateResultFromFilesTest,
                        ::testing::Values(
                                grad_and_var_paths("images_k_input/huge.tif",
                                                   "images_k/test_huge",
                                                   "images_input/test_huge.tif")
                        ));


#endif // PROFILING==0

int main(int argc, char **argv) {


    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}