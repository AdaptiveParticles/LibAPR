//
// Created by msusik on 29.08.16.
//

#include "alg_cells_test.h"
#include "utils.h"

#include "benchmarks/development/old_io/readimage.h"
#include "benchmarks/development/old_io/write_parts.h"
#include "benchmarks/development/old_structures/particle_map.hpp"

void CreateResultTest::SetUp(){
    tests_directory = testFilesDirectory();
}

Particle_map<float> CreateResultFromFilesTest::create_result(std::string grad_path,
                                                             std::string input_path){

    MeshData<uint16_t> grad_int, input_int;
    MeshData<float> var, grad, input;

    load_image_tiff(grad_int, grad_path);
    load_image_tiff(input_int, input_path);

    grad = grad_int.to_type<float>();
    input = input_int.to_type<float>();

    p_rep = Part_rep(grad.y_num, grad.x_num, grad.z_num);
    p_rep.pars.dy = 1;
    p_rep.pars.dx = 1;
    p_rep.pars.dz = 1;
    p_rep.pars.rel_error = 1000;
    timer.verbose_flag = true;

    Particle_map<float> part_map(p_rep);
    preallocate(part_map.layers, grad.y_num, grad.x_num, grad.z_num, p_rep);

    MeshData<float> temp;
    temp.preallocate(grad.y_num, grad.x_num, grad.z_num, 0);
    var.preallocate(grad.y_num, grad.x_num, grad.z_num, 1);

    get_level_3D(var, grad, p_rep, part_map, temp);

    part_map.pushing_scheme(p_rep);

    std::vector<MeshData<float>> down_sampled_images;
    preallocate(down_sampled_images, grad.y_num, grad.x_num, grad.z_num, p_rep);

    return part_map;
}


TEST_P(CreateResultFromFilesTest, RESULT_FROMFILE_IMAGE_PARAM){

    std::string grad = tests_directory + std::get<0>(GetParam());
    std::string output_dir = tests_directory + std::get<1>(GetParam());
    std::string input_dir = tests_directory + std::get<2>(GetParam());

    Particle_map<float> to_compare = create_result(grad, input_dir);

    /*timer.start_timer("writing");
    write_apr_to_hdf5(to_compare, tests_directory + "hdf5/", std::get<1>(GetParam()));
    timer.stop_timer();*/
    ASSERT_TRUE(compare_part_rep_with_particle_map(to_compare, output_dir));
}


#if PROFILING == 0


INSTANTIATE_TEST_CASE_P(RESULT_RANDOM_IMAGE,
                        CreateResultFromFilesTest,
                        ::testing::Values(
                                grad_and_var_paths("files/images_k_input/bigx.tif",
                                                   "files/hdf5/bigx_full.h5",
                                                   "files/images_input/test_bigx.tif"),
                                grad_and_var_paths("files/images_k_input/bigy.tif",
                                                   "files/hdf5/bigy_full.h5",
                                                   "files/images_input/test_bigy.tif"),
                                grad_and_var_paths("files/images_k_input/bigz.tif",
                                                   "files/hdf5/bigz_full.h5",
                                                   "files/images_input/test_bigz.tif")
                        ));




INSTANTIATE_TEST_CASE_P(RESULT_PROFILING,
                        CreateResultFromFilesTest,
                        ::testing::Values(
                                grad_and_var_paths("files/images_k_input/huge.tif",
                                                   "files/hdf5/huge_full.h5",
                                                   "files/images_input/test_huge.tif")
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