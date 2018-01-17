#include "k_test.h"
#include "utils.h"

#include "benchmarks/development/old_io/readimage.h"
#include "benchmarks/development/old_io/writeimage.h"


#include <algorithm>

void CreateKTest::SetUp(){
    tests_directory = get_source_directory();
}

Particle_map<float> CreateKFromFilesTest::create_k(std::string grad_path,
                                                   std::string output_path){

    MeshData<uint16_t > grad_int;
    MeshData<float> var, grad;

    load_image_tiff(grad_int, grad_path);

    grad = grad_int.to_type<float>();

    p_rep = Part_rep(grad.y_num, grad.x_num, grad.z_num);
    p_rep.pars.dy = 1;
    p_rep.pars.dx = 1;
    p_rep.pars.dz = 1;
    p_rep.pars.rel_error = 1000;
    timer.verbose_flag = true;

    Particle_map<float> part_map(p_rep);

    timer.start_timer("get_level_3D");

    MeshData<float> temp;
    temp.preallocate(grad.y_num, grad.x_num, grad.z_num, 0);
    var.preallocate(grad.y_num, grad.x_num, grad.z_num, 1);

    get_level_3D(var, grad, p_rep, part_map, temp);

    timer.stop_timer();

    return part_map;

}

TEST_P(CreateKFromFilesTest, K_FROMFILE_IMAGE_PARAM){

    std::string grad = tests_directory + std::get<0>(GetParam());
    std::string output_dir = tests_directory + std::get<1>(GetParam());

    Particle_map<float> to_compare = create_k(grad, output_dir);

    ASSERT_TRUE(compare_two_ks(to_compare, output_dir));
}

#if PROFILING == 0


INSTANTIATE_TEST_CASE_P(K_RANDOM_IMAGE,
        CreateKFromFilesTest,
        ::testing::Values(
        grad_and_var_paths("files/images_k_input/bigx.tif",
                           "files/images_k/test_bigx"),
        grad_and_var_paths("files/images_k_input/bigy.tif",
                           "files/images_k/test_bigy"),
        grad_and_var_paths("files/images_k_input/bigz.tif",
                           "files/images_k/test_bigz")
));


INSTANTIATE_TEST_CASE_P(K_PROFILING,
        CreateKFromFilesTest,
        ::testing::Values(
        grad_and_var_paths("files/images_k_input/huge.tif",
                           "files/images_k/test_huge")
));


#elif PROFILING == 2

INSTANTIATE_TEST_CASE_P(BSPLINE_PROFILING,
                        CreateKFromFilesTest,
                        ::testing::Values(
                                grad_and_var_paths("images_k_input/huge.tif",
                                                   "images_k/test_huge")
                        ));



#endif // PROFILING==0

int main(int argc, char **argv) {


    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}