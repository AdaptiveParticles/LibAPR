

#include "tests.h"
#include "utils.h"

#include <algorithm>
#include "../src/io/writeimage.h"


TEST_P(CreateImageFromFileTest, BSPLINE_FROMFILE_IMAGE_PARAM)
{

    // param -> tuple<string. string. string, float> (input filename, output filename, stats filename lambda)
    std::string input = tests_directory + std::get<0>(GetParam());
    std::string output = tests_directory + std::get<1>(GetParam());

    std::string stats_filename = std::get<2>(GetParam());
    if( stats_filename.length() ) {
        get_image_stats(p_rep.pars, tests_directory, stats_filename);
    } else {
        p_rep.pars.dy = p_rep.pars.dx = p_rep.pars.dz = 1;
    }

    p_rep.pars.lambda = std::get<3>(GetParam());
    Mesh_data<uint16_t> to_compare = create_bspline(input);
    //write_image_tiff(to_compare, output);
    ASSERT_TRUE(compare_two_images(to_compare, output));
}

#if PROFILING == 0


TEST_F(CreateImageTest, BSPLINE_EMPTY_IMAGE)
{
    auto uint_grad = create_bspline_empty();

    // The result should be a vector of zeros. Below we compute
    // the vector of unique values
    auto it = std::unique(uint_grad.mesh.begin(), uint_grad.mesh.end());
    uint_grad.mesh.resize(std::distance(uint_grad.mesh.begin(), it));

    ASSERT_EQ(uint_grad.mesh.size(), 1);
    ASSERT_EQ(uint_grad.mesh.at(0), 0);

}




INSTANTIATE_TEST_CASE_P(BSPLINE_FROMFILE_IMAGE,
        CreateImageFromFileTest,
        ::testing::Values(
        filepaths_lambda("files/images_input/test_sphere.tif",
                         "files/images_output/test_sphere0.tif",
                         "files/stats_input/test_sphere", 0.1),
        filepaths_lambda("files/images_input/test_sphere.tif",
                         "files/images_output/test_sphere1.tif",
                         "files/stats_input/test_sphere", 1),
        filepaths_lambda("files/images_input/test_sphere.tif",
                         "files/images_output/test_sphere2.tif",
                         "files/stats_input/test_sphere", 10),
        filepaths_lambda("files/images_input/test_sphere_clean.tif",
                         "files/images_output/test_sphere_clean0.tif",
                         "files/stats_input/test_sphere_clean", 0.1),
        filepaths_lambda("files/images_input/test_sphere_clean.tif",
                         "files/images_output/test_sphere_clean1.tif",
                         "files/stats_input/test_sphere_clean", 1),
        filepaths_lambda("files/images_input/test_sphere_clean.tif",
                         "files/images_output/test_sphere_clean2.tif",
                         "files/stats_input/test_sphere_clean", 10)
));


INSTANTIATE_TEST_CASE_P(BSPLINE_RANDOM_IMAGE,
        CreateImageFromFileTest,
        ::testing::Values(
        filepaths_lambda("files/images_input/test_bigx.tif",
                         "files/images_output/test_bigx.tif",
                         "", 1),
        filepaths_lambda("files/images_input/test_bigy.tif",
                         "files/images_output/test_bigy.tif",
                         "", 1),
        filepaths_lambda("files/images_input/test_bigz.tif",
                         "files/images_output/test_bigz.tif",
                         "", 1)
));


INSTANTIATE_TEST_CASE_P(BSPLINE_PROFILING,
        CreateImageFromFileTest,
        ::testing::Values(
        filepaths_lambda("files/images_input/test_huge.tif",
                         "files/images_output/test_huge.tif",
                         "", 100)
));

#elif PROFILING == 1

INSTANTIATE_TEST_CASE_P(BSPLINE_FROMFILE_IMAGE,
                        CreateImageFromFileTest,
                        ::testing::Values(
                                filepaths_lambda("images_input/test_sphere.tif",
                                                 "images_output/test_sphere0.tif",
                                                 "stats_input/test_sphere", 0.1)
                        )
);

#elif PROFILING == 2

INSTANTIATE_TEST_CASE_P(BSPLINE_PROFILING,
                        CreateImageFromFileTest,
                        ::testing::Values(
                                filepaths_lambda("images_input/test_huge.tif",
                                                 "images_output/test_huge.tif",
                                                 "", 100)
                        ));


#else

std::string global;

INSTANTIATE_TEST_CASE_P(BSPLINE_PROFILING,
                        CreateImageFromFileTest,
                        ::testing::Values(
                                filepaths_lambda(global,
                                                 "",
                                                 "", 100)
                        ));

#endif // PROFILING==0

int main(int argc, char **argv) {

#if PROFILING == 3

    global = argv[1];


#endif

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}