////////////////////////
//
//  Bevan Cheeseman, Mateusz Susik 2016
//
//  Benchmarking test code for calculating image gradients using smoothing BSplines
//
////////////////////////

#include "tests.h"
#include "utils.h"

#include "../src/io/readimage.h"

Mesh_data<uint16_t> CreateImageTest::create_bspline_empty() {

    return create_test_empty(false);

}

Mesh_data<uint16_t> CreateImageTest::create_variance_empty() {

    return create_test_empty(true);

}

Mesh_data<uint16_t> CreateImageTest::create_test_empty(bool variance) {

    //generate the input image
    Mesh_data<float> input;

    //set to constant 13
    input.initialize(SIZE, SIZE, SIZE, 13);

    //the output of the grad magnitude
    Mesh_data<float> grad;

    grad.initialize(SIZE, SIZE, SIZE, 0);

    //start timer
    timer.start_timer("grad_cpu");

    //call the pipeline
    if(variance) {
        p_rep = Part_rep(input.y_num, input.x_num, input.z_num);
        p_rep.pars.var_th = 0;
        p_rep.pars.var_th_max = 0;
        p_rep.pars.dy = p_rep.pars.dx = p_rep.pars.dz = 1;
        p_rep.pars.psfx = p_rep.pars.psfy = p_rep.pars.psfz = 0.1;
        grad.preallocate(input.y_num,
                         input.x_num,
                         input.z_num, 0);
        get_variance_3D(p_rep, input, grad);
    } else {
        get_gradient_3D(p_rep, input, grad);
    }

    timer.stop_timer();

    Mesh_data<uint16_t> grad2 = grad.to_type<uint16_t>();

    return grad2;

}

Mesh_data<uint16_t> CreateImageFromFileTest::create_variance(std::string image_path) {

    return create_test(image_path, true);

}

Mesh_data<uint16_t> CreateImageFromFileTest::create_bspline(std::string image_path) {

    return create_test(image_path, false);

}


Mesh_data<uint16_t> CreateImageFromFileTest::create_test(std::string image_path,
                                                         bool variance) {

    //////////////////////////////////////////
    //
    //
    //  Test with input tiff images
    //
    //
    //////////////////////////////////////////

    //generate the input image
    Mesh_data<uint16_t> input_image;

    load_image_tiff(input_image, image_path);

    //the output of the grad magnitude
    Mesh_data<float> grad_tiff;

    if(!variance) {
        grad_tiff.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    }

    //start timer

    Mesh_data<float> input_image_float = input_image.to_type<float>();
    timer.start_timer("whole_part");

    //call the pipeline
    if(variance) {
        p_rep = Part_rep(input_image_float.y_num, input_image_float.x_num, input_image_float.z_num);
        p_rep.pars.var_th = 0;
        p_rep.pars.var_th_max = 0;
        p_rep.pars.dy = p_rep.pars.dx = p_rep.pars.dz = 0.5;
        p_rep.pars.psfx = p_rep.pars.psfy = p_rep.pars.psfz = 1;
        grad_tiff.preallocate(input_image_float.y_num,
                              input_image_float.x_num,
                              input_image_float.z_num, 0);
        get_variance_3D(p_rep, input_image_float, grad_tiff);
    } else {
        get_gradient_3D(p_rep, input_image_float, grad_tiff);
    }

    timer.stop_timer();

    return grad_tiff.to_type<uint16_t>();
}

void CreateImageTest::SetUp() {
    //set output timer to true
    timer.verbose_flag = false;

    //generate a part_rep object (only used for parameters here)
    p_rep = Part_rep(SIZE, SIZE, SIZE);
    //lambda is default set to 1

    tests_directory = get_source_directory();
}

