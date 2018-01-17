//
// Created by cheesema on 06.10.17.
//

//
// Created by cheesema on 31/01/17.
//

#include "generate_anisotropic_single.hpp"


int main(int argc, char **argv) {

    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    std::cout << "Generate Single Syn Image" << std::endl;

    cmdLineOptionsBench options = read_command_line_options(argc, argv);

    SynImage syn_image;

    std::string image_name = options.template_name;

    benchmark_settings bs;

    set_up_benchmark_defaults(syn_image, bs);

    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////

    AnalysisData analysis_data(options.description, "Gen Single Image", argc, argv);

    process_input(options, syn_image, analysis_data, bs);

    bs.num_objects = options.delta;

    bs.obj_size = 2;
    bs.sig = 1;
    //bs.desired_I = 10000;
    float ratio = 10;
    bs.num_objects = 5*pow(bs.x_num,3)/(33400*ratio);

    bs.num_objects = 10;

    bs.desired_I = 1000;
    //bs.int_scale_max = 1;
    //bs.int_scale_min = 1;
    bs.int_scale_min = 1;
    bs.int_scale_max = 10;

    syn_image.global_trans.const_shift = 1000;

    obj_properties obj_prop(bs);

    Object_template basic_object = get_object_template(options, obj_prop);

    syn_image.object_templates.push_back(basic_object);

    SynImage syn_image_loc = syn_image;

    //add the basic sphere as the standard template

    ///////////////////////////////////////////////////////////////////
    //PSF properties

    set_gaussian_psf(syn_image_loc, bs);

    generate_objects(syn_image_loc, bs);

    SynImage syn_image_aniso = syn_image_loc;

    float z_ratio = 4;

    float dx = 0.1;
    float dz = 0.1;

    float psfx = bs.sig;
    float psfz = bs.sig*z_ratio;

    syn_image_aniso.sampling_properties.sampling_delta[0] = dx;
    syn_image_aniso.sampling_properties.sampling_delta[1] = dx;
    syn_image_aniso.sampling_properties.sampling_delta[2] = dz;

    syn_image_aniso.PSF_properties.real_sigmas[0] = psfx*dx;
    syn_image_aniso.PSF_properties.real_sigmas[1] = psfx*dx;
    syn_image_aniso.PSF_properties.real_sigmas[2] = psfz*dx;

    syn_image_aniso.PSF_properties.set_guassian_window_size();

    bs.rel_error = 0.1;


    ///////////////////////////////
    //
    //  Generate the image
    //
    ////////////////////////////////

    MeshDataAF<uint16_t> gen_image;

    syn_image_loc.generate_syn_image(gen_image);

    MeshData<uint16_t> input_img;

    copy_mesh_data_structures(gen_image, input_img);

    ///////////////////////////////
    //
    //  Get the APR
    //
    //////////////////////////////

    Part_rep p_rep;

    set_up_part_rep(syn_image_loc, p_rep, bs);

    PartCellStructure<float, uint64_t> pc_struct;
    //bench_get_apr(input_img, p_rep, pc_struct, analysis_data);

    //write_image_tiff(input_img, p_rep.pars.output_path + p_rep.pars.name + ".tif");

    //std::cout << pc_struct.get_number_parts() << std::endl;




    ///////////////////////////////
    //
    //  Calculate analysis of the result
    //
    ///////////////////////////////

    //produce_apr_analysis(input_img, analysis_data, pc_struct, syn_image_loc, p_rep.pars);



    ///////////////////////////////
    //
    //  Generate the aniso image
    //
    ////////////////////////////////

    MeshDataAF<uint16_t> gen_image_a;

    syn_image_aniso.generate_syn_image(gen_image_a);

    MeshData<uint16_t> input_img_a;

    copy_mesh_data_structures(gen_image_a, input_img_a);

    ///////////////////////////////
    //
    //  Get the APR
    //
    //////////////////////////////

    Part_rep p_rep_a;

    set_up_part_rep(syn_image_aniso, p_rep_a, bs);

    p_rep_a.pars.name = "aniso";

    p_rep_a.pars.dz = dz;
    p_rep_a.pars.psfz = psfz*dx;



    PartCellStructure<float, uint64_t> pc_struct_a;
    bench_get_apr(input_img_a, p_rep_a, pc_struct_a, analysis_data);

    write_image_tiff(input_img_a, p_rep_a.pars.output_path + p_rep_a.pars.name + ".tif");

    std::cout << pc_struct_a.get_number_parts() << std::endl;

    ///////////////////////////////
    //
    //  Calculate analysis of the result
    //
    ///////////////////////////////

    produce_apr_analysis(input_img_a, analysis_data, pc_struct_a, syn_image_aniso, p_rep_a.pars);

    analysis_data.write_analysis_data_hdf5();

    af::info();

    std::vector<float> scale = {2,2,4};

    MeshData<float> smooth_img;

    interp_parts_to_smooth(smooth_img,pc_struct_a.part_data.particle_data,pc_struct_a,scale);

    debug_write(smooth_img,"smooth_test");

    MeshData<float> rec_img;

    pc_struct_a.interp_parts_to_pc(rec_img,pc_struct_a.part_data.particle_data);

    debug_write(rec_img,"rec_img");


    return 0;
}
