//
// Created by cheesema on 31/01/17.
//

#include "generate_single_image.hpp"


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

    Genrand_uni gen_rand;

    obj_properties obj_prop(bs);

    Object_template basic_object = get_object_template(options, obj_prop);

    syn_image.object_templates.push_back(basic_object);


    SynImage syn_image_loc = syn_image;

    //add the basic sphere as the standard template

    ///////////////////////////////////////////////////////////////////
    //PSF properties

    set_gaussian_psf(syn_image_loc, bs);


    generate_objects(syn_image_loc, bs);


    ///////////////////////////////
    //
    //  Generate the image
    //
    ////////////////////////////////

    MeshDataAF<uint16_t> gen_image;

    syn_image_loc.generate_syn_image(gen_image);

    Mesh_data<uint16_t> input_img;

    copy_mesh_data_structures(gen_image, input_img);


    ///////////////////////////////
    //
    //  Get the APR
    //
    //////////////////////////////


    Part_rep p_rep;

    set_up_part_rep(syn_image_loc, p_rep, bs);

    // Get the APR

    for(int i = 0; i < 4;++i) {

        PartCellStructure<float, uint64_t> pc_struct;

        p_rep.pars.interp_type = i;

        bench_get_apr(input_img, p_rep, pc_struct, analysis_data);

        write_image_tiff(input_img, p_rep.pars.output_path + p_rep.pars.name + ".tif");

        ///////////////////////////////
        //
        //  Calculate analysis of the result
        //
        ///////////////////////////////

        produce_apr_analysis(input_img, analysis_data, pc_struct, syn_image_loc, p_rep.pars);

    }

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;
}
