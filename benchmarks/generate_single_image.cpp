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


    bs.obj_size = 3;
    bs.sig = 2.0;
    //bs.desired_I = 10000;
    float ratio = 10;
    bs.num_objects = 5*pow(bs.x_num,3)/(33400*ratio);

    bs.num_objects = 10;

    bs.desired_I = 1000  ;
    //bs.int_scale_max = 1;
    //bs.int_scale_min = 1;
    bs.int_scale_min = 1;
    bs.int_scale_max = 10;

    obj_properties obj_prop(bs);

    Object_template basic_object = get_object_template(options, obj_prop);

    syn_image.object_templates.push_back(basic_object);


    SynImage syn_image_loc = syn_image;

    //add the basic sphere as the standard template

    ///////////////////////////////////////////////////////////////////
    //PSF properties


    //bs.sig = 5;

    bs.rel_error = 0.1;

    // Get the APR
    //bs.num_objects = 10;
    //bs.lambda = 50;


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


        p_rep.timer.verbose_flag = true;

        p_rep.pars.pull_scheme = 2;

       // p_rep.pars.var_th = 1;

        p_rep.pars.lambda = 1.0;

        //p_rep.pars.interp_type = 4;

        PartCellStructure<float, uint64_t> pc_struct;

        //p_rep.pars.interp_type = i;

       // p_rep.pars.lambda = 4;

        p_rep.pars.var_scale = 1.0;

        p_rep.pars.padd_dims = {2,2,2,4,4,4};

        bench_get_apr(input_img, p_rep, pc_struct, analysis_data);


        write_image_tiff(input_img, p_rep.pars.output_path + p_rep.pars.name + ".tif");


        std::cout << pc_struct.get_number_parts() << std::endl;

        ///////////////////////////////
        //
        //  Calculate analysis of the result
        //
        ///////////////////////////////

        produce_apr_analysis(input_img, analysis_data, pc_struct, syn_image_loc, p_rep.pars);



    //compute_var_ratio_perfect(syn_image_loc,p_rep,input_img,analysis_data);

//    p_rep.pars.name = "perfect";
//
//    Mesh_data<float> norm_grad_image;
//
//
//    generate_gt_norm_grad(norm_grad_image,syn_image_loc,true,.1,.1,.1);
//    debug_write(norm_grad_image,"norm_grad");
//
//
//    Mesh_data<float> grad_image;
//
//
//    generate_gt_norm_grad(grad_image,syn_image_loc,false,p_rep.pars.dx,p_rep.pars.dy,p_rep.pars.dz);
//    debug_write(grad_image,"grad");
//
//
//    Mesh_data<float> var_gt;
//
//    generate_gt_var(var_gt,syn_image_loc,p_rep.pars);
//
//    debug_write(var_gt,"var_gt");
//
//
//    PartCellStructure<float, uint64_t> pc_struct_perfect;
//    get_apr_perfect(input_img,grad_image,var_gt,p_rep,pc_struct_perfect,analysis_data);
//
//    produce_apr_analysis(input_img, analysis_data, pc_struct_perfect, syn_image_loc, p_rep.pars);

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();




//
//    ParticleDataNew<float, uint64_t> part_new;
//    //flattens format to particle = cell, this is in the classic access/part paradigm
//    part_new.initialize_from_structure(pc_struct);
//
//    //generates the nieghbour structure
//    PartCellData<uint64_t> pc_data;
//    part_new.create_pc_data_new(pc_data);
//
//    pc_data.org_dims = pc_struct.org_dims;
//    part_new.access_data.org_dims = pc_struct.org_dims;
//
//    part_new.particle_data.org_dims = pc_struct.org_dims;
//
//    Mesh_data<float> w_interp_out;
//
//    weigted_interp_img(w_interp_out, pc_data, part_new, part_new.particle_data,false,true);
//
//    debug_write(w_interp_out,"weighted_interp_out_n");
//
//
//    Mesh_data<float> min_img;
//    Mesh_data<float> max_img;
//
//    min_max_interp(min_img,max_img,pc_data,part_new,part_new.particle_data,false);
//
//    debug_write(max_img,"max_img");
//    debug_write(min_img,"min_img");
//
////    for (int i = 0; i < max_img.mesh.size(); ++i) {
////
////        max_img.mesh[i] += min_img.mesh[i];
////        max_img.mesh[i] *= 0.5;
////
////    }
//
//    debug_write(max_img,"avg_img");
//
//    Mesh_data<uint16_t> gt_image;
//    generate_gt_image(gt_image, syn_image_loc);
//
//    std::string name = "we_";
//    compare_E_debug( gt_image,w_interp_out, p_rep.pars, name, analysis_data);
//
//    name = "max_";
//    compare_E_debug( gt_image,max_img, p_rep.pars, name, analysis_data);
//
//    name = "min_";
//    compare_E_debug( gt_image,min_img, p_rep.pars, name, analysis_data);
//
//    Mesh_data<float> interp;
//    interp_img(interp, pc_data, part_new, part_new.particle_data,false);
//
//    name = "interp_";
//    compare_E_debug( gt_image,interp, p_rep.pars, name, analysis_data);
//
//    debug_write(interp,"pc_interp");

    return 0;
}
