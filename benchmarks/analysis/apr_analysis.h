//
// Created by cheesema on 23/01/17.
//

#ifndef PARTPLAY_APR_ANALYSIS_H
#define PARTPLAY_APR_ANALYSIS_H

#include "../../src/algorithm/apr_pipeline.hpp"
#include "AnalysisData.hpp"
#include "MeshDataAF.h"
#include "../../src/io/parameters.h"
#include "SynImageClasses.hpp"
#include "numerics_benchmarks.hpp"
#include "../../src/numerics/misc_numerics.hpp"
#include "../../src/data_structures/APR/APR.hpp"
#include <assert.h>


void calc_information_content(SynImage syn_image,AnalysisData& analysis_data);
void calc_information_content_new(SynImage syn_image,AnalysisData& analysis_data);

template<typename S>
void copy_mesh_data_structures(MeshDataAF<S>& input_syn,Mesh_data<S>& input_img){
    //copy across metadata
    input_img.y_num = input_syn.y_num;
    input_img.x_num = input_syn.x_num;
    input_img.z_num = input_syn.z_num;

    std::swap(input_img.mesh,input_syn.mesh);

}

void get_apr_ground_thruth(SynImage& syn_image,Part_rep& part_rep,PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Calculates Ground Truth APR, using synthetic image generator
    //
    //

//    int interp_type = part_rep.pars.interp_type;
//
//    // COMPUTATIONS
//
//    Mesh_data<float> input_image_float;
//    Mesh_data<float> gradient, variance;
//    Mesh_data<float> interp_img;
//
//    gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
//    part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);
//
//    input_image_float = input_image.to_type<float>();
//    interp_img = input_image.to_type<float>();
//    // After this block, input_image will be freed.
//
//    Part_timer t;
//    t.verbose_flag = false;
//
//    // preallocate_memory
//    Particle_map<float> part_map(part_rep);
//    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
//    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
//    std::vector<Mesh_data<float>> down_sampled_images;
//
//    Mesh_data<float> temp;
//    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
//
//
//    generate_gt_image(Mesh_data<T>& gt_image,SynImage& syn_image)
//
//
//    t.start_timer("whole");
//
//
//    part_rep.timer.start_timer("get_level_3D");
//    get_level_3D(variance, gradient, part_rep, part_map, temp);
//    part_rep.timer.stop_timer();
//
//    // free memory (not used anymore)
//    std::vector<float>().swap(gradient.mesh);
//    std::vector<float>().swap(variance.mesh);
//
//    part_rep.timer.start_timer("pushing_scheme");
//    part_map.pushing_scheme(part_rep);
//    part_rep.timer.stop_timer();
//
//
//    part_rep.timer.start_timer("sample");
//
//    if (interp_type == 0) {
//        part_map.downsample(interp_img);
//        calc_median_filter(part_map.downsampled[part_map.k_max+1]);
//    }
//
//    if (interp_type == 1) {
//        part_map.downsample(input_image_float);
//    } else if (interp_type == 2) {
//        part_map.downsample(interp_img);
//    } else if (interp_type ==3){
//        part_map.downsample(interp_img);
//        calc_median_filter_n(part_map.downsampled[part_map.k_max+1],input_image_float);
//    }
//
//    part_rep.timer.stop_timer();
//
//    part_rep.timer.start_timer("construct_pcstruct");
//
//    pc_struct.initialize_structure(part_map);
//
//    part_rep.timer.stop_timer();
//
//    t.stop_timer();



}




template <typename T>
void bench_get_apr(Mesh_data<T>& input_image,Part_rep& p_rep,PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){
    //
    //
    //  Calculates the APR from image
    //
    //

    p_rep.pars.tol = 0.0005f;

    p_rep.pars.var_scale = 2;

    float k_diff = -3.0f;

    //set lambda


    if(p_rep.pars.lambda == 0) {

        float lambda = expf((-1.0f/0.6161f) * logf((p_rep.pars.var_th/p_rep.pars.noise_sigma) *
                                                   powf(2.0f,k_diff + log2f(p_rep.pars.rel_error))/0.12531f));
        std::cout << lambda << std::endl;

        p_rep.pars.lambda = expf((-1.0f/0.6161f) * logf((p_rep.pars.var_th/p_rep.pars.noise_sigma) *
                                                        powf(2.0f,k_diff + log2f(.05))/0.12531f));

        std::cout << p_rep.pars.lambda << std::endl;

        float lambda_min = 0.05f;
        float lambda_max = 5000;

        p_rep.pars.lambda = std::max(lambda_min,p_rep.pars.lambda);
        p_rep.pars.lambda = std::min(p_rep.pars.lambda,lambda_max);

        float max_var_th = 1.2f * p_rep.pars.noise_sigma * expf(-0.5138f * logf(p_rep.pars.lambda)) *
                           (0.1821f * logf(p_rep.pars.lambda) + 1.522f);
        if (max_var_th > .25*p_rep.pars.var_th){
            float desired_th = 0.1*p_rep.pars.var_th;
            p_rep.pars.lambda = std::max((float)exp((-1.0/0.5138)*log(desired_th/p_rep.pars.noise_sigma)),p_rep.pars.lambda);
            p_rep.pars.var_th_max = .25*p_rep.pars.var_th;

        } else {
            p_rep.pars.var_th_max = .25*p_rep.pars.var_th;

        }
    } else {
        p_rep.pars.var_th_max =  .25*p_rep.pars.var_th;
    }

    if(p_rep.pars.lambda == -1){
        p_rep.pars.lambda = 0;
    }

    std::cout << "Lamda: " << p_rep.pars.lambda << std::endl;

    get_apr(input_image,p_rep,pc_struct,analysis_data);


}

template <typename T>
void bench_get_apr_part_time(Mesh_data<T>& input_image,Part_rep& p_rep,PartCellStructure<float,uint64_t>& pc_struct,AnalysisData& analysis_data){
    //
    //
    //  Calculates the APR from image
    //
    //

    p_rep.pars.tol = 0.0005f;

    p_rep.pars.var_scale = 2;

    float k_diff = -3.0f;

    //set lambda


    if(p_rep.pars.lambda == 0) {

        float lambda = expf((-1.0f/0.6161f) * logf((p_rep.pars.var_th/p_rep.pars.noise_sigma) *
                                                   powf(2.0f,k_diff + log2f(p_rep.pars.rel_error))/0.12531f));
        std::cout << lambda << std::endl;

        p_rep.pars.lambda = expf((-1.0f/0.6161f) * logf((p_rep.pars.var_th/p_rep.pars.noise_sigma) *
                                                        powf(2.0f,k_diff + log2f(.05))/0.12531f));

        std::cout << p_rep.pars.lambda << std::endl;

        float lambda_min = 0.05f;
        float lambda_max = 5000;

        p_rep.pars.lambda = std::max(lambda_min,p_rep.pars.lambda);
        p_rep.pars.lambda = std::min(p_rep.pars.lambda,lambda_max);

        float max_var_th = 1.2f * p_rep.pars.noise_sigma * expf(-0.5138f * logf(p_rep.pars.lambda)) *
                           (0.1821f * logf(p_rep.pars.lambda) + 1.522f);
        if (max_var_th > .25*p_rep.pars.var_th){
            float desired_th = 0.1*p_rep.pars.var_th;
            p_rep.pars.lambda = std::max((float)exp((-1.0/0.5138)*log(desired_th/p_rep.pars.noise_sigma)),p_rep.pars.lambda);
            p_rep.pars.var_th_max = .25*p_rep.pars.var_th;

        } else {
            p_rep.pars.var_th_max = .25*p_rep.pars.var_th;

        }
    } else {
        p_rep.pars.var_th_max =  .25*p_rep.pars.var_th;
    }

    if(p_rep.pars.lambda == -1){
        p_rep.pars.lambda = 0;
    }

    std::cout << "Lamda: " << p_rep.pars.lambda << std::endl;

    get_apr_part_timing(input_image,p_rep,pc_struct,analysis_data);


}


void gen_parameter_pars(SynImage& syn_image,Proc_par& pars,std::string image_name){
    //
    //
    //  Takes in the SynImage model parameters and outputs them to the APR parameter class
    //
    //
    //

    pars.name = image_name;

    pars.dy = syn_image.sampling_properties.voxel_real_dims[0];
    pars.dx = syn_image.sampling_properties.voxel_real_dims[1];
    pars.dz = syn_image.sampling_properties.voxel_real_dims[2];

    pars.psfy = syn_image.PSF_properties.real_sigmas[0];
    pars.psfx = syn_image.PSF_properties.real_sigmas[1];
    pars.psfz = syn_image.PSF_properties.real_sigmas[2];

    pars.ydim = syn_image.real_domain.size[0];
    pars.xdim = syn_image.real_domain.size[1];
    pars.zdim = syn_image.real_domain.size[2];

    pars.noise_sigma = sqrt(syn_image.noise_properties.gauss_var);
    pars.background = syn_image.global_trans.const_shift;

}


template<typename S,typename U>
void compare_reconstruction_to_original(Mesh_data<S>& org_img,PartCellStructure<float,U>& pc_struct,cmdLineOptions& options){
    //
    //  Bevan Cheeseman 2017
    //
    //  Compares an APR pc reconstruction with original image.
    //
    //

    AnalysisData analysis_data;

    Mesh_data<S> rec_img;
    pc_struct.interp_parts_to_pc(rec_img,pc_struct.part_data.particle_data);

    std::string name = "input";
    //get the MSE
    //calc_mse(org_img,rec_img,name,analysis_data);

    debug_write(rec_img,name +"rec_img");
    //compare_E(org_img,rec_img,options,name,analysis_data);

}
template<typename S,typename T>
void compare_E(Mesh_data<S>& org_img,Mesh_data<T>& rec_img,Proc_par& pars,std::string name,AnalysisData& analysis_data){

    Mesh_data<float> variance;

    get_variance(org_img,variance,pars);

    uint64_t z_num_o = org_img.z_num;
    uint64_t x_num_o = org_img.x_num;
    uint64_t y_num_o = org_img.y_num;

    uint64_t z_num_r = rec_img.z_num;
    uint64_t x_num_r = rec_img.x_num;
    uint64_t y_num_r = rec_img.y_num;

    variance.x_num = x_num_r;
    variance.y_num = y_num_r;
    variance.z_num = z_num_r;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    Mesh_data<float> SE;
    //SE.initialize(y_num_o,x_num_o,z_num_o,0);

    double mean = 0;
    double inf_norm = 0;
    uint64_t counter = 0;
    double MSE = 0;

#pragma omp parallel for default(shared) private(j,i,k) reduction(+: MSE) reduction(+: counter) reduction(+: mean) reduction(max: inf_norm)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){
                double val = abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k])/(1.0*variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k]);
                //SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] = 1000*val;

                if(variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k] < 50000) {
                    MSE += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);

                    mean += val;
                    inf_norm = std::max(inf_norm, val);
                    counter++;
                }
            }
        }
    }

    mean = mean/(1.0*counter);
    MSE = MSE/(1*counter);

    //debug_write(SE,name + "E_diff");

    double MSE_var = 0;
    //calculate the variance
    double var = 0;
    counter = 0;

#pragma omp parallel for default(shared) private(j,i,k) reduction(+: var) reduction(+: counter) reduction(+: MSE_var)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                if(variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k] < 50000) {
                    var += pow(pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k],2) - MSE,2);
                    MSE_var += pow(pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k],2) - MSE,2);

                    counter++;
                }
            }
        }
    }


    //debug_write(variance,name + "var");
    //debug_write(rec_img,name +"rec_img");
    //debug_write(org_img,name +"org_img");

    //get variance
    var = var/(1.0*counter);
    double se = 1.96*sqrt(var);
    MSE_var = MSE_var/(1.0*counter);

    analysis_data.add_float_data(name+"_Ediff",mean);
    analysis_data.add_float_data(name+"_Ediff_sd",sqrt(var));
    analysis_data.add_float_data(name+"_Ediff_inf",inf_norm);
    analysis_data.add_float_data(name+"_Ediff_se",se);

    double PSNR = 10*log10(64000.0/MSE);

    analysis_data.add_float_data(name+"_vMSE",MSE);
    analysis_data.add_float_data(name+"_vMSE_sd",sqrt(MSE_var));
    analysis_data.add_float_data(name+"_vPSNR",PSNR);

    float rel_error = pars.rel_error;

    if(pars.lambda == 0) {
        if (inf_norm > rel_error) {
            int stop = 1;
            std::cout << "*********Out of bounds!*********" << std::endl;
           // assert(inf_norm < rel_error);
        }
    }

}





uint64_t get_size_of_pc_struct(PartCellData<uint64_t>& pc_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Get the number of elements in the pc struct data structure
    //

    uint64_t counter = 0;


    for(uint64_t depth = (pc_data.depth_min);depth <= pc_data.depth_max;depth++) {
        //loop over the resolutions of the structure
         for(int i = 0;i < pc_data.data[depth].size();i++){

             counter += pc_data.data[depth][i].size();
         }

    }

    return counter;

}
template<typename T>
void produce_apr_analysis(Mesh_data<T>& input_image,AnalysisData& analysis_data,PartCellStructure<float,uint64_t>& pc_struct,SynImage& syn_image,Proc_par& pars) {
    //
    //  Computes anslysis of part rep dataset
    //
    //  Bevan Cheeseman 2017
    //

    Part_timer timer;

    analysis_data.push_proc_par(pars);


    //////////////////////////////////////
    //
    //  Initialize - data anslysis fields
    //
    /////////////////////////////////////////

    //first check if its first run and the variables need to be set up
    Part_data<int> *check_ref = analysis_data.get_data_ref<int>("num_parts");

    if (check_ref == nullptr) {
    // First experiment need to set up the variables

    //pipeline parameters
        analysis_data.create_int_dataset("num_parts", 0);
        analysis_data.create_int_dataset("num_cells", 0);
        analysis_data.create_int_dataset("num_seed_cells", 0);
        analysis_data.create_int_dataset("num_boundary_cells", 0);
        analysis_data.create_int_dataset("num_filler_cells", 0);
        analysis_data.create_int_dataset("num_ghost_cells", 0);

        analysis_data.create_int_dataset("num_pixels", 0);

        analysis_data.create_float_dataset("apr_comp_size", 0);
        analysis_data.create_float_dataset("apr_full_size", 0);
        analysis_data.create_float_dataset("image_size", 0);
        analysis_data.create_float_dataset("comp_image_size", 0);

        analysis_data.create_float_dataset("information_content", 0);
        analysis_data.create_float_dataset("relerror", 0);

    //set up timing variables

        for (int i = 0; i < timer.timings.size(); i++) {
            analysis_data.create_float_dataset(timer.timing_names[i], 0);
        }


    }

    /////////////////////////////////////////////
    //
    //  APR Size Info
    //
    //////////////////////////////////////////////


    analysis_data.get_data_ref<int>("num_parts")->data.push_back(pc_struct.get_number_parts());
    analysis_data.part_data_list["num_parts"].print_flag = true;

    //calc number of active cells
    int num_seed_cells = 0;
    int num_boundary_cells = 0;
    int num_filler_cells = 0;
    int num_ghost_cells = 0;
    int num_cells = 0;


    analysis_data.get_data_ref<int>("num_cells")->data.push_back(pc_struct.get_number_parts());
    analysis_data.part_data_list["num_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_boundary_cells")->data.push_back(num_boundary_cells);
    analysis_data.part_data_list["num_boundary_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_filler_cells")->data.push_back(num_filler_cells);
    analysis_data.part_data_list["num_filler_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_seed_cells")->data.push_back(num_seed_cells);
    analysis_data.part_data_list["num_seed_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_ghost_cells")->data.push_back(num_ghost_cells);
    analysis_data.part_data_list["num_ghost_cells"].print_flag = true;

    analysis_data.get_data_ref<int>("num_pixels")->data.push_back(
            input_image.y_num * input_image.x_num * input_image.z_num);
    analysis_data.part_data_list["num_pixels"].print_flag = true;

    std::cout << (1.0*input_image.y_num * input_image.x_num * input_image.z_num)/(1.0*pc_struct.get_number_parts()) << std::endl;

    ////////////////////////////////////////////////////////////////////
    //
    //  Timing Information
    //
    ////////////////////////////////////////////////////////////////////

    for (int i = 0; i < timer.timings.size(); i++) {
        analysis_data.get_data_ref<float>(timer.timing_names[i])->data.push_back(timer.timings[i]);
        analysis_data.part_data_list[timer.timing_names[i]].print_flag = true;
    }

    //////////////////////////////////////////////////////////////////////
    //
    //  Run segmentation Benchmark
    //
    //////////////////////////////////////////////////////////////////

    if(analysis_data.segmentation_mesh) {

        run_segmentation_benchmark_mesh(pc_struct, analysis_data);

    }

    if(analysis_data.segmentation_parts) {

        run_segmentation_benchmark_parts(pc_struct, analysis_data);

    }


    if(analysis_data.filters_mesh){

        run_filter_benchmarks_mesh(pc_struct, analysis_data);

    }

    if(analysis_data.filters_parts){

        run_filter_benchmarks_parts(pc_struct, analysis_data);

    }

    if(analysis_data.segmentation_eval) {
        //evaluate_segmentation(pc_struct,analysis_data,syn_image);

        //evaluate_filters(pc_struct,analysis_data,syn_image,input_image);

        Part_rep p_rep;
        p_rep.pars = pars;


        run_real_segmentation(pc_struct,analysis_data, pars);

        run_ray_cast(pc_struct,analysis_data,input_image,pars);

        real_adaptive_grad(pc_struct,analysis_data,input_image, pars);


       // evaluate_enhancement(pc_struct,analysis_data,syn_image,input_image,p_rep);
    }

    if(analysis_data.filters_eval) {

       // evaluate_filters(pc_struct,analysis_data,syn_image,input_image);
        evaluate_filters_guassian(pc_struct,analysis_data,syn_image,input_image,.5);
        evaluate_filters_guassian(pc_struct,analysis_data,syn_image,input_image, 2);

        evaluate_adaptive_smooth(pc_struct,analysis_data,syn_image,input_image);
        evaluate_adaptive_grad(pc_struct,analysis_data,syn_image,input_image);

        //evaluate_filters_log(pc_struct,analysis_data,syn_image,input_image);
    }

    ////////////////////////////////////////////////////////////////////
    //
    //  File Size Information
    //
    /////////////////////////////////////////////////////////////////////

    //write apr and image to file

    if(analysis_data.file_size) {

        std::string file_name = pars.output_path + pars.name;

        write_apr_pc_struct(pc_struct, pars.output_path, pars.name);

        write_image_tiff(input_image, file_name + ".tif");

        analysis_data.get_data_ref<float>("apr_full_size")->data.push_back(
                GetFileSize(file_name + "_pcstruct_part.h5"));
        analysis_data.part_data_list["apr_full_size"].print_flag = true;

        analysis_data.get_data_ref<float>("image_size")->data.push_back(GetFileSize(file_name + ".tif"));
        analysis_data.part_data_list["image_size"].print_flag = true;



        //produce the compressed image file to get baseline
        std::string compress_file_name = file_name + ".bz2";
        std::string system_string = "pbzip2 -c -9 <" + file_name + ".tif" + ">" + compress_file_name;
        system(system_string.c_str());

        analysis_data.get_data_ref<float>("comp_image_size")->data.push_back(GetFileSize(compress_file_name));
        analysis_data.part_data_list["comp_image_size"].print_flag = true;
    }

    analysis_data.get_data_ref<float>("relerror")->data.push_back(pars.rel_error);
    analysis_data.part_data_list["relerror"].print_flag = true;

    ///////////////////////////
    //
    //  Compute Reconstruction Quality Metrics
    //
    ///////////////////////////

    timer.verbose_flag = false;

    timer.start_timer("Image Quality");

    Mesh_data<float> rec_img;
    std::string name;

    if(analysis_data.quality_metrics_gt || analysis_data.quality_metrics_input) {

        //get the pc reconstruction

        pc_struct.interp_parts_to_pc(rec_img, pc_struct.part_data.particle_data);


    }


    if(analysis_data.debug == true){

        pc_struct.interp_parts_to_pc(rec_img, pc_struct.part_data.particle_data);
        write_image_tiff(rec_img, pars.output_path + pars.name + "_rec.tif");

        write_image_tiff(input_image, pars.output_path + pars.name + "_debug.tif");

        write_apr_pc_struct(pc_struct, pars.output_path, pars.name);

        write_apr_full_format(pc_struct,pars.output_path, pars.name);

        Mesh_data<float> variance;

        get_variance(input_image,variance,pars);

        debug_write(variance,"var");

        Mesh_data<uint8_t> k_img;

        interp_depth_to_mesh(k_img,pc_struct);

        write_image_tiff(k_img, pars.output_path + pars.name + "_k.tif");

        Mesh_data<T> gt_image;
        generate_gt_image(gt_image, syn_image);

        name = "debug";
        compare_E_debug( gt_image,rec_img, pars, name, analysis_data);



    }


    if(analysis_data.quality_true_int) {

        //Generate clean gt image
        Mesh_data<uint16_t> gt_image;
        generate_gt_image(gt_image, syn_image);

        Part_rep p_rep;
        p_rep.pars = pars;

        APR<float> t_apr(pc_struct);

        ExtraPartCellData<float> true_parts;
        true_parts.initialize_structure_parts(t_apr.y_vec);

        Particle_map<uint16_t> part_map(p_rep);

        part_map.downsample(gt_image);


        int z_, x_, j_, y_, i, k;

        for (uint64_t depth = (t_apr.y_vec.depth_min); depth <= t_apr.y_vec.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = t_apr.y_vec.x_num[depth];
            const unsigned int z_num_ = t_apr.y_vec.z_num[depth];

            const float step_size_x = pow(2, t_apr.y_vec.depth_max - depth);
            const float step_size_y = pow(2, t_apr.y_vec.depth_max - depth);
            const float step_size_z = pow(2, t_apr.y_vec.depth_max - depth);

            for (z_ = 0; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = 0; x_ < x_num_; x_++) {

                    const unsigned int pc_offset = x_num_ * z_ + x_;

                    for (j_ = 0; j_ < t_apr.y_vec.data[depth][pc_offset].size(); j_++) {


                        const int y = t_apr.y_vec.data[depth][pc_offset][j_];

                        const float y_actual = floor((y+0.5) * step_size_y);
                        const float x_actual = floor((x_+0.5) * step_size_x);
                        const float z_actual = floor((z_+0.5) * step_size_z);

                        true_parts.data[depth][pc_offset][j_] = part_map.downsampled[depth](y_actual,x_actual,z_actual);

                    }
                }
            }
        }

        Mesh_data<uint16_t> true_int_m;

        t_apr.init_pc_data();

        interp_img(true_int_m, t_apr.pc_data, t_apr.part_new, true_parts,true);


        name = "true";
        compare_E(true_int_m, gt_image, pars, name, analysis_data);

        calc_mse(true_int_m, gt_image, name, analysis_data);



    }




    if(analysis_data.quality_metrics_gt) {


        //Generate clean gt image
        Mesh_data<uint16_t> gt_image;
        generate_gt_image(gt_image, syn_image);


        name = "orggt";
        compare_E(input_image, gt_image, pars, name, analysis_data);

        calc_mse(input_image, gt_image, name, analysis_data);

        name = "recgt";

        //get the MSE
        calc_mse(gt_image, rec_img, name, analysis_data);
        compare_E(gt_image, rec_img, pars, name, analysis_data);
    }

    if(analysis_data.quality_metrics_input){

        name = "input";

        //get the MSE
        calc_mse(input_image, rec_img, name, analysis_data);
        compare_E(input_image, rec_img, pars, name, analysis_data);

    }

    timer.stop_timer();

    timer.start_timer("information_content");

    ///////////////////////////////////////
    //
    //  Calc information content
    //
    /////////////////////////////////////////////

    if(analysis_data.information_content) {

        calc_information_content(syn_image, analysis_data);

    }
    timer.stop_timer();

}
template<typename T>
void produce_apr_analysis(Mesh_data<T>& input_image,AnalysisData& analysis_data,PartCellStructure<float,uint64_t>& pc_struct,Proc_par& pars){
    //
    //  Bevan Cheeseman 2017
    //
    //  Interface for running on real data without syn image benchmarks
    //
    //

    SynImage syn_image_temp;

    //these cannot be run without a syntehtic image ground truth
    analysis_data.quality_metrics_gt = false;
    analysis_data.information_content = false;

    produce_apr_analysis(input_image,analysis_data,pc_struct,syn_image_temp,pars);

}

void calc_information_content(SynImage syn_image,AnalysisData& analysis_data){
    //
    //  Bevan Cheeseman 2016
    //
    //  Information content of a synthetic image.
    //
    //  Computes as the integral of the normalized gradient normalized by voxel size.
    //

    MeshDataAF<float> info_x;
    MeshDataAF<float> info_y;
    MeshDataAF<float> info_z;

    syn_image.PSF_properties.type = "gauss_dx";
    syn_image.noise_properties.noise_type = "none";
    syn_image.global_trans.gt_ind = false;
    syn_image.PSF_properties.normalize = true;

    syn_image.generate_syn_image(info_x);

    info_x.transfer_from_arrayfire();
    info_x.free_arrayfire();

    //af::sync();
    af::deviceGC();

    syn_image.PSF_properties.type = "gauss_dy";
    syn_image.noise_properties.noise_type = "none";
    syn_image.global_trans.gt_ind = false;
    syn_image.PSF_properties.normalize = true;

    syn_image.generate_syn_image(info_y);

    info_y.transfer_from_arrayfire();
    info_y.free_arrayfire();

    //af::sync();
    af::deviceGC();

    syn_image.PSF_properties.type = "gauss_dz";
    syn_image.noise_properties.noise_type = "none";
    syn_image.global_trans.gt_ind = false;
    syn_image.PSF_properties.normalize = true;

    syn_image.generate_syn_image(info_z);

    info_z.transfer_from_arrayfire();
    info_z.free_arrayfire();

    //af::sync();
    af::deviceGC();

    af::array info_;

    info_x.transfer_to_arrayfire();
    info_y.transfer_to_arrayfire();
    info_z.transfer_to_arrayfire();

    info_x.af_mesh = sum(sum(sum(sqrt(pow(info_x.af_mesh,2)+pow(info_y.af_mesh,2)+pow(info_z.af_mesh,2)))));

    info_y.free_arrayfire();
    info_z.free_arrayfire();

    float info_content = info_x.af_mesh.scalar<float>()*((syn_image.sampling_properties.voxel_real_dims[0]*syn_image.sampling_properties.voxel_real_dims[1]*syn_image.sampling_properties.voxel_real_dims[2]));

    analysis_data.add_float_data("info",info_content);

}

#endif //PARTPLAY_ANALYZE_APR_H
