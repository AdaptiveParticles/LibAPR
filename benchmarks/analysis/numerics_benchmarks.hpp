//
// Created by cheesema on 27/01/17.
//

#ifndef PARTPLAY_NUMERICS_BENCHMARKS_HPP
#define PARTPLAY_NUMERICS_BENCHMARKS_HPP

#include "../../src/data_structures/APR/APR.hpp"

#include "benchmarks/development/old_numerics/graph_cut_seg.hpp"
#include "benchmarks/development/old_numerics/filter_numerics.hpp"
#include "../../src/numerics/ray_cast.hpp"
#include "benchmarks/development/old_numerics/enhance_parts.hpp"

template<typename S,typename T>
void compare_E_debug(MeshData<S>& org_img,MeshData<T>& rec_img,Proc_par& pars,std::string name,AnalysisData& analysis_data){

    MeshData<float> variance;

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

    MeshData<float> SE;
    SE.initialize(y_num_o,x_num_o,z_num_o,0);

    double mean = 0;
    double inf_norm = 0;
    uint64_t counter = 0;
    double MSE = 0;


    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){
                double val = abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k])/(1.0*variance.mesh[j*x_num_r*y_num_r + i*y_num_r + k]);
                SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] = 1000*val;

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

    debug_write(SE,name + "E_diff");

    float rel_error = pars.rel_error;

    std::cout << name << " E: " << pars.rel_error << " inf_norm: " << inf_norm << std::endl;

    if(pars.lambda == 0) {
        if (inf_norm > rel_error) {
            int stop = 1;
            std::cout << "*********Out of bounds!*********" << std::endl;
            //assert(inf_norm < rel_error);
        }
    }

    for (int l = 0; l < SE.mesh.size(); ++l) {
        if(SE.mesh[l] < 1000*rel_error){
            SE.mesh[l] = 0;

        }
    }

    debug_write(SE,name + "E_break_bound");

}
template<typename S,typename T>
void calc_mse_part_locations(MeshData<S>& org_img,MeshData<T>& rec_img,std::string name,AnalysisData& analysis_data,PartCellStructure<float,uint64_t>& pc_struct) {
//
    //
    //  Bevan Cheeseman 2017
    //
    //  Computes the MSE and PSNR at particle locations
    //
    //
    //
    //


    uint64_t z_num_o = org_img.z_num;
    uint64_t x_num_o = org_img.x_num;
    uint64_t y_num_o = org_img.y_num;

    uint64_t z_num_r = rec_img.z_num;
    uint64_t x_num_r = rec_img.x_num;
    uint64_t y_num_r = rec_img.y_num;

    double MSE =0;


    uint8_t k,type;
    uint16_t x_c;
    uint16_t y_c;
    uint16_t z_c;
    uint16_t Ip;


    //initialize
    uint64_t node_val_part;
    uint64_t y_coord;
    int x_;
    int z_;

    uint64_t j_;
    uint64_t status;
    uint64_t curr_key=0;
    uint64_t part_offset=0;

    uint64_t p;

    uint64_t counter = 0;


    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){

        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];

        for(z_ = 0;z_ < z_num_;z_++){

            curr_key = 0;

            pc_struct.part_data.access_data.pc_key_set_depth(curr_key,i);
            pc_struct.part_data.access_data.pc_key_set_z(curr_key,z_);

            for(x_ = 0;x_ < x_num_;x_++){

                pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();

                y_coord = 0;

                for(j_ = 0;j_ < j_num;j_++){


                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];

                    if (!(node_val_part&1)){
                        //get the index gap node
                        y_coord++;

                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);

                        //neigh_keys.resize(0);
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);

                        pc_struct.part_data.access_data.pc_key_set_status(curr_key,status);

                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            pc_struct.part_data.access_data.pc_key_set_partnum(curr_key,p);


                            pc_struct.part_data.access_data.get_coordinates_part_full(y_coord,curr_key,x_c,z_c,y_c,k,type);

                            z_c = z_c/2;
                            x_c = x_c/2;
                            y_c = y_c/2;

                            MSE += pow(org_img(y_c,x_c,z_c) - rec_img(y_c,x_c,z_c),2);
                            counter++;

                        }

                    } else {

                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }

                }

            }

        }
    }

    MSE = MSE/(1.0*counter);

    double PSNR = 10*log10(64000.0/MSE);

    //std::cout << PSNR << std::endl;


}


template<typename S,typename T>
void calc_mse(MeshData<S>& org_img,MeshData<T>& rec_img,std::string name,AnalysisData& analysis_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Calculates the mean squared error
    //

    uint64_t z_num_o = org_img.z_num;
    uint64_t x_num_o = org_img.x_num;
    uint64_t y_num_o = org_img.y_num;

    uint64_t z_num_r = rec_img.z_num;
    uint64_t x_num_r = rec_img.x_num;
    uint64_t y_num_r = rec_img.y_num;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    double MSE = 0;
    double L1 = 0;

#pragma omp parallel for default(shared) private(j,i,k) reduction(+: MSE)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                MSE += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);
                L1 += abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k]);
            }
        }
    }

    MSE = MSE/(z_num_o*x_num_o*y_num_o*1.0);
    L1 = L1/(z_num_o*x_num_o*y_num_o*1.0);

    // MeshData<S> SE;
    //SE.initialize(y_num_o,x_num_o,z_num_o,0);

    double var = 0;
    double counter = 0;
#pragma omp parallel for default(shared) private(j,i,k) reduction(+: var) reduction(+: counter)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                //SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);
                var += pow(pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2) - MSE,2);
                counter++;
            }
        }
    }

    var = var/(counter*1.0);

    double se = sqrt(var)*1.96;

    double PSNR = 10*log10(64000.0/MSE);

    float sd = sqrt(var);

    //commit results
    analysis_data.add_float_data(name + "_L1",L1);
    analysis_data.add_float_data(name + "_MSE",MSE);
    analysis_data.add_float_data(name +"_MSE_SE",se);
    analysis_data.add_float_data(name +"_PSNR",PSNR);
    analysis_data.add_float_data(name +"_MSE_sd",sd);


}

template<typename S,typename T>
void calc_mse_debug(MeshData<S>& org_img,MeshData<T>& rec_img,std::string name,AnalysisData& analysis_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Calculates the mean squared error
    //

    uint64_t z_num_o = org_img.z_num;
    uint64_t x_num_o = org_img.x_num;
    uint64_t y_num_o = org_img.y_num;

    uint64_t z_num_r = rec_img.z_num;
    uint64_t x_num_r = rec_img.x_num;
    uint64_t y_num_r = rec_img.y_num;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    double MSE = 0;

    MeshData<S> SE;
    SE.initialize(y_num_o,x_num_o,z_num_o,0);

#pragma omp parallel for default(shared) private(j,i,k) reduction(+: MSE)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                MSE += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);
                SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] =  abs(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k]);
            }
        }
    }

    MSE = MSE/(z_num_o*x_num_o*y_num_o*1.0);




    double var = 0;
    double counter = 0;
#pragma omp parallel for default(shared) private(j,i,k) reduction(+: var) reduction(+: counter)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                //SE.mesh[j*x_num_o*y_num_o + i*y_num_o + k] += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);
                var += pow(pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k],2) - MSE,2);
                counter++;
            }
        }
    }

    var = var/(counter*1.0);

    double se = sqrt(var)*1.96;

    double PSNR = 10*log10(64000.0/MSE);

    float sd = sqrt(var);

    //commit results
    analysis_data.add_float_data(name + "_MSE",MSE);
    analysis_data.add_float_data(name +"_MSE_SE",se);
    analysis_data.add_float_data(name +"_PSNR",PSNR);
    analysis_data.add_float_data(name +"_MSE_sd",sd);

    std::cout << name << ": " << PSNR << std::endl;

    debug_write(SE,name + "_MSE");

}


template <typename T>
void generate_gt_image(MeshData<T>& gt_image,SynImage& syn_image){
    //get a clean image

    MeshDataAF<T> gen_img;

    std::string prev_noise =  syn_image.noise_properties.noise_type;

    syn_image.noise_properties.noise_type = "none";

    syn_image.generate_syn_image(gen_img);

    gt_image.y_num = gen_img.y_num;
    gt_image.x_num = gen_img.x_num;
    gt_image.z_num = gen_img.z_num;

    //copy accross
    gt_image.mesh = gen_img.mesh;

    syn_image.noise_properties.noise_type = prev_noise;

}
template <typename T>
void generate_gt_norm_grad(MeshData<T>& gt_image,SynImage syn_image,bool normalize,float hx,float hy,float hz){
    //get a clean image

    MeshDataAF<T> gen_imgx;

    //for (int i = 0; i < syn_image.real_objects.size(); ++i) {
      //  syn_image.real_objects[i].int_scale = 1000;
    //}

    syn_image.noise_properties.noise_type = "none";

    syn_image.PSF_properties.type = "gauss_dx";
    //syn_image.PSF_properties.type = "var";
    syn_image.noise_properties.noise_type = "none";
    syn_image.global_trans.gt_ind = false;
    syn_image.PSF_properties.normalize = normalize;

    syn_image.generate_syn_image(gen_imgx);

    MeshDataAF<T> gen_imgy;

    //for (int i = 0; i < syn_image.real_objects.size(); ++i) {
    //  syn_image.real_objects[i].int_scale = 1000;
    //}

    syn_image.noise_properties.noise_type = "none";

    syn_image.PSF_properties.type = "gauss_dy";
    //syn_image.PSF_properties.type = "var";
    syn_image.noise_properties.noise_type = "none";
    syn_image.global_trans.gt_ind = false;
    syn_image.PSF_properties.normalize = normalize;

    syn_image.generate_syn_image(gen_imgy);

    MeshDataAF<T> gen_imgz;

    //for (int i = 0; i < syn_image.real_objects.size(); ++i) {
    //  syn_image.real_objects[i].int_scale = 1000;
    //}

    syn_image.noise_properties.noise_type = "none";

    syn_image.PSF_properties.type = "gauss_dz";
    //syn_image.PSF_properties.type = "var";
    syn_image.noise_properties.noise_type = "none";
    syn_image.global_trans.gt_ind = false;
    syn_image.PSF_properties.normalize = normalize;

    syn_image.generate_syn_image(gen_imgz);


    gt_image.y_num = gen_imgx.y_num;
    gt_image.x_num = gen_imgx.x_num;
    gt_image.z_num = gen_imgx.z_num;

    //copy accross
    gt_image.mesh = gen_imgx.mesh;


    for (int i = 0; i < gen_imgz.mesh.size(); ++i) {
        gt_image.mesh[i] = sqrt(pow(gen_imgx.mesh[i]/hx,2.0) + pow(gen_imgy.mesh[i]/hy,2.0) + pow(gen_imgz.mesh[i]/hz,2.0));
    }


    //copy accross
    //gt_image.mesh = gen_img.mesh;

}

template <typename T>
void generate_gt_var(MeshData<T>& gt_image,MeshData<T>& var_out,SynImage syn_image,Proc_par& pars){
    //get a clean image


    MeshData<float> norm_grad_image;

    generate_gt_norm_grad(norm_grad_image,syn_image,true,pars.dx,pars.dy,pars.dz);
    //debug_write(norm_grad_image,"norm_grad");

    MeshData<float> grad_image;

    generate_gt_norm_grad(grad_image,syn_image,false,pars.dx,pars.dy,pars.dz);
   // debug_write(grad_image,"grad");


   // MeshData<float> grad_image;

    //grad_image.initialize(gt_image.y_num,gt_image.x_num,gt_image.z_num,0);

   // Part_rep p_rep(gt_image.y_num,gt_image.x_num,gt_image.z_num);
   // p_rep.pars = pars;

   // p_rep.pars.lambda = -1;

    //generate_gt_norm_grad(norm_grad_image,syn_image,true,pars.dx,pars.dy,pars.dz);
   // get_gradient_3D(p_rep, gt_image, grad_image);

    float factor = 1.0;


    MeshData<float> norm;

    norm.initialize(grad_image.y_num,grad_image.x_num,grad_image.z_num,0);

    //std::cout << round(1000*syn_image.scaling_factor/syn_image.object_templates[0].max_sampled_int) << std::endl;

    for (int i = 0; i < norm.mesh.size(); ++i) {
        if(grad_image.mesh[i] > 1) {
            norm.mesh[i] = round(factor*1000*syn_image.scaling_factor/syn_image.object_templates[0].max_sampled_int) * grad_image.mesh[i] / norm_grad_image.mesh[i];
        } else {
            norm.mesh[i] = 64000;
        }
    }

    var_out.y_num = norm.y_num;
    var_out.x_num = norm.x_num;
    var_out.z_num = norm.z_num;

    //copy accross
    var_out.mesh = norm.mesh;

}

void run_segmentation_benchmark_mesh(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Runs the graph cuts segmentation benchmarks
    //

    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,2000,1,1,2,2,2,3,0,0};

    MeshData<uint8_t> seg_mesh;

    //memory on this machine can't handle anything bigger
    if(pc_struct.org_dims[0] <= 550){
        std::cout << "gc_seg_mesh" << std::endl;
        std:: cout << pc_struct.org_dims[0] << std::endl;
        calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc,analysis_data);
        std::cout << "gc_seg_mesh_complete" << std::endl;
    }

}
template <typename T>
void generate_gt_seg(MeshData<T>& gt_image,SynImage& syn_image){
    //get a clean image

    SynImage syn_image_seg = syn_image;

    MeshDataAF<T> gen_img;

    syn_image_seg.noise_properties.noise_type = "none";
    syn_image_seg.PSF_properties.type = "none";
    syn_image_seg.global_trans.gt_ind = false;

    syn_image_seg.generate_syn_image(gen_img);

    gt_image.y_num = gen_img.y_num;
    gt_image.x_num = gen_img.x_num;
    gt_image.z_num = gen_img.z_num;

    //copy accross
    gt_image.mesh = gen_img.mesh;


}
float compute_dice_coeff(MeshData<uint16_t>& gt,MeshData<uint8_t>& seg){
    //
    //  Bevan Cheeseman 2017
    //
    //  Calculating the Dice co-efficient for the segmentation
    //
    //

    uint64_t correct = 0;
    uint64_t gt_count = 0;
    uint64_t seg_count = 0;


    for (int i = 0; i < gt.z_num; ++i) {
        for (int j = 0; j < gt.x_num; ++j) {
            for (int k = 0; k < gt.y_num; ++k) {

                int gt_val = (gt(i,j,k) > 0);
                int seg_val = (seg(i,j,k) > 0);

                gt_count += gt_val;
                seg_count += seg_val;
                correct += gt_val*seg_val;


            }
        }
    }


    return 2.0*correct/(1.0*gt_count + 1.0*seg_count);


}


void evaluate_segmentation(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image){

    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,100,2,2,2,2,2,3,0,0};

    MeshData<uint8_t> seg_mesh;

    //memory on this machine can't handle anything bigger
    if(pc_struct.org_dims[0] <= 550){
        calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc);
    }

    ExtraPartCellData<uint8_t> seg_parts;

    if(pc_struct.get_number_parts() <= pow(550,3)) {

        calc_graph_cuts_segmentation(pc_struct, seg_parts, parameters_nuc);
    }

    MeshData<uint8_t> seg_img;

    pc_struct.interp_parts_to_pc(seg_img,seg_parts);

    MeshData<uint16_t> gt_image;
    generate_gt_seg(gt_image,syn_image);

    float dice_mesh = compute_dice_coeff(gt_image,seg_mesh);
    float dice_parts = compute_dice_coeff(gt_image,seg_img);

    analysis_data.add_float_data("dice_mesh",dice_mesh);
    analysis_data.add_float_data("dice_parts",dice_parts);


    MeshData<uint8_t> seg_img_new;

    ExtraPartCellData<uint8_t> seg_parts_new;

    std::array<float,13> parameters_new = {0,2,2,2,2,2,3,1,1,1,60000,0};

    if(pc_struct.get_number_parts() <= pow(500,3)) {

        calc_graph_cuts_segmentation_new(pc_struct, seg_parts_new,analysis_data,parameters_new);
    }


    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);


    interp_img(seg_img_new, pc_data, part_new, seg_parts_new,true);

    float dice_parts_new = compute_dice_coeff(gt_image,seg_img_new);


    if(analysis_data.debug) {

        debug_write(seg_mesh, "seg_mesh");
        debug_write(seg_img, "seg_parts");
        debug_write(gt_image, "seg_gt");
        debug_write(seg_img_new, "seg_img_new");

        std::cout << "Dice m: " << dice_mesh << std::endl;
        std::cout << "Dice p: " << dice_parts << std::endl;
        std::cout << "Dice p_n: " << dice_parts_new << std::endl;
    }

}

void run_segmentation_benchmark_parts(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Runs the graph cuts segmentation benchmarks
    //

    ExtraPartCellData<uint8_t> seg_parts;

    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,2000,1,1,2,2,2,3,0,0};

    if(pc_struct.get_number_parts() <= pow(550,3)) {

        calc_graph_cuts_segmentation(pc_struct, seg_parts, parameters_nuc, analysis_data);
    }
}


void run_filter_benchmarks_mesh(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Runs the filtering and neighbour access benchmarks
    //

    float num_repeats = 5;

    //Get neighbours (linear)

    Part_timer timer;

    timer.verbose_flag = false;

    timer.start_timer("lin");

    //pixels
    pixels_linear_neigh_access(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],num_repeats,analysis_data);

    timer.stop_timer();

    timer.start_timer("random");

    pixel_neigh_random(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],analysis_data);

    timer.stop_timer();

    // Filtering

    uint64_t filter_offset = 10;


    std::vector<float> filter;
    filter.resize(2*filter_offset + 1,1.0/(2*filter_offset + 1));

    num_repeats = 1;

    MeshData<float> output_image;

    MeshData<float> input_image;

    timer.start_timer("filter");

    pc_struct.interp_parts_to_pc(input_image,pc_struct.part_data.particle_data);

    output_image =  pixel_filter_full(input_image,filter,num_repeats,analysis_data);

    timer.stop_timer();

    ExtraPartCellData<float> filter_output_mesh;

    timer.start_timer("full_apr_filter");

    filter_output_mesh = filter_apr_input_img<float>(input_image,pc_struct,filter,analysis_data,num_repeats);

    timer.stop_timer();

}

void run_filter_benchmarks_parts(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Runs the filtering and neighbour access benchmarks
    //

    float num_repeats = 10;

    //Get neighbours (linear)

    //particles
   // particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);

    //Get neighbours (random access)

    particle_random_access(pc_struct,analysis_data);

    // Filtering

    uint64_t filter_offset = 10;

    std::vector<float> filter;
    filter.resize(2*filter_offset + 1,1.0/(2*filter_offset + 1));

    ExtraPartCellData<float> filter_output;

  //  filter_output = filter_apr_by_slice<float>(pc_struct,filter,analysis_data,num_repeats);

}

template<typename T>
void remove_boundary(MeshData<T>& img,int sz){
    //
    //  Bevan Cheeseman 2017
    //
    //  Because I haven't dealt with boundary conditions properly for the filter, I want to remove those involved in boundary points from all computations and displaying results
    //

    unsigned int new_y = img.y_num - 2*sz;
    unsigned int new_x = img.x_num - 2*sz;
    unsigned int new_z = img.z_num - 2*sz;

    MeshData<T> new_mesh;
    new_mesh.initialize(new_y,new_x,new_z,0);


    for (int i = 0; i < new_z ; ++i) {
        for (int j = 0; j < new_x; ++j) {
            for (int k = 0; k < new_y; ++k) {
                new_mesh(i,j,k) = img(i+sz,j+sz,k+sz);
            }
        }
    }

    std::swap(img,new_mesh);


}

template<typename T>
void evaluate_enhancement(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,MeshData<T>& input_image,Part_rep& part_rep){
    //
    //  Benchmark Code to Evaluate Enhancement
    //
    //
    //


    MeshData<float> input_image_float;

    //transfer across to float
    input_image_float.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(input_image.mesh.begin(),input_image.mesh.end(),input_image_float.mesh.begin());

    PartCellStructure<float,uint64_t> pc_struct_new = compute_guided_apr(input_image_float,pc_struct,part_rep);

    MeshData<float> output;

    pc_struct_new.interp_parts_to_pc(output,pc_struct_new.part_data.particle_data);

    debug_write(output,"new_apr");

    MeshData<uint8_t> k_img;
    interp_depth_to_mesh(k_img,pc_struct);
    debug_write(k_img,"k_debug_old");

    interp_depth_to_mesh(k_img,pc_struct_new);
    debug_write(k_img,"k_debug_new");




};




template<typename T>
void evaluate_filters(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,MeshData<T>& input_image){
    //
    //  Bevan Cheeseman 2017
    //
    //  Evaluate the accuracy of the filters
    //
    //

    //filter set up
    uint64_t filter_offset = 8;

    std::vector<float> filter;
    filter.resize(2*filter_offset + 1,1.0/(2*filter_offset + 1));

    //filter = create_gauss_filter<float>(filter_offset,1.5);
    filter = create_dog_filter<float>(filter_offset,1.5,3);

    float num_repeats = 1;

    //Generate clean gt image
    MeshData<uint16_t> gt_image;
    generate_gt_image(gt_image, syn_image);

    MeshData<float> gt_image_f;
    gt_image_f.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(gt_image.mesh.begin(),gt_image.mesh.end(),gt_image_f.mesh.begin());

    MeshData<float> gt_output;

    gt_output =  pixel_filter_full(gt_image_f,filter,num_repeats,analysis_data);

    remove_boundary(gt_output,2*filter_offset +1);

    debug_write(gt_output,"filt_gt");

    // Compute Full Filter on input
    MeshData<float> output_image_org;

    MeshData<float> input_image_org;

    //transfer across to float
    input_image_org.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(input_image.mesh.begin(),input_image.mesh.end(),input_image_org.mesh.begin());

    output_image_org =  pixel_filter_full(input_image_org,filter,num_repeats,analysis_data);

    remove_boundary(output_image_org,2*filter_offset +1);

    calc_mse(gt_output,output_image_org,"filt_org",analysis_data);

    debug_write(output_image_org,"filt_org");

    // Compute Full Filter on APR reconstruction
    MeshData<float> output_image_rec;

    MeshData<float> input_image_rec;

    pc_struct.interp_parts_to_pc(input_image_rec,pc_struct.part_data.particle_data);

    output_image_rec =  pixel_filter_full(input_image_rec,filter,num_repeats,analysis_data);

    remove_boundary(output_image_rec,2*filter_offset +1);

    calc_mse(gt_output,output_image_rec,"filt_rec",analysis_data);

    debug_write(output_image_rec,"filt_rec");

    //Compute APR Filter

    ExtraPartCellData<float> filter_output;

    filter_output = filter_apr_by_slice<float>(pc_struct,filter,analysis_data,num_repeats);

    MeshData<float> output_image_apr;

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    interp_img(output_image_apr, pc_data, part_new, filter_output);

    remove_boundary(output_image_apr,2*filter_offset +1);

    calc_mse(gt_output,output_image_apr,"filt_apr",analysis_data);

    debug_write(output_image_apr,"filt_apr");

    if(analysis_data.debug){
       //
       //
       //
       //
    }

}



template<typename T>
void evaluate_filters_guassian(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,MeshData<T>& input_image,float sigma){
    //
    //  Bevan Cheeseman 2017
    //
    //  Evaluate the accuracy of the filters
    //
    //

    //filter set up


    std::vector<float> filter;

    uint64_t filter_offset = 8;

    //filter = create_dog_filter<float>(filter_offset,1.5,3);
    filter = create_gauss_filter<float>(sigma,filter_offset);


    float num_repeats = 1;

    //Generate clean gt image
    MeshData<uint16_t> gt_image;
    generate_gt_image(gt_image, syn_image);


    MeshData<float> gt_image_f;
    gt_image_f.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(gt_image.mesh.begin(),gt_image.mesh.end(),gt_image_f.mesh.begin());

    MeshData<float> gt_output;

    gt_output =  pixel_filter_full_mult(gt_image_f,filter,filter,filter,num_repeats,analysis_data);

    remove_boundary(gt_output,2*filter_offset +2);

    //debug_write(gt_output,"gauss_filt_gt_" + std::to_string(sigma));

    // Compute Full Filter on input
    MeshData<float> output_image_org;

    MeshData<float> input_image_org;

    //transfer across to float
    input_image_org.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(input_image.mesh.begin(),input_image.mesh.end(),input_image_org.mesh.begin());

    output_image_org =  pixel_filter_full_mult(input_image_org,filter,filter,filter,num_repeats,analysis_data);

    remove_boundary(output_image_org,2*filter_offset +2);

    calc_mse(gt_output,output_image_org,"gauss_filt_org" + std::to_string(sigma),analysis_data);
    calc_mse_part_locations(gt_output,output_image_org,"gauss_filt_org" + std::to_string(sigma),analysis_data,pc_struct);

    //debug_write(output_image_org,"gauss_filt_org"  + std::to_string(sigma));

    // Compute Full Filter on APR reconstruction
    MeshData<float> output_image_rec;

    MeshData<float> input_image_rec;

    pc_struct.interp_parts_to_pc(input_image_rec,pc_struct.part_data.particle_data);

    output_image_rec =  pixel_filter_full_mult(input_image_rec,filter,filter,filter,num_repeats,analysis_data);

    remove_boundary(output_image_rec,2*filter_offset +2);

    calc_mse(gt_output,output_image_rec,"gauss_filt_rec" + std::to_string(sigma),analysis_data);
    calc_mse_part_locations(gt_output,output_image_rec,"gauss_filt_rec" + std::to_string(sigma),analysis_data,pc_struct);

    //debug_write(output_image_rec,"gauss_filt_rec" + std::to_string(sigma));

    //Compute APR Filter

    ExtraPartCellData<float> filter_output;

    filter_output = filter_apr_by_slice_mult<float>(pc_struct,filter,filter,filter,analysis_data,num_repeats);

    MeshData<float> output_image_apr;

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    interp_img(output_image_apr, pc_data, part_new, filter_output);

    remove_boundary(output_image_apr,2*filter_offset +2);

    calc_mse(gt_output,output_image_apr,"gauss_filt_apr" + std::to_string(sigma),analysis_data);
    calc_mse_part_locations(gt_output,output_image_apr,"gauss_filt_apr" + std::to_string(sigma),analysis_data,pc_struct);

   // debug_write(output_image_apr,"gauss_filt_apr" + std::to_string(sigma));

}
template<typename T>
void evaluate_filters_log(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,MeshData<T>& input_image){
    //
    //  Bevan Cheeseman 2017
    //
    //  Evaluate the accuracy of the filters
    //
    //

    //filter set up
    uint64_t filter_offset = 8;
    float sigma = 0.1;

    std::vector<float> filter_f;
    std::vector<float> filter_b;

    create_LOG_filter(filter_offset,sigma,filter_f,filter_b);
    filter_f = create_dog_filter<float>(filter_offset,1.5,3);
    filter_b = create_gauss_filter<float>(sigma,filter_offset);

    float num_repeats = 1;

    //Generate clean gt image
    MeshData<uint16_t> gt_image;
    generate_gt_image(gt_image, syn_image);


    MeshData<float> gt_image_f;
    gt_image_f.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(gt_image.mesh.begin(),gt_image.mesh.end(),gt_image_f.mesh.begin());

    MeshData<float> gt_output;

    MeshData<float> temp;
    temp.initialize(gt_image.y_num,gt_image.x_num,gt_image.z_num,0);

    //first y direction
    gt_output =  pixel_filter_full_mult(gt_image_f,filter_f,filter_b,filter_b,num_repeats,analysis_data);

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        temp.mesh[k] += (gt_output.mesh[k]);
    }

    //first x direction
    gt_output =  pixel_filter_full_mult(gt_image_f,filter_b,filter_f,filter_b,num_repeats,analysis_data);

    //std::transform (temp.mesh.begin(), temp.mesh.end(), gt_output.mesh.begin(), gt_output.mesh.begin(), std::plus<float>());

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        temp.mesh[k] += (gt_output.mesh[k]);
    }

    //first z direction
    gt_output =  pixel_filter_full_mult(gt_image_f,filter_b,filter_b,filter_f,num_repeats,analysis_data);

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        gt_output.mesh[k] += (temp.mesh[k]);
    }

    //std::transform (gt_output.mesh.begin(), gt_output.mesh.end(), temp.mesh.begin(), gt_output.mesh.begin(), std::plus<float>());

    for (int k = 0; k < gt_output.mesh.size(); ++k) {
        gt_output.mesh[k] = (gt_output.mesh[k]);
    }

    remove_boundary(gt_output,2*filter_offset +2);

    debug_write(gt_output,"log_filt_gt");



}


void compute_guass_smooth(float sigma,MeshData<float> input,MeshData<float> gt_output,AnalysisData& analysis_data,std::string name){
    //
    //  Performs Guassian Smoothing on Image and compares to GT.
    //
    std::vector<float> filter;

    uint64_t filter_offset = 10;

    float num_repeats = 1;

    filter = create_gauss_filter<float>(sigma,filter_offset);

    MeshData<float> output_image_org;

    output_image_org =  pixel_filter_full_mult(input,filter,filter,filter,num_repeats,analysis_data);

    remove_boundary(output_image_org,2*filter_offset +2);
    remove_boundary(gt_output,2*filter_offset +2);

    calc_mse(gt_output,output_image_org,"test_smooth_" + name + std::to_string(sigma),analysis_data);

}



template<typename T>
void evaluate_adaptive_smooth(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,MeshData<T>& input_image){
    //
    //  Bevan Cheeseman 2017
    //
    //  Evaluate the accuracy of the filters
    //
    //





    float num_repeats = 1;

    //Generate clean gt image
    MeshData<uint16_t> gt_image;
    generate_gt_image(gt_image, syn_image);

    debug_write(gt_image,"gt_image");

    MeshData<float> gt_image_f;
    gt_image_f.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(gt_image.mesh.begin(),gt_image.mesh.end(),gt_image_f.mesh.begin());

    MeshData<float> input_image_org;
    MeshData<float> output_image_org;

    //transfer across to float
    input_image_org.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(input_image.mesh.begin(),input_image.mesh.end(),input_image_org.mesh.begin());

    calc_mse(gt_image_f,input_image_org,"org_psnr",analysis_data);

    // Compute multiple smoothed versions of the original, and check the psnr how does that compare??

    std::vector<float> sig_vec = {0.5,.75,1,1.5,2,4,8};

    for (int i = 0; i < sig_vec.size(); ++i) {
        compute_guass_smooth(sig_vec[i],input_image_org,gt_image_f,analysis_data,"org");
    }

    debug_write(input_image_org,"org_image");

    // Compute Full Filter on APR reconstruction

    MeshData<float> input_image_rec;

    pc_struct.interp_parts_to_pc(input_image_rec,pc_struct.part_data.particle_data);

    calc_mse(gt_image_f,input_image_rec,"rec_psnr",analysis_data);

    for (int i = 0; i < sig_vec.size(); ++i) {
        compute_guass_smooth(sig_vec[i],input_image_rec,gt_image_f,analysis_data,"rec");
    }

    debug_write(input_image_rec,"rec_image");

    MeshData<float> output_image_apr;

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    std::vector<float> filter_s = {.1,.8,.1};

    ExtraPartCellData<float> particle_data;

    part_new.create_particles_at_cell_structure(particle_data);

    particle_data = sep_neigh_filter(pc_data,particle_data,filter_s);

    interp_img(output_image_apr, pc_data, part_new, particle_data,true);
    //remove_boundary(output_image_apr,2*filter_offset +1);
    //debug_write(output_image_apr,"adaptive_smooth_1");
    calc_mse(gt_image_f,output_image_apr,"adaptive_smooth_1",analysis_data);

    particle_data = sep_neigh_filter(pc_data,particle_data,filter_s);

    interp_img(output_image_apr, pc_data, part_new, particle_data,true);
    //remove_boundary(output_image_apr,2*filter_offset +1);
   // debug_write(output_image_apr,"adaptive_smooth_2");
    calc_mse(gt_image_f,output_image_apr,"adaptive_smooth_2",analysis_data);

    particle_data = sep_neigh_filter(pc_data,particle_data,filter_s);

    interp_img(output_image_apr, pc_data, part_new, particle_data,true);
    //remove_boundary(output_image_apr,2*filter_offset +1);
   // debug_write(output_image_apr,"adaptive_smooth_3");
    calc_mse(gt_image_f,output_image_apr,"adaptive_smooth_3",analysis_data);

    particle_data = sep_neigh_filter(pc_data,particle_data,filter_s);

    interp_img(output_image_apr, pc_data, part_new, particle_data,true);
    //remove_boundary(output_image_apr,2*filter_offset +1);
    debug_write(output_image_apr,"adaptive_smooth_4");
    calc_mse(gt_image_f,output_image_apr,"adaptive_smooth_4",analysis_data);

    particle_data = sep_neigh_filter(pc_data,particle_data,filter_s);

    interp_img(output_image_apr, pc_data, part_new, particle_data,true);
    //remove_boundary(output_image_apr,2*filter_offset +1);
   // debug_write(output_image_apr,"adaptive_smooth_5");
    calc_mse(gt_image_f,output_image_apr,"adaptive_smooth_5",analysis_data);

    particle_data = sep_neigh_filter(pc_data,particle_data,filter_s);

    interp_img(output_image_apr, pc_data, part_new, particle_data,true);
    //remove_boundary(output_image_apr,2*filter_offset +1);
    //debug_write(output_image_apr,"adaptive_smooth_6");
    calc_mse(gt_image_f,output_image_apr,"adaptive_smooth_6",analysis_data);

    particle_data = sep_neigh_filter(pc_data,particle_data,filter_s);

    interp_img(output_image_apr, pc_data, part_new, particle_data,true);
    //remove_boundary(output_image_apr,2*filter_offset +1);
   // debug_write(output_image_apr,"adaptive_smooth_7");
    calc_mse(gt_image_f,output_image_apr,"adaptive_smooth_7",analysis_data);

}
void run_real_segmentation(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,Proc_par& pars){


    ExtraPartCellData<uint16_t> seg_parts;

    std::array<float,13> parameters_new = {pars.I_th,1,2,3,1,2,3,pars.dy/pars.dy,pars.dx/pars.dy,pars.dz/pars.dy,60000,0};

    Part_timer timer;
    timer.verbose_flag = true;
    timer.start_timer("full seg");

    calc_graph_cuts_segmentation_new(pc_struct, seg_parts,analysis_data,parameters_new);

    timer.stop_timer();

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    MeshData<uint16_t> seg_mesh;

    timer.start_timer("interp_img");

    interp_img(seg_mesh, pc_data, part_new, seg_parts,true);

    timer.stop_timer();

    debug_write(seg_mesh,pc_struct.name + "_new_seg");

//    proj_par proj_pars;
//
//    proj_pars.theta_0 = 0;
//    proj_pars.theta_final = 3.14;
//    proj_pars.radius_factor = 1.00;
//    proj_pars.theta_delta = 0.1;
//    proj_pars.scale_z = pc_struct.pars.aniso;
//
//    ExtraPartCellData<uint16_t> y_vec;
//
//    create_y_data(y_vec,part_new);
//
//    APR<float> apr;
//
//    apr.init(pc_struct);
//
//    apr.shift_particles_from_cells(seg_parts);
//
//    ExtraPartCellData<uint16_t> seg_parts_depth = multiply_by_depth(seg_parts);
//
//    proj_pars.name = pc_struct.name +"_z_depth";
//
//    apr_perspective_raycast_depth(y_vec,seg_parts,seg_parts_depth,proj_pars,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);},true);
//
//    analysis_data.add_timer(timer);
}

//NEEDS TO BE UPDATED BELOW TO NEW STRUCTURES IF I WANT TO BE ABLE TO KEEP IT IN

//template<typename T>
//void run_ray_cast(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,MeshData<T>& input_image,Proc_par& pars){
//    //
//    //  Bevan Cheeseman 2017
//    //
//    //  Benchmark Ray Cast
//    //
//    //
//
//    Part_timer timer;
//
//    /////////////////
//    //
//    //  Parameters
//    ////////////////
//
//    proj_par proj_pars;
//
//    proj_pars.theta_0 = 0;
//    proj_pars.theta_final = 3.14;
//    proj_pars.radius_factor = 1.00;
//    proj_pars.theta_delta = 0.1;
//    proj_pars.scale_z = pars.aniso;
//
//    ParticleDataNew<float, uint64_t> part_new;
//    //flattens format to particle = cell, this is in the classic access/part paradigm
//    part_new.initialize_from_structure(pc_struct);
//
//    ExtraPartCellData<uint16_t> y_vec;
//
//    create_y_data(y_vec,part_new);
//
//    ExtraPartCellData<uint16_t> particles_int;
//    part_new.create_particles_at_cell_structure(particles_int);
//
//    shift_particles_from_cells(part_new,particles_int);
//
//    proj_pars.name = pc_struct.name;
//
//    timer.start_timer("ray_cast_max_part");
//
//    float t_apr = apr_perspective_raycast(y_vec,particles_int,proj_pars,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});
//
//    timer.stop_timer();
//
//    timer.start_timer("ray_cast_max_mesh");
//    float t_mesh = perpsective_mesh_raycast(pc_struct,proj_pars,input_image);
//    timer.stop_timer();
//
//    analysis_data.add_float_data("apr_ray_cast",t_apr);
//    analysis_data.add_float_data("mesh_ray_cast",t_mesh);
//
//
//}

template<typename T>
void evaluate_adaptive_grad(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,MeshData<T>& input_image){
    //
    //  Bevan Cheeseman 2017
    //
    //  Evaluate the accuracy of the filters
    //
    //

    //get gt and original gradient

    //Generate clean gt image
    MeshData<uint16_t> gt_image;
    generate_gt_image(gt_image, syn_image);

    debug_write(gt_image,"gt_image");

    MeshData<float> output_gt;
    output_gt = compute_grad(gt_image);

    remove_boundary(output_gt,2);
    debug_write(output_gt,"gt_grad");

    MeshData<float> output_org;
    output_org = compute_grad(input_image);

    remove_boundary(output_org,2);
    debug_write(output_org,"org_grad");

    calc_mse(output_gt,output_org,"org_grad",analysis_data);

    //compute grad on APR reconstructed

    MeshData<float> input_image_rec;

    pc_struct.interp_parts_to_pc(input_image_rec,pc_struct.part_data.particle_data);

    MeshData<float> output_rec;
    output_rec = compute_grad(input_image_rec);

    remove_boundary(output_rec,2);
    debug_write(output_rec,"rec_grad");

    calc_mse(output_gt,output_rec,"rec_grad",analysis_data);

    MeshData<float> output_image_apr;

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    ExtraPartCellData<float> particle_data;

    part_new.create_particles_at_cell_structure(particle_data);

    ExtraPartCellData<float> grad_output_y =  sep_neigh_grad(pc_data,particle_data,0);

    ExtraPartCellData<float> grad_output_x =  sep_neigh_grad(pc_data,particle_data,1);

    grad_output_y = transform_parts(grad_output_y,square<float>);
    grad_output_x = transform_parts(grad_output_x,square<float>);

    transform_parts(grad_output_y,grad_output_x,std::plus<float>());

    ExtraPartCellData<float> grad_output_z =  sep_neigh_grad(pc_data,particle_data,2);

    grad_output_z = transform_parts(grad_output_z,square<float>);

    transform_parts(grad_output_y,grad_output_z,std::plus<float>());

    grad_output_y = transform_parts(grad_output_y,square_root<float>);

    interp_img(output_image_apr, pc_data, part_new, grad_output_y,true);

    remove_boundary(output_image_apr,2);
    debug_write(output_image_apr,"adaptive_grad");

    remove_boundary(input_image,2);
    debug_write(input_image,"input_image");

    calc_mse(output_gt,output_image_apr,"adaptive_grad",analysis_data);

}

template<typename T>
void real_adaptive_grad(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,MeshData<T>& input_image,Proc_par& pars){
    //
    //  Bevan Cheeseman 2017
    //
    //  Evaluate the accuracy of the filters
    //
    //

    Part_timer timer;

    MeshData<float> output_org;


    //compute grad on APR reconstructed
    MeshData<float> input_image_org;


    //transfer across to float
    input_image_org.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(input_image.mesh.begin(),input_image.mesh.end(),input_image_org.mesh.begin());

    timer.start_timer("mesh_adapt_grad");

    output_org = compute_grad(input_image_org);

    timer.stop_timer();


    remove_boundary(output_org,2);
    debug_write(output_org,pc_struct.name + "_org_grad");

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    std::vector<float> delta = {pars.dy/pars.dy,pars.dx/pars.dy,pars.dz/pars.dy};

    ExtraPartCellData<float> particle_data;

    part_new.create_particles_at_cell_structure(particle_data);

    timer.start_timer("mesh_adapt_grad");

    ExtraPartCellData<float> smoothed_gradient_mag = adaptive_grad(pc_data,particle_data,3,delta);

    timer.stop_timer();

    MeshData<float> output_image_apr;

    interp_img(output_image_apr, pc_data, part_new, smoothed_gradient_mag,true);

    remove_boundary(output_image_apr,2);
    debug_write(output_image_apr,pc_struct.name + "adaptive_grad");

    analysis_data.add_timer(timer);

}


#endif //PARTPLAY_NUMERICS_BENCHMARKS_HPP
