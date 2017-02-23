//
// Created by cheesema on 27/01/17.
//

#ifndef PARTPLAY_NUMERICS_BENCHMARKS_HPP
#define PARTPLAY_NUMERICS_BENCHMARKS_HPP

#include "../../src/numerics/graph_cut_seg.hpp"
#include "../../src/numerics/filter_numerics.hpp"
#include "../../src/numerics/ray_cast.hpp"

template<typename S,typename T>
void calc_mse(Mesh_data<S>& org_img,Mesh_data<T>& rec_img,std::string name,AnalysisData& analysis_data){
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

#pragma omp parallel for default(shared) private(j,i,k) reduction(+: MSE)
    for(j = 0; j < z_num_o;j++){
        for(i = 0; i < x_num_o;i++){

            for(k = 0;k < y_num_o;k++){

                MSE += pow(org_img.mesh[j*x_num_o*y_num_o + i*y_num_o + k] - rec_img.mesh[j*x_num_r*y_num_r + i*y_num_r + k],2);

            }
        }
    }

    MSE = MSE/(z_num_o*x_num_o*y_num_o*1.0);


    // Mesh_data<S> SE;
    //SE.initialize(y_num_o,x_num_o,z_num_o,0);

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

    std::cout << PSNR << std::endl;

}

template <typename T>
void generate_gt_image(Mesh_data<T>& gt_image,SynImage& syn_image){
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
void run_segmentation_benchmark_mesh(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Runs the graph cuts segmentation benchmarks
    //

    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,2000,1,1,2,2,2,3,0,0};

    Mesh_data<uint8_t> seg_mesh;

    //memory on this machine can't handle anything bigger
    if(pc_struct.org_dims[0] <= 450){
        std::cout << "gc_seg_mesh" << std::endl;
        calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc,analysis_data);
        std::cout << "gc_seg_mesh_complete" << std::endl;
    }

}
template <typename T>
void generate_gt_seg(Mesh_data<T>& gt_image,SynImage& syn_image){
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
float compute_dice_coeff(Mesh_data<uint16_t>& gt,Mesh_data<uint8_t>& seg){
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

    Mesh_data<uint8_t> seg_mesh;

    //memory on this machine can't handle anything bigger
    if(pc_struct.org_dims[0] <= 450){
        calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc);
    }

    ExtraPartCellData<uint8_t> seg_parts;

    if(pc_struct.get_number_parts() <= pow(500,3)) {

        calc_graph_cuts_segmentation(pc_struct, seg_parts, parameters_nuc);
    }

    Mesh_data<uint8_t> seg_img;

    pc_struct.interp_parts_to_pc(seg_img,seg_parts);

    Mesh_data<uint16_t> gt_image;
    generate_gt_seg(gt_image,syn_image);

    float dice_mesh = compute_dice_coeff(gt_image,seg_mesh);
    float dice_parts = compute_dice_coeff(gt_image,seg_img);

    analysis_data.add_float_data("dice_mesh",dice_mesh);
    analysis_data.add_float_data("dice_parts",dice_parts);

    if(analysis_data.debug) {

        debug_write(seg_mesh, "seg_mesh");
        debug_write(seg_img, "seg_parts");
        debug_write(gt_image, "seg_gt");

        std::cout << "Dice m: " << dice_mesh << std::endl;
        std::cout << "Dice p: " << dice_parts << std::endl;
    }

}

void run_segmentation_benchmark_parts(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Runs the graph cuts segmentation benchmarks
    //

    ExtraPartCellData<uint8_t> seg_parts;

    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,2000,1,1,2,2,2,3,0,0};

    if(pc_struct.get_number_parts() <= pow(500,3)) {

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

    Mesh_data<float> output_image;

    Mesh_data<float> input_image;

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
    particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);

    //Get neighbours (random access)

    particle_random_access(pc_struct,analysis_data);

    // Filtering

    uint64_t filter_offset = 10;

    std::vector<float> filter;
    filter.resize(2*filter_offset + 1,1.0/(2*filter_offset + 1));

    ExtraPartCellData<float> filter_output;

    filter_output = filter_apr_by_slice<float>(pc_struct,filter,analysis_data,num_repeats);

}

template<typename T>
void remove_boundary(Mesh_data<T>& img,int sz){
    //
    //  Bevan Cheeseman 2017
    //
    //  Because I haven't dealt with boundary conditions properly for the filter, I want to remove those involved in boundary points from all computations and displaying results
    //

    unsigned int new_y = img.y_num - 2*sz;
    unsigned int new_x = img.x_num - 2*sz;
    unsigned int new_z = img.z_num - 2*sz;

    Mesh_data<T> new_mesh;
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
void evaluate_filters(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,Mesh_data<T>& input_image){
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
    Mesh_data<uint16_t> gt_image;
    generate_gt_image(gt_image, syn_image);

    Mesh_data<float> gt_image_f;
    gt_image_f.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(gt_image.mesh.begin(),gt_image.mesh.end(),gt_image_f.mesh.begin());

    Mesh_data<float> gt_output;

    gt_output =  pixel_filter_full(gt_image_f,filter,num_repeats,analysis_data);

    remove_boundary(gt_output,2*filter_offset +1);

    debug_write(gt_output,"filt_gt");

    // Compute Full Filter on input
    Mesh_data<float> output_image_org;

    Mesh_data<float> input_image_org;

    //transfer across to float
    input_image_org.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(input_image.mesh.begin(),input_image.mesh.end(),input_image_org.mesh.begin());

    output_image_org =  pixel_filter_full(input_image_org,filter,num_repeats,analysis_data);

    remove_boundary(output_image_org,2*filter_offset +1);

    calc_mse(gt_output,output_image_org,"filt_org",analysis_data);

    debug_write(output_image_org,"filt_org");

    // Compute Full Filter on APR reconstruction
    Mesh_data<float> output_image_rec;

    Mesh_data<float> input_image_rec;

    pc_struct.interp_parts_to_pc(input_image_rec,pc_struct.part_data.particle_data);

    output_image_rec =  pixel_filter_full(input_image_rec,filter,num_repeats,analysis_data);

    remove_boundary(output_image_rec,2*filter_offset +1);

    calc_mse(gt_output,output_image_rec,"filt_rec",analysis_data);

    debug_write(output_image_rec,"filt_rec");

    //Compute APR Filter

    ExtraPartCellData<float> filter_output;

    filter_output = filter_apr_by_slice<float>(pc_struct,filter,analysis_data,num_repeats);

    Mesh_data<float> output_image_apr;

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
void evaluate_filters_guassian(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data,SynImage& syn_image,Mesh_data<T>& input_image){
    //
    //  Bevan Cheeseman 2017
    //
    //  Evaluate the accuracy of the filters
    //
    //

    //filter set up
    uint64_t filter_offset = 5;

    std::vector<float> filter;

    //filter = create_dog_filter<float>(filter_offset,1.5,3);
    filter = create_gauss_filter<float>(filter_offset,1.5);

    float num_repeats = 1;

    //Generate clean gt image
    Mesh_data<uint16_t> gt_image;
    generate_gt_image(gt_image, syn_image);

    Mesh_data<float> gt_image_f;
    gt_image_f.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(gt_image.mesh.begin(),gt_image.mesh.end(),gt_image_f.mesh.begin());

    Mesh_data<float> gt_output;

    gt_output =  pixel_filter_full_mult(gt_image_f,filter,num_repeats,analysis_data);

    remove_boundary(gt_output,2*filter_offset +1);

    debug_write(gt_output,"gauss_filt_gt");

    // Compute Full Filter on input
    Mesh_data<float> output_image_org;

    Mesh_data<float> input_image_org;

    //transfer across to float
    input_image_org.initialize(input_image.y_num,input_image.x_num,input_image.z_num,0);
    std::copy(input_image.mesh.begin(),input_image.mesh.end(),input_image_org.mesh.begin());

    output_image_org =  pixel_filter_full_mult(input_image_org,filter,num_repeats,analysis_data);

    remove_boundary(output_image_org,2*filter_offset +1);

    calc_mse(gt_output,output_image_org,"gauss_filt_org",analysis_data);

    debug_write(output_image_org,"gauss_filt_org");

    // Compute Full Filter on APR reconstruction
    Mesh_data<float> output_image_rec;

    Mesh_data<float> input_image_rec;

    pc_struct.interp_parts_to_pc(input_image_rec,pc_struct.part_data.particle_data);

    output_image_rec =  pixel_filter_full_mult(input_image_rec,filter,num_repeats,analysis_data);

    remove_boundary(output_image_rec,2*filter_offset +1);

    calc_mse(gt_output,output_image_rec,"gauss_filt_rec",analysis_data);

    debug_write(output_image_rec,"gauss_filt_rec");

    //Compute APR Filter

    ExtraPartCellData<float> filter_output;

    filter_output = filter_apr_by_slice_mult<float>(pc_struct,filter,analysis_data,num_repeats);

    Mesh_data<float> output_image_apr;

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

    calc_mse(gt_output,output_image_apr,"gauss_filt_apr",analysis_data);

    debug_write(output_image_apr,"gauss_filt_apr");

    if(analysis_data.debug){
        //
        //
        //
        //
    }

}





#endif //PARTPLAY_NUMERICS_BENCHMARKS_HPP
