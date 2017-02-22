//
// Created by cheesema on 27/01/17.
//

#ifndef PARTPLAY_NUMERICS_BENCHMARKS_HPP
#define PARTPLAY_NUMERICS_BENCHMARKS_HPP

#include "../../src/numerics/graph_cut_seg.hpp"
#include "../../src/numerics/filter_numerics.hpp"
#include "../../src/numerics/ray_cast.hpp"


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





#endif //PARTPLAY_NUMERICS_BENCHMARKS_HPP
