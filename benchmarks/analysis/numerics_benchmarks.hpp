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
    if(pc_struct.org_dims[0] <= 500){
        std::cout << "gc_seg_mesh" << std::endl;
        calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc,analysis_data);
        std::cout << "gc_seg_mesh_complete" << std::endl;
    }

}

void run_segmentation_benchmark_parts(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Runs the graph cuts segmentation benchmarks
    //

    ExtraPartCellData<uint8_t> seg_parts;

    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,2000,1,1,2,2,2,3,0,0};

    calc_graph_cuts_segmentation(pc_struct, seg_parts,parameters_nuc,analysis_data);

}


void run_filter_benchmarks_mesh(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Runs the filtering and neighbour access benchmarks
    //

    float num_repeats = 10;

    //Get neighbours (linear)

    //pixels
    pixels_linear_neigh_access(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],num_repeats,analysis_data);


    pixel_neigh_random(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],analysis_data);

    // Filtering

    uint64_t filter_offset = 10;


    std::vector<float> filter;
    filter.resize(2*filter_offset + 1,1.0/(2*filter_offset + 1));


    Mesh_data<float> output_image;

    Mesh_data<float> input_image;

    pc_struct.interp_parts_to_pc(input_image,pc_struct.part_data.particle_data);

    output_image =  pixel_filter_full(input_image,filter,num_repeats,analysis_data);

    ExtraPartCellData<float> filter_output_mesh;

    filter_output_mesh = filter_apr_input_img<float>(input_image,pc_struct,filter,analysis_data,num_repeats);


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
