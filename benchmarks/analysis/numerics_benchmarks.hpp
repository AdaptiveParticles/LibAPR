//
// Created by cheesema on 27/01/17.
//

#ifndef PARTPLAY_NUMERICS_BENCHMARKS_HPP
#define PARTPLAY_NUMERICS_BENCHMARKS_HPP

#include "../../src/numerics/graph_cut_seg.hpp"
#include "../../src/numerics/filter_numerics.hpp"


void run_segmentation_benchmark(PartCellStructure<float,uint64_t> pc_struct,AnalysisData& analysis_data){
    //
    //  Runs the graph cuts segmentation benchmarks
    //

    ExtraPartCellData<uint8_t> seg_parts;

    //nuclei
    std::array<uint64_t,10> parameters_nuc = {100,2000,1,1,2,2,2,3,0,0};

    calc_graph_cuts_segmentation(pc_struct, seg_parts,parameters_nuc,analysis_data);

    Mesh_data<uint8_t> seg_mesh;

    if(pc_struct.org_dims[0] <400){

        calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters_nuc,analysis_data);

    }

}








#endif //PARTPLAY_NUMERICS_BENCHMARKS_HPP
