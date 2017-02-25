//
// Created by cheesema on 25/02/17.
//

#ifndef PARTPLAY_ENHANCE_PARTS_HPP
#define PARTPLAY_ENHANCE_PARTS_HPP

#include <functional>
#include <string>

#include "filter_numerics.hpp"
#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"


#include "../../benchmarks/analysis/AnalysisData.hpp"
#include "parent_numerics.hpp"
#include "misc_numerics.hpp"
#include "../../src/algorithm/apr_pipeline.hpp"

template<typename U>
PartCellStructure<U,uint64_t> compute_guided_apr(Mesh_data<U> input_image,PartCellStructure<U,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2017
    //
    //  Computes the APR using an already established APR
    //

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

    //set up some new structures used in this test
    AnalysisData analysis_data;

    float num_repeats = 1;

    std::vector<float> filter = {.05,.9,.05};
    std::vector<float> delta = {1,1,4};

    int num_tap = 4;

    ExtraPartCellData<float> smoothed_parts = adaptive_smooth(pc_data,particle_data,num_tap,filter);

    ExtraPartCellData<float> smoothed_gradient_mag = adaptive_grad(pc_data,smoothed_parts,3,delta);


    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;

    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {2,2,2};
    std::vector<unsigned int> status_offsets_max = {2,2,2};

    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,0,0);

    transform_parts(adaptive_max,adaptive_min,std::minus<float>());

    Mesh_data<float> var;
    float min = 100;
//
//    for (int i = 0; i < var.mesh.size(); ++i) {
//        if (var.mesh[i] > min) {
//        } else {
//            var.mesh[i] = min;
//        }
//    }

    //pc_struct.interp_parts_to_pc(var,adaptive_max);
    interp_extrapc_to_mesh(var,pc_struct,adaptive_max);



    Mesh_data<float> norm_grad;

    interp_img(norm_grad, pc_data, part_new, smoothed_gradient_mag,true);

    //debug_write(norm_grad,"adaptive_max");


    for (int j = 0; j < norm_grad.z_num; ++j) {
        for (int i = 0; i < norm_grad.x_num; ++i) {
            for (int k = 0; k < norm_grad.y_num; ++k) {
                if (var(i,j,k) > min) {
                    norm_grad(i,j,k) = 1000*norm_grad(i,j,k) / var(i,j,k);
                } else {
                    norm_grad(i,j,k) = 1000*norm_grad(i,j,k) / min;
                }
            }
        }
    }


    debug_write(norm_grad,"norm_grad");

    return pc_struct;
}





#endif //PARTPLAY_ENHANCE_PARTS_HPP
