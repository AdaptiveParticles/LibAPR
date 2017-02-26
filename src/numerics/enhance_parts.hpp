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
PartCellStructure<U,uint64_t> compute_guided_apr(Mesh_data<U> input_image,PartCellStructure<U,uint64_t>& pc_struct,Part_rep& part_rep){
    //
    //  Bevan Cheeseman 2017
    //
    //  Computes the APR using an already established APR
    //

    part_rep.initialize(pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);

    part_rep.pars.rel_error = 0.1;

    part_rep.pars.ydim = pc_struct.org_dims[0];
    part_rep.pars.xdim = pc_struct.org_dims[1];
    part_rep.pars.zdim = pc_struct.org_dims[2];

    part_rep.pars.dy = part_rep.pars.dx = part_rep.pars.dz = 1;
    part_rep.pars.psfx = part_rep.pars.psfy = part_rep.pars.psfz = 1;
    part_rep.pars.rel_error = 0.1;
    part_rep.pars.var_th = 5;
    part_rep.pars.var_th_max = 10;
    part_rep.pars.I_th = 950;

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

    std::vector<float> filter = {.0125,.975,.0125};
    std::vector<float> delta = {1,1,4};

    int num_tap = 1;

   //

    ExtraPartCellData<float> smoothed_gradient_mag = adaptive_grad(pc_data,particle_data,3,delta);

    if(input_image.x_num == 0){
        ExtraPartCellData<float> smoothed_parts = adaptive_smooth(pc_data,particle_data,num_tap,filter);
        interp_img(input_image, pc_data, part_new, smoothed_parts,true);
    }

    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;

    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {2,2,2};
    std::vector<unsigned int> status_offsets_max = {2,2,2};

    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,0,0);



    transform_parts(adaptive_max,adaptive_min,std::minus<float>());

    //pc_struct.interp_parts_to_pc(var,adaptive_max);

    //
    //  Second Pass Through
    //
    //

    Mesh_data<float> grad;
//
    interp_img(grad, pc_data, part_new, smoothed_gradient_mag,true);
//
    Mesh_data<float> var;

    float min = 10;
    var.preallocate(grad.y_num, grad.x_num, grad.z_num, 0);


//
//    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, grad.y_num, grad.x_num, grad.z_num, part_rep);
//
    Mesh_data<float> temp;
//
//
    interp_extrapc_to_mesh(temp,pc_struct,adaptive_max);


    Mesh_data<float> norm_grad;
    interp_img(norm_grad, pc_data, part_new, smoothed_gradient_mag,true);

        for (int j = 0; j < norm_grad.z_num; ++j) {
        for (int i = 0; i < norm_grad.x_num; ++i) {
            for (int k = 0; k < norm_grad.y_num; ++k) {
                if (var(i,j,k) > min) {
                    norm_grad(i,j,k) = 1000*norm_grad(i,j,k) / temp(i,j,k);
                } else {
                    norm_grad(i,j,k) = 1000*norm_grad(i,j,k) / min;
                }
            }
        }
    }

    debug_write(norm_grad,"norm_grad");


    if(part_rep.pars.I_th > 0) {
        intensity_th(input_image, temp,
                     part_rep.pars.I_th);
    }

    debug_write(temp,"var_input");


//
    down_sample(temp,var,
                [](float x, float y) { return x+y; },
                [](float x) { return x * (1.0/8.0); },true);


    rescale_var_and_threshold( temp,1,part_rep);

//
//
//
    //debug_write(norm_grad,"adaptive_max");
//
////

    temp.preallocate(grad.y_num, grad.x_num, grad.z_num, 0);

    get_level_3D(var, grad, part_rep,part_map,
                 temp);

    part_map.pushing_scheme(part_rep);



    part_map.downsample(input_image);

    //debug_write(part_map.layers[pc_struct.depth_max].mesh);

    PartCellStructure<float,uint64_t> pc_struct_new;

    pc_struct_new.initialize_structure(part_map);

    return pc_struct_new;
}





#endif //PARTPLAY_ENHANCE_PARTS_HPP
