//
// Created by cheesema on 25/02/17.
//

#ifndef PARTPLAY_ENHANCE_PARTS_HPP
#define PARTPLAY_ENHANCE_PARTS_HPP

#include <functional>
#include <string>

#include "filter_numerics.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "src/data_structures/APR/ExtraPartCellData.hpp"
#include "benchmarks/development/Tree/PartCellParent.hpp"


#include "benchmarks/analysis/AnalysisData.hpp"
#include "parent_numerics.hpp"
#include "misc_numerics.hpp"
#include "benchmarks/development/old_algorithm/apr_pipeline.hpp"
#include "src/data_structures/APR/APR.hpp"

template<typename U>
void compute_guided_var(Mesh_data<U> input_image,PartCellStructure<U,uint64_t>& pc_struct,Part_rep& part_rep){
    //
    //  Bevan Cheeseman 2017
    //
    //  Computes the APR using an already established APR
    //

    part_rep.initialize(pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);

    //part_rep.pars.rel_error = 0.1;

    part_rep.pars.ydim = pc_struct.org_dims[0];
    part_rep.pars.xdim = pc_struct.org_dims[1];
    part_rep.pars.zdim = pc_struct.org_dims[2];

    part_rep.pars.dy = part_rep.pars.dx = part_rep.pars.dz = 1;

    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;

    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {1,2,3};
    std::vector<unsigned int> status_offsets_max = {1,2,3};

    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,0,0);

    Mesh_data<float> temp;

    interp_extrapc_to_mesh(temp,pc_struct,adaptive_max);

    Mesh_data<float> var;

    down_sample(temp,var,
                [](float x, float y) { return x+y; },
                [](float x) { return x * (1.0/8.0); },true);

}

//template<typename U>
//ExtraPartCellData<float> update_new_particles(Mesh_data<U>& input_image,APR<float>& apr){
//    //
//    //  Gradient using previous APR structure
//    //
//    //
//
//    ExtraPartCellData<float> new_particle_intensities;
//
//    new_particle_intensities.initialize_structure_cells(apr.pc_data);
//
//    int z_, x_, j_, y_, i, k;
//
//    for (uint64_t depth = (apr.y_vec.depth_min); depth <= apr.y_vec.depth_max; depth++) {
//        //loop over the resolutions of the structure
//        const unsigned int x_num_ = apr.y_vec.x_num[depth];
//        const unsigned int z_num_ = apr.y_vec.z_num[depth];
//
//        const float step_size_x = pow(2, apr.y_vec.depth_max - depth);
//        const float step_size_y = pow(2, apr.y_vec.depth_max - depth);
//        const float step_size_z = pow(2, apr.y_vec.depth_max - depth);
//
//
//#pragma omp parallel for default(shared) private(z_,x_,j_,i,k) schedule(guided) if(z_num_*x_num_ > 1000)
//        for (z_ = 0; z_ < z_num_; z_++) {
//            //both z and x are explicitly accessed in the structure
//
//            for (x_ = 0; x_ < x_num_; x_++) {
//
//                const unsigned int pc_offset = x_num_ * z_ + x_;
//
//                for (j_ = 0; j_ < apr.y_vec.data[depth][pc_offset].size(); j_++) {
//
//
//                    const int y = apr.y_vec.data[depth][pc_offset][j_];
//
//                    const float y_actual = floor((y+0.5) * step_size_y);
//                    const float x_actual = floor((x_+0.5) * step_size_x);
//                    const float z_actual = floor((z_+0.5) * step_size_z);
//
//                    new_particle_intensities.data[depth][pc_offset][j_]=input_image(y_actual,x_actual,z_actual);
//
//
//                }
//            }
//        }
//    }
//
//    return new_particle_intensities;
//
//
//
//}


//
//template<typename U>
//PartCellStructure<U,uint64_t> compute_guided_apr_time(Mesh_data<U>& input_image,PartCellStructure<U,uint64_t>& pc_struct,Part_rep& part_rep,APR<float>& apr_c,ExtraPartCellData<float>& scale){
//    //
//    //  Bevan Cheeseman 2017
//    //
//    //  Computes the APR using an already established APR
//    //
//
//    //set up some new structures used in this test
//    AnalysisData analysis_data;
//
//    std::vector<float> delta = {part_rep.pars.dy,part_rep.pars.dx,part_rep.pars.dz};
//
//    Particle_map<float> part_map(part_rep);
//    preallocate(part_map.layers, input_image.y_num, input_image.x_num, input_image.z_num, part_rep);
//
//
//    //ExtraPartCellData<float> particles_update = update_new_particles(input_image,apr_c);
//
//    ExtraPartCellData<float> gradient_mag = adaptive_grad(apr_c.pc_data,apr_c.particles_int,3,delta);
//
//    //adaptive mean
//    ExtraPartCellData<float> adaptive_min;
//    ExtraPartCellData<float> adaptive_max;
//
//    //offsets past on cell status (resolution)
//    std::vector<unsigned int> status_offsets_min = {2,2,3};
//    std::vector<unsigned int> status_offsets_max = {2,2,3};
//
//    part_map.downsample(input_image);
//
//    pc_struct.update_parts(part_map);
//
//    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,0,0);
//
//    Mesh_data<float> grad;
////
//    interp_img(grad, apr_c.pc_data, apr_c.part_new, gradient_mag,true);
////
//    Mesh_data<float> var;
//
//    Mesh_data<float> temp;
//
//    float min = 10;
//    var.preallocate(grad.y_num, grad.x_num, grad.z_num, 0);
//
//    interp_extrapc_to_mesh(temp,pc_struct,adaptive_max);
//
//    if(part_rep.pars.I_th > 0) {
//        intensity_th(input_image, temp,
//                     part_rep.pars.I_th);
//    }
//
//
//    rescale_var_and_threshold( temp,1,part_rep);
//
//
//
//    down_sample(temp,var,
//                [](float x, float y) { return x+y; },
//                [](float x) { return x * (1.0/8.0); },true);
//
//    debug_write(var,"var");
//    debug_write(grad,"grad");
//
//
//    scale = update_new_particles(temp,apr_c);
//
//    temp.preallocate(grad.y_num, grad.x_num, grad.z_num, 0);
//
//    get_level_3D(var, grad, part_rep,part_map,
//                 temp);
//
//    part_map.pushing_scheme(part_rep);
//
//
//
//    PartCellStructure<float,uint64_t> pc_struct_new;
//
//    pc_struct_new.initialize_structure(part_map);
//
//    return pc_struct_new;
//}



template<typename U>
PartCellStructure<U,uint64_t> compute_guided_apr(Mesh_data<U> input_image,PartCellStructure<U,uint64_t>& pc_struct,Part_rep& part_rep){
    //
    //  Bevan Cheeseman 2017
    //
    //  Computes the APR using an already established APR
    //

    part_rep.initialize(pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2]);

    //part_rep.pars.rel_error = 0.1;

    part_rep.pars.ydim = pc_struct.org_dims[0];
    part_rep.pars.xdim = pc_struct.org_dims[1];
    part_rep.pars.zdim = pc_struct.org_dims[2];

    part_rep.pars.dy = part_rep.pars.dx = part_rep.pars.dz = 1;
    //part_rep.pars.psfx = part_rep.pars.psfy = part_rep.pars.psfz = 1;
    //part_rep.pars.rel_error = 0.1;
    //part_rep.pars.var_th = 5;
    //part_rep.pars.var_th_max = 5;
    //part_rep.pars.I_th = 950;

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

    std::vector<float> filter =  {0,1,0};
    std::vector<float> delta = {part_rep.pars.dy,part_rep.pars.dx,part_rep.pars.dz};

    int num_tap = 1;

   //
    ExtraPartCellData<float> smoothed_parts = adaptive_smooth(pc_data,particle_data,num_tap,filter);
    interp_img(input_image, pc_data, part_new, smoothed_parts,true);


    ExtraPartCellData<float> smoothed_gradient_mag = adaptive_grad(pc_data,smoothed_parts,3,delta);

    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;

    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {1,2,3};
    std::vector<unsigned int> status_offsets_max = {1,2,3};

    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,0,0);


    Mesh_data<float> temp;

    interp_extrapc_to_mesh(temp,pc_struct,adaptive_max);

    debug_write(temp,"max_tt");

    interp_extrapc_to_mesh(temp,pc_struct,adaptive_min);

    debug_write(temp,"min");

    transform_parts(adaptive_max,adaptive_min,std::minus<float>());



    Mesh_data<float> grad;
//
    interp_img(grad, pc_data, part_new, smoothed_gradient_mag,true);
//
    Mesh_data<float> var;

    float min = 10;
    var.preallocate(grad.y_num, grad.x_num, grad.z_num, 0);

    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, grad.y_num, grad.x_num, grad.z_num, part_rep);

    interp_extrapc_to_mesh(temp,pc_struct,adaptive_max);

    if(part_rep.pars.I_th > 0) {
        intensity_th(input_image, temp,
                     part_rep.pars.I_th);
    }


    down_sample(temp,var,
                [](float x, float y) { return x+y; },
                [](float x) { return x * (1.0/8.0); },true);


    rescale_var_and_threshold( temp,1,part_rep);

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
