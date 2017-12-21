//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "src/data_structures/Tree/PartCellStructure.hpp"

#include "src/data_structures/Tree/PartCellParent.hpp"
#include "src/numerics/ray_cast.hpp"
#include "src/numerics/filter_numerics.hpp"
#include "src/numerics/misc_numerics.hpp"

template<typename ImageType>
class APR {

public:

    ParticleDataNew<ImageType, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm

    ExtraPartCellData<uint16_t> y_vec;

    ExtraPartCellData<ImageType> particles_int;

    PartCellData<uint64_t> pc_data;


    APR(){

    }

    APR(PartCellStructure<float,uint64_t>& pc_struct){
        init(pc_struct);

    }

    void init(PartCellStructure<float,uint64_t>& pc_struct){
        part_new.initialize_from_structure(pc_struct);

        create_y_data();

        part_new.create_particles_at_cell_structure(particles_int);

        shift_particles_from_cells(particles_int);

        part_new.initialize_from_structure(pc_struct);


    }


    void init_pc_data(){


        part_new.create_pc_data_new(pc_data);

        pc_data.org_dims = y_vec.org_dims;
    }

    void create_y_data(){
        //
        //  Bevan Cheeseman 2017
        //
        //  Creates y index
        //

        y_vec.initialize_structure_parts(part_new.particle_data);

        y_vec.org_dims = part_new.access_data.org_dims;

        int z_,x_,j_,y_;

        for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = part_new.access_data.x_num[depth];
            const unsigned int z_num_ = part_new.access_data.z_num[depth];

            CurrentLevel<ImageType, uint64_t> curr_level(part_new);
            curr_level.set_new_depth(depth, part_new);

            const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
            for (z_ = 0; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = 0; x_ < x_num_; x_++) {

                    curr_level.set_new_xz(x_, z_, part_new);

                    int counter = 0;

                    for (j_ = 0; j_ < curr_level.j_num; j_++) {

                        bool iscell = curr_level.new_j(j_, part_new);

                        if (iscell) {
                            //Indicates this is a particle cell node
                            curr_level.update_cell(part_new);

                            y_vec.data[depth][curr_level.pc_offset][counter] = curr_level.y;

                            counter++;
                        } else {

                            curr_level.update_gap();

                        }


                    }
                }
            }
        }



    }

    template<typename U>
    void shift_particles_from_cells(ExtraPartCellData<U>& pdata_old){
        //
        //  Bevan Cheesean 2017
        //
        //  Transfers them to align with the part data, to align with particle data no gaps
        //
        //

        ExtraPartCellData<U> pdata_new;

        pdata_new.initialize_structure_parts(part_new.particle_data);

        uint64_t z_,x_,j_,node_val;
        uint64_t part_offset;

        for(uint64_t i = part_new.access_data.depth_min;i <= part_new.access_data.depth_max;i++){

            const unsigned int x_num_ = part_new.access_data.x_num[i];
            const unsigned int z_num_ = part_new.access_data.z_num[i];

#pragma omp parallel for default(shared) private(z_,x_,j_,part_offset,node_val)  if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = part_new.access_data.data[i][offset_pc_data].size();

                    int counter = 0;

                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val = part_new.access_data.data[i][offset_pc_data][j_];

                        if(!(node_val&1)){

                            pdata_new.data[i][offset_pc_data][counter] = pdata_old.data[i][offset_pc_data][j_];

                            counter++;

                        } else {

                        }

                    }
                }
            }
        }

        std::swap(pdata_new,pdata_old);

    }




    void get_type(ExtraPartCellData<uint8_t>& type){
        //
        //  Bevan Cheesean 2017
        //
        //  Transfers them to align with the part data, to align with particle data no gaps
        //
        //

        type.initialize_structure_parts(part_new.particle_data);

        uint64_t z_,x_,j_,node_val;
        uint64_t part_offset;

        for(uint64_t i = part_new.access_data.depth_min;i <= part_new.access_data.depth_max;i++){

            const unsigned int x_num_ = part_new.access_data.x_num[i];
            const unsigned int z_num_ = part_new.access_data.z_num[i];

#pragma omp parallel for default(shared) private(z_,x_,j_,part_offset,node_val)  if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = part_new.access_data.data[i][offset_pc_data].size();

                    int counter = 0;

                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val = part_new.access_data.data[i][offset_pc_data][j_];

                        if(!(node_val&1)){

                            type.data[i][offset_pc_data][counter] = part_new.access_node_get_status(node_val);

                            counter++;

                        } else {

                        }

                    }
                }
            }
        }



    }


};

#endif //PARTPLAY_APR_HPP
