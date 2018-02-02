//
// Created by cheesema on 01.02.18.
//

#ifndef PARTPLAY_OLDAPRCONVERTER_HPP
#define PARTPLAY_OLDAPRCONVERTER_HPP

#include <src/data_structures/APR/APR.hpp>
#include <benchmarks/development/Tree/PartCellStructure.hpp>

class OldAPRConverter {

public:
    template<typename T>
    void create_apr_from_pc_struct(APR<T>& apr,PartCellStructure<float,uint64_t>& pc_struct){

        apr.apr_access.org_dims[0] = pc_struct.org_dims[0];
        apr.apr_access.org_dims[1] = pc_struct.org_dims[1];
        apr.apr_access.org_dims[2] = pc_struct.org_dims[2];

        //first add the layers
        apr.apr_access.level_max = pc_struct.depth_max + 1;
        apr.apr_access.level_min = pc_struct.depth_min;

        apr.apr_access.z_num.resize(apr.apr_access.level_max+1);
        apr.apr_access.x_num.resize(apr.apr_access.level_max+1);
        apr.apr_access.y_num.resize(apr.apr_access.level_max+1);

        for(uint64_t i = apr.apr_access.level_min;i < apr.apr_access.level_max;i++){
            apr.apr_access.z_num[i] = pc_struct.z_num[i];
            apr.apr_access.x_num[i] = pc_struct.x_num[i];
            apr.apr_access.y_num[i] = pc_struct.y_num[i];
        }

        apr.apr_access.z_num[apr.apr_access.level_max] = pc_struct.org_dims[2];
        apr.apr_access.x_num[apr.apr_access.level_max] = pc_struct.org_dims[1];
        apr.apr_access.y_num[apr.apr_access.level_max] = pc_struct.org_dims[0];

        std::vector<MeshData<uint8_t>> p_map;
        p_map.resize(apr.apr_access.level_max);

        //initialize loop variables
        int x_;
        int z_;
        int y_;

        uint64_t j_;

        uint64_t status;
        uint64_t node_val;
        uint16_t node_val_part;

        for(uint64_t i = (apr.apr_access.level_max-1);i >= apr.apr_access.level_min;i--){

            const unsigned int x_num = apr.apr_access.x_num[i];
            const unsigned int z_num = apr.apr_access.z_num[i];
            const unsigned int y_num = apr.apr_access.y_num[i];

            p_map[i].initialize(y_num,x_num,z_num,0);

#pragma omp parallel for default(shared) private(j_,z_,x_,y_,node_val,status) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++) {

                for (x_ = 0; x_ < x_num; x_++) {

                    //access variables
                    const size_t offset_pc_data = x_num * z_ + x_;
                    const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                    const size_t offset_p_map = y_num*x_num*z_ + y_num*x_;

                    y_ = 0;

                    //first loop over
                    for (j_ = 0; j_ < j_num; j_++) {
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];

                        if (!(node_val & 1)) {
                            //normal node
                            y_++;
                            //create pindex, and create status (0,1,2,3) and type
                            status = (node_val & STATUS_MASK)
                                    >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.

                            p_map[i].mesh[offset_p_map + y_]=status;

                        } else {

                            y_ = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_--;
                        }
                    }
                }
            }
        }

        apr.apr_access.initialize_structure_from_particle_cell_tree(apr,p_map);


        apr.particles_intensities.data.resize(apr.total_number_particles());


        //initialize

        uint64_t y_coord;

        uint64_t curr_key=0;
        uint64_t part_offset=0;

        uint64_t p;

        uint64_t counter = 0;


        for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){

            const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[i];

            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
//#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_part,curr_key,part_offset,status) if(z_num_*x_num_ > 100)
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

                                //set the cooridnates info
                                //get the intensity
                                apr.particles_intensities.data[counter] = pc_struct.part_data.get_part(curr_key);

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



    }

    template<typename T,typename U>
    void transfer_intensities(APR<T>& apr,PartCellStructure<float,uint64_t>& pc_struct,ExtraPartCellData<U> extra_data){

        //initialize loop variables
        int x_;
        int z_;
        int y_;

        uint64_t j_;

        uint64_t status;
        uint64_t node_val;
        uint16_t node_val_part;



        apr.particles_intensities.data.resize(apr.total_number_particles());


        //initialize

        uint64_t y_coord;

        uint64_t curr_key=0;
        uint64_t part_offset=0;

        uint64_t p;

        uint64_t counter = 0;


        for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){

            const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[i];

            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
//#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_part,curr_key,part_offset,status) if(z_num_*x_num_ > 100)
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

                                //set the cooridnates info
                                //get the intensity
                                apr.particles_intensities.data[counter] = pc_struct.part_data.get_part(curr_key);

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



    }


};


#endif //PARTPLAY_OLDAPRCONVERTER_HPP
