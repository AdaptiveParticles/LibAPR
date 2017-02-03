/////////////////////////////////
//
//
//  Contains Miscellaneous APR numerics codes and output
//
//
//////////////////////////////////
#ifndef _misc_num_h
#define _misc_num_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "filter_help/FilterLevel.hpp"

template<typename S>
void interp_depth_to_mesh(Mesh_data<uint8_t>& k_img,PartCellStructure<S,uint64_t>& pc_struct);


template<typename S>
void interp_depth_to_mesh(Mesh_data<uint8_t>& k_img,PartCellStructure<S,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes a pc_struct and interpolates the depth to the mesh
    //
    //
    
    //initialize dataset to interp to
    ExtraPartCellData<uint8_t> k_parts(pc_struct.part_data.particle_data);
    
    
    
    //initialize variables required
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    // Extra variables required
    //
    
    uint64_t status=0;
    uint64_t part_offset=0;
    uint64_t p;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //Indicates this is a particle cell node
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        uint8_t depth = 2*i + (status == SEED);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            k_parts.get_part(curr_key) = depth;
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    
    //now interpolate this to the mesh
    pc_struct.interp_parts_to_pc(k_img,k_parts);
    
    
}


template<typename S>
void interp_status_to_mesh(Mesh_data<uint8_t>& status_img,PartCellStructure<S,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes a pc_struct and interpolates the depth to the mesh
    //
    //
    
    //initialize dataset to interp to
    ExtraPartCellData<uint8_t> status_parts(pc_struct.part_data.particle_data);
    
    
    
    //initialize variables required
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    // Extra variables required
    //
    
    uint64_t status=0;
    uint64_t part_offset=0;
    uint64_t p;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //Indicates this is a particle cell node
                        
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            status_parts.get_part(curr_key) = status;
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    
    //now interpolate this to the mesh
    pc_struct.interp_parts_to_pc(status_img,status_parts);
    
    
}

template<typename S,typename T>
void interp_extrapc_to_mesh(Mesh_data<T>& output_img,PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<T>& cell_data){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes a pc_struct and interpolates the depth to the mesh
    //
    //
    
    //initialize dataset to interp to
    ExtraPartCellData<T> extra_parts(pc_struct.part_data.particle_data);

    
    //initialize variables required
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    // Extra variables required
    //
    
    uint64_t status=0;
    uint64_t part_offset=0;
    uint64_t p;
    
    
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //Indicates this is a particle cell node
                        
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        float val = cell_data.get_val(curr_key);
                        
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            extra_parts.get_part(curr_key) = val;
                            
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    
    //now interpolate this to the mesh
    pc_struct.interp_parts_to_pc(output_img,extra_parts);
    
    
}
template<typename T>
void threshold_part(PartCellStructure<float,uint64_t>& pc_struct,ExtraPartCellData<T>& th_output,float threshold){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //
    //
    
    uint64_t x_;
    uint64_t z_;
    uint64_t j_;
    
    th_output.initialize_structure_parts(pc_struct.part_data.particle_data);
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    timer.start_timer("Threshold Loop");
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
        for(uint64_t depth = (pc_struct.depth_max); depth >= (pc_struct.depth_min); depth--){
            
            const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
            
            //initialize layer iterators
            FilterLevel<uint64_t,float> curr_level;
            curr_level.set_new_depth(depth,pc_struct);
            
#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                
                for(x_ = 0;x_ < x_num_;x_++){
                    curr_level.set_new_xz(x_,z_,pc_struct);
                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < curr_level.j_num_();j_++){
                        
                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        bool iscell = curr_level.new_j(j_,pc_struct);
                        
                        if (iscell){
                            curr_level.update_cell(pc_struct);
                            
                            for(int p = 0;p < pc_struct.part_data.get_num_parts(curr_level.status);p++){
                                th_output.data[depth][curr_level.pc_offset][curr_level.part_offset+p] = pc_struct.part_data.particle_data.data[depth][curr_level.pc_offset][curr_level.part_offset+p] > threshold ;
                            }
                            
                        } else {
                            curr_level.update_gap(pc_struct);
                        }
                    }
                    
                }
            }
            
            
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Particle Threshold Size: " << pc_struct.get_number_parts() << " took: " << time << std::endl;
    
    
    
    
}

template<typename U>
void threshold_pixels(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    float threshold = 50;
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 6;
    filter.resize(filter_offset*2 +1,1);
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                
                for(k = 0;k < y_num;k++){
                        
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = input_data.mesh[j*x_num*y_num + i*y_num + k] < threshold;
                    
                }
            }
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Pixel Threshold Size: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}



template<typename U,typename V>
void interp_slice(PartCellStructure<float,uint64_t>& pc_struct,ExtraPartCellData<V>& interp_data,int dir,int num){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //


    std::vector<int> x_num_min;
    std::vector<int> x_num;

    std::vector<int> z_num_min;
    std::vector<int> z_num;

    x_num.resize(pc_struct.depth_max + 1);
    z_num.resize(pc_struct.depth_max + 1);
    x_num_min.resize(pc_struct.depth_max + 1);
    z_num_min.resize(pc_struct.depth_max + 1);

    int x_dim = ceil(org_dims[0]/2.0)*2;
    int z_dim = ceil(org_dims[1]/2.0)*2;
    int y_dim = ceil(org_dims[2]/2.0)*2;




    if(dir == 2) {
        //yz case
        z_num = pc_struct.z_num;

        for (int i = pc_struct.depth_min; i <= pc_struct.depth_max ; ++i) {
            x_num[i] = num/pow(2,pc_struct.depth_max - i);
        }

        prev_k_img.mesh.resize(z_dim*y_dim);
        curr_k_img.mesh.resize(z_dim*y_dim);

        prev_k_img.set_size(pow(2,depth_min-1),1,pow(2,depth_min-1));

    } else {
        //yx case
        x_num = pc_struct.x_num;

        for (int i = pc_struct.depth_min; i <= pc_struct.depth_max ; ++i) {
            z_num[i] = num/pow(2,pc_struct.depth_max - i);
        }

        prev_k_img.mesh.resize(x_dim*y_dim);
        curr_k_img.mesh.resize(x_dim*y_dim);

        prev_k_img.set_size(pow(2,depth_min-1),pow(2,depth_min-1),1);
    }

    Mesh_data<U> curr_k_img;
    Mesh_data<U> prev_k_img;

    constexpr int y_incr[8] = {0,1,0,1,0,1,0,1};
    constexpr int x_incr[8] = {0,0,1,1,0,0,1,1};
    constexpr int z_incr[8] = {0,0,0,0,1,1,1,1};


    Part_timer timer;
    timer.verbose_flag = false;

    Part_timer t_n;
    t_n.verbose_flag = false;
    t_n.start_timer("loop");

    uint64_t z_ = 0;
    uint64_t x_ = 0;
    uint64_t j_ = 0;
    uint64_t y_coord = 0;
    uint64_t status = 0;
    uint64_t part_offset = 0;
    uint64_t curr_key = 0;
    uint64_t node_val = 0;

    uint64_t x_p = 0;
    uint64_t y_p = 0;
    uint64_t z_p = 0;
    uint64_t depth_ = 0;
    uint64_t status_ = 0;

    //loop over all levels of k
    for (uint64_t d = depth_min; depth_max >= d; d++) {

        ///////////////////////////////////////////////////////////////
        //
        //  Transfer previous level to current level
        //
        ////////////////////////////////////////////////////////////////
        timer.start_timer("upsample");

        const_upsample_img(curr_k_img,prev_k_img,org_dims);

        timer.stop_timer();

        /////////////////////////////////////////////////////////////////////
        //
        //  Place seed particles
        //
        //
        /////////////////////////////////////////////////////////////////

        timer.start_timer("particle loop");

        if ( d > depth_min){


            const unsigned int x_num_ = x_num[d-1];
            const unsigned int z_num_ = z_num[d-1];

            const unsigned int x_num_min = x_num_min[d-1];
            const unsigned int z_num_min = z_num_min[d-1];

#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,status,part_offset,x_p,y_p,z_p,depth_,status_,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = z_num_min;z_ < z_num_;z_++){

                curr_key = 0;

                //set the key values
                part_data.access_data.pc_key_set_z(curr_key,z_);
                part_data.access_data.pc_key_set_depth(curr_key,d-1);

                for(x_num_min = 0;x_ < x_num_;x_++){

                    part_data.access_data.pc_key_set_x(curr_key,x_);

                    const size_t offset_pc_data = x_num_*z_ + x_;

                    //number of nodes on the level
                    const size_t j_num = part_data.access_data.data[d-1][offset_pc_data].size();

                    y_coord = 0;

                    for(j_ = 0;j_ < j_num;j_++){

                        //this value encodes the state and neighbour locations of the particle cell
                        node_val = part_data.access_data.data[d-1][offset_pc_data][j_];

                        if (!(node_val&1)){
                            //This node represents a particle cell
                            y_coord++;

                            //set the key index
                            part_data.access_data.pc_key_set_j(curr_key,j_);

                            //get all the neighbours
                            status = part_data.access_node_get_status(node_val);

                            if(status == SEED){

                                part_offset = part_data.access_node_get_part_offset(node_val);

                                part_data.access_data.pc_key_set_status(curr_key,status);

                                part_data.access_data.get_coordinates_cell(y_coord,curr_key,x_p,z_p,y_p,depth_,status_);


                                //loop over the particles
                                for(int p = 0;p < part_data.get_num_parts(status);p++){
                                    // get coordinates
                                    part_data.access_data.pc_key_set_index(curr_key,part_offset+p);

                                    curr_k_img.mesh[2*y_p+ y_incr[p] +  curr_k_img.y_num*(2*x_p + x_incr[p]) + curr_k_img.x_num*curr_k_img.y_num*(2*z_p + z_incr[p])] = interp_data.get_part(curr_key);

                                }

                            } else {

                            }

                        } else {
                            //This is a gap node
                            y_coord += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                            y_coord--;
                        }

                    }
                }
            }

        }

        timer.stop_timer();



        //////////////////////////////////////
        //
        //  Get cell info from representation
        //
        ///////////////////////////////////


        const unsigned int x_num_ = x_num[d];
        const unsigned int z_num_ = z_num[d];

        const unsigned int x_num_min = x_num_min[d];
        const unsigned int z_num_min = z_num_min[d];

        timer.start_timer("particle loop");

#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,status,part_offset,x_p,y_p,z_p,depth_,status_,y_coord) if(z_num_*x_num_ > 100)
        for(z_ = z_num_min;z_ < z_num_;z_++){

            curr_key = 0;

            //set the key values
            part_data.access_data.pc_key_set_z(curr_key,z_);
            part_data.access_data.pc_key_set_depth(curr_key,d);

            for(x_ = x_num_min;x_ < x_num_;x_++){

                part_data.access_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                //number of nodes on the level
                const size_t j_num = part_data.access_data.data[d][offset_pc_data].size();

                y_coord = 0;

                for(j_ = 0;j_ < j_num;j_++){

                    //this value encodes the state and neighbour locations of the particle cell
                    node_val = part_data.access_data.data[d][offset_pc_data][j_];

                    if (!(node_val&1)){
                        //This node represents a particle cell
                        y_coord++;

                        //set the key index
                        part_data.access_data.pc_key_set_j(curr_key,j_);

                        //get all the neighbours
                        status = part_data.access_node_get_status(node_val);


                        if(status == SEED){


                        } else {

                            part_offset = part_data.access_node_get_part_offset(node_val);

                            part_data.access_data.pc_key_set_status(curr_key,status);

                            part_data.access_data.get_coordinates_cell(y_coord,curr_key,x_p,z_p,y_p,depth_,status_);

                            part_data.access_data.pc_key_set_index(curr_key,part_offset);
                            curr_k_img.mesh[y_p + curr_k_img.y_num*x_p + curr_k_img.x_num*curr_k_img.y_num*z_p] = interp_data.get_part(curr_key);

                        }

                    } else {
                        //This is a gap node
                        y_coord += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }

                }
            }
        }

        timer.stop_timer();

        /////////////////////////////////////////////////
        //
        //  Place single particles into image
        //
        /////////////////////////////////////////////////


        std::swap(prev_k_img,curr_k_img);

    }

    timer.start_timer("upsample");

    const_upsample_img(curr_k_img,prev_k_img,org_dims);

    timer.stop_timer();


    timer.start_timer("particle loop");

    const unsigned int x_num_ = x_num[depth_max];
    const unsigned int z_num_ = z_num[depth_max];

    const unsigned int x_num_min = x_num_min[depth_max];
    const unsigned int z_num_min = z_num_min[depth_max];

#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,status,part_offset,x_p,y_p,z_p,depth_,status_,y_coord) if(z_num_*x_num_ > 100)
    for(z_ = z_num_min;z_ < z_num_;z_++){

        curr_key = 0;

        //set the key values
        part_data.access_data.pc_key_set_z(curr_key,z_);
        part_data.access_data.pc_key_set_depth(curr_key,depth_max);

        for(x_ = x_num_min;x_ < x_num_;x_++){

            part_data.access_data.pc_key_set_x(curr_key,x_);

            const size_t offset_pc_data = x_num_*z_ + x_;

            //number of nodes on the level
            const size_t j_num = part_data.access_data.data[depth_max][offset_pc_data].size();

            y_coord = 0;

            for(j_ = 0;j_ < j_num;j_++){

                //this value encodes the state and neighbour locations of the particle cell
                node_val = part_data.access_data.data[depth_max][offset_pc_data][j_];

                if (!(node_val&1)){
                    //This node represents a particle cell
                    y_coord++;

                    //set the key index
                    part_data.access_data.pc_key_set_j(curr_key,j_);

                    //get all the neighbours
                    status = part_data.access_node_get_status(node_val);


                    if(status == SEED){

                        part_offset = part_data.access_node_get_part_offset(node_val);

                        part_data.access_data.pc_key_set_status(curr_key,status);

                        part_data.access_data.get_coordinates_cell(y_coord,curr_key,x_p,z_p,y_p,depth_,status_);


                        //loop over the particles
                        for(int p = 0;p < part_data.get_num_parts(status);p++){
                            // get coordinates
                            part_data.access_data.pc_key_set_index(curr_key,part_offset+p);

                            curr_k_img.mesh[2*y_p+ y_incr[p] +  curr_k_img.y_num*(2*x_p + x_incr[p]) + curr_k_img.x_num*curr_k_img.y_num*(2*z_p + z_incr[p])] = interp_data.get_part(curr_key);

                        }

                    } else {

                    }

                } else {
                    //This is a gap node
                    y_coord += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                    y_coord--;
                }

            }
        }
    }


    timer.stop_timer();

    t_n.stop_timer();


}


};






#endif
    
    




