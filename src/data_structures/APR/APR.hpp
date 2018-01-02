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

#include "src/numerics/filter_help/CurrLevel.hpp"

template<typename ImageType>
class APR {

public:

    ParticleDataNew<ImageType, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm

    ExtraPartCellData<uint16_t> y_vec;

    ExtraPartCellData<ImageType> particles_int;

    PartCellData<uint64_t> pc_data;

    CurrentLevel<ImageType,uint64_t> curr_level;

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

    void init_cells(PartCellStructure<float,uint64_t>& pc_struct){
        part_new.initialize_from_structure(pc_struct);

        create_y_data();

        part_new.create_particles_at_cell_structure(particles_int);

        //shift_particles_from_cells(particles_int);

        part_new.initialize_from_structure(pc_struct);

        part_new.create_pc_data_new(pc_data);

        curr_level.init(pc_data);

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


    uint64_t begin(){

        return curr_level.init_iterate(pc_data);

    }

    uint64_t begin(unsigned int depth){

    }


    uint64_t end(){
        return curr_level.counter > 0;
    }

    uint64_t end(unsigned int depth){

    }

    uint64_t it_forward(){

        curr_level.move_to_next_pc(pc_data);

        return curr_level.counter;
    }

    template<typename S>
    void get_neigh_all(ExtraPartCellData<S>& parts,std::vector<std::vector<S>>& neigh_vec){
        //
        // gets all particle neighbours and returns them in a vector with 6 vectors of the particles on each face of the Particle Cell
        //

        curr_level.update_and_get_neigh_all(parts,pc_data,neigh_vec);

    }

    template<typename S>
    void get_neigh_all_avg(ExtraPartCellData<S>& parts,std::vector<std::vector<S>>& neigh_vec){
        //
        // gets all particle neighbours and returns them in a vector with 6 vectors, if exists provides the average of the neighbours
        //

        curr_level.update_and_get_neigh_all_avg(parts,pc_data,neigh_vec);

    }

    template<typename S>
    void get_neigh_dir(ExtraPartCellData<S>& parts,std::vector<S>& neigh_vec,unsigned int dir){
        //
        // gets all particle neighbours and returns them in a vector with 6 vectors, if exists provides the average of the neighbours
        //

        curr_level.update_get_neigh_dir(parts,pc_data,neigh_vec,dir);

    }

    template<typename S>
    S& operator()(ExtraPartCellData<S>& parts){
        return curr_level.get_val(parts);

    }


    ////////////////////////
    //
    //  Accessing info when iterating. Does not make sense outisde of a looping structure
    //
    //////////////////////////

    inline unsigned int x(){
        //get x
        return curr_level.x;
    }

    inline unsigned int y(){
        //get x
        return curr_level.y;
    }

    inline unsigned int z(){
        //get x
        return curr_level.z;
    }

    inline unsigned int type(){
        //get x
        return curr_level.status;
    }

    inline unsigned int depth(){
        //get x
        return curr_level.depth;
    }

    inline unsigned int x_nearest_pixel(){
        //get x
        return floor((curr_level.x+0.5)*pow(2, pc_data.depth_max - curr_level.depth));
    }

    inline float x_global(){
        //get x
        return (curr_level.x+0.5)*pow(2, pc_data.depth_max - curr_level.depth);
    }

    inline unsigned int y_nearest_pixel(){
        //get x
        return floor((curr_level.y+0.5)*pow(2, pc_data.depth_max - curr_level.depth));
    }

    inline float y_global(){
        //get x
        return (curr_level.y+0.5)*pow(2, pc_data.depth_max - curr_level.depth);
    }

    inline unsigned int z_nearest_pixel(){
        //get x
        return floor((curr_level.z+0.5)*pow(2, pc_data.depth_max - curr_level.depth));
    }

    inline float z_global(){
        //get x
        return (curr_level.z+0.5)*pow(2, pc_data.depth_max - curr_level.depth);
    }

    template<typename U,typename V>
    void interp_img(Mesh_data<U>& img,ExtraPartCellData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        img.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],0);

        int z_,x_,j_,y_;

        for(uint64_t depth = (pc_data.depth_min);depth <= pc_data.depth_max;depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[depth];
            const unsigned int z_num_ = pc_data.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            CurrentLevel<float, uint64_t> curr_level_l(pc_data);
            curr_level_l.set_new_depth(depth, pc_data);

            const float step_size = pow(2,curr_level_l.depth_max - curr_level_l.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level_l) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    curr_level_l.set_new_xz(x_, z_, pc_data);

                    for (j_ = 0; j_ < curr_level_l.j_num; j_++) {

                        bool iscell = curr_level_l.new_j(j_, pc_data);

                        if (iscell) {
                            //Indicates this is a particle cell node
                            curr_level_l.update_cell(pc_data);

                            int dim1 = curr_level_l.y * step_size;
                            int dim2 = curr_level_l.x * step_size;
                            int dim3 = curr_level_l.z * step_size;

                            float temp_int;
                            //add to all the required rays

                            temp_int = curr_level_l.get_val(parts);

                            const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                            const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                            const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                            for (int q = dim3; q < offset_max_dim3; ++q) {

                                for (int k = dim2; k < offset_max_dim2; ++k) {
#pragma omp simd
                                    for (int i = dim1; i < offset_max_dim1; ++i) {
                                        img.mesh[i + (k) * img.y_num + q*img.y_num*img.x_num] = temp_int;
                                    }
                                }
                            }


                        } else {

                            curr_level_l.update_gap(pc_data);

                        }


                    }
                }
            }
        }




    }


    template<typename U>
    void interp_depth(Mesh_data<U>& img){

        //get depth
        ExtraPartCellData<U> depth_parts;
        depth_parts.initialize_structure_cells(pc_data);

        for (begin(); end() == true ; it_forward()) {
            //
            //  Demo APR iterator
            //

            //access and info
            curr_level.get_val(depth_parts) = this->depth();

        }

        interp_img(img,depth_parts);


    }

    template<typename U>
    void interp_type(Mesh_data<U>& img){

        //get depth
        ExtraPartCellData<U> type_parts;
        type_parts.initialize_structure_cells(pc_data);

        for (begin(); end() == true ; it_forward()) {
            //
            //  Demo APR iterator
            //

            //access and info
            curr_level.get_val(type_parts) = this->type();

        }

        interp_img(img,type_parts);


    }


    template<typename U,typename V>
    void interp_parts_smooth(Mesh_data<U>& out_image,ExtraPartCellData<V>& interp_data,std::vector<float> scale_d = {1,1,1}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        Part_timer timer;
        timer.verbose_flag = true;

        Mesh_data<U> pc_image;
        Mesh_data<uint8_t> k_img;

        interp_img(pc_image,interp_data);

        interp_depth(k_img);

        int filter_offset = 0;

        unsigned int x_num = pc_image.x_num;
        unsigned int y_num = pc_image.y_num;
        unsigned int z_num = pc_image.z_num;

        Mesh_data<U> output_data;
        output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

        std::vector<U> temp_vec;
        temp_vec.resize(y_num,0);

        uint64_t offset_min;
        uint64_t offset_max;

        uint64_t j = 0;
        uint64_t k = 0;
        uint64_t i = 0;

        float factor = 0;

        int k_max = pc_data.depth_max;

        timer.start_timer("x direction");

        std::copy(k_img.mesh.begin(),k_img.mesh.end(),output_data.mesh.begin());

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max,filter_offset,factor)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    filter_offset = floor(pow(2,k_max - output_data.mesh[j*x_num*y_num + i*y_num + k])/scale_d[0]);

                    offset_max = std::min((int)(k + filter_offset),(int)(y_num-1));
                    offset_min = std::max((int)(k - filter_offset),(int)0);

                    factor = 1.0/(offset_max - offset_min+1);

                    uint64_t f = 0;
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        output_data.mesh[j*x_num*y_num + i*y_num + k] += pc_image.mesh[j*x_num*y_num + i*y_num + c]*factor;

                    }

                }
            }
        }

        timer.stop_timer();

        timer.start_timer("y direction");

        std::swap(output_data.mesh,pc_image.mesh);

        std::copy(k_img.mesh.begin(),k_img.mesh.end(),output_data.mesh.begin());

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max,filter_offset,factor)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    filter_offset = floor(pow(2,k_max - output_data.mesh[j*x_num*y_num + i*y_num + k])/scale_d[1]);

                    offset_max = std::min((int)(i + filter_offset),(int)(x_num-1));
                    offset_min = std::max((int)(i - filter_offset),(int)0);

                    factor = 1.0/(offset_max - offset_min+1);

                    uint64_t f = 0;
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        output_data.mesh[j*x_num*y_num + i*y_num + k] += pc_image.mesh[j*x_num*y_num + c*y_num + k]*factor;
                    }

                }
            }
        }

//
//
//    // z loop
//

        timer.stop_timer();

        std::swap(output_data.mesh,pc_image.mesh);

        timer.start_timer("z direction");

        std::copy(k_img.mesh.begin(),k_img.mesh.end(),output_data.mesh.begin());

#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max,filter_offset,factor)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){


                for(k = 0;k < y_num;k++){

                    filter_offset = floor(pow(2,k_max - output_data.mesh[j*x_num*y_num + i*y_num + k])/scale_d[2]);

                    offset_max = std::min((int)(j + filter_offset),(int)(z_num-1));
                    offset_min = std::max((int)(j - filter_offset),(int)0);

                    factor = 1.0/(offset_max - offset_min+1);

                    uint64_t f = 0;
                    output_data.mesh[j*x_num*y_num + i*y_num + k]=0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        output_data.mesh[j*x_num*y_num + i*y_num + k] += pc_image.mesh[c*x_num*y_num + i*y_num + k]*factor;

                    }

                }
            }
        }

        timer.stop_timer();

        out_image = output_data;

    }



};

#endif //PARTPLAY_APR_HPP
