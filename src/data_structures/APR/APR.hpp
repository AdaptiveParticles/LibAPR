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

    std::vector<float> num_parts;

    std::vector<float> num_elements;

    double num_parts_total;
    double num_elements_total;

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
        create_pc_data_new(pc_struct);

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
        //accesses the value of particle data when iterating
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



    template<typename U,typename V>
    void get_parts_from_img(Mesh_data<U>& img,ExtraPartCellData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using the nearest pixel (rounded up, i.e. next pixel after particles that sit on off pixel locations)
        //

        parts.initialize_structure_cells(pc_data);

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

                            int dim1 = floor((curr_level_l.y+0.5) * step_size);
                            int dim2 = floor((curr_level_l.x+0.5) * step_size);
                            int dim3 = floor((curr_level_l.z+0.5) * step_size);

                            curr_level_l.get_val(parts) = img(dim1,dim2,dim3);

                        } else {

                            curr_level_l.update_gap(pc_data);

                        }


                    }
                }
            }
        }




    }

    template<typename U,typename V>
    void get_parts_from_img(std::vector<Mesh_data<U>>& img_by_level,ExtraPartCellData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using an image tree (img_by_level is a vector of images)
        //

        parts.initialize_structure_cells(pc_data);

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

                            int dim1 = floor((curr_level_l.y+0.5) * step_size);
                            int dim2 = floor((curr_level_l.x+0.5) * step_size);
                            int dim3 = floor((curr_level_l.z+0.5) * step_size);

                            curr_level_l.get_val(parts) = img_by_level[depth](curr_level_l.y,curr_level_l.x,curr_level_l.z);

                        } else {

                            curr_level_l.update_gap(pc_data);

                        }


                    }
                }
            }
        }




    }


    void get_part_numbers() {
        //
        //  Computes totals of total number of particles, and the total number of elements (PC and gap nodes)
        //



        this->num_parts.resize(pc_data.depth_max + 1);
        this->num_elements.resize(pc_data.depth_max + 1);

        int z_, x_, j_, y_;

        for (uint64_t depth = (pc_data.depth_min); depth <= pc_data.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[depth];
            const unsigned int z_num_ = pc_data.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            CurrentLevel<float, uint64_t> curr_level_l(pc_data);
            curr_level_l.set_new_depth(depth, pc_data);

            const float step_size = pow(2, curr_level_l.depth_max - curr_level_l.depth);

            uint64_t counter_parts = 0;
            uint64_t counter_elements = 0;

            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    curr_level_l.set_new_xz(x_, z_, pc_data);

                    for (j_ = 0; j_ < curr_level_l.j_num; j_++) {

                        bool iscell = curr_level_l.new_j(j_, pc_data);

                        counter_elements++;

                        if (iscell) {
                            //Indicates this is a particle cell node
                            curr_level_l.update_cell(pc_data);

                            counter_parts++;

                        } else {

                            curr_level_l.update_gap(pc_data);

                        }


                    }
                }
            }

            this->num_parts[depth] = counter_parts;
            this->num_elements[depth] = counter_elements;

        }

        num_parts_total = 0;
        num_elements_total = 0;

        for (int i = 0; i < pc_data.depth_max; ++i) {
            num_parts_total += num_parts[i];
            num_elements_total += num_elements[i];
        }

    }


        template<typename U,typename V>
    void create_pc_data_new(PartCellStructure<U,V>& pc_struct){
        //
        //
        //  Moves from the old data structure to the new datastructure (PC = particles, stores V)
        //
        //

        pc_data.org_dims = pc_struct.org_dims;

        pc_data.y_num = pc_struct.y_num;

        //first add the layers
        pc_data.depth_max = pc_struct.depth_max + 1;
        pc_data.depth_min = pc_struct.depth_min;

        pc_data.z_num.resize(pc_data.depth_max+1);
        pc_data.x_num.resize(pc_data.depth_max+1);

        pc_data.y_num.resize(pc_data.depth_max+1);

        pc_data.y_num[pc_data.depth_max] = pc_struct.org_dims[0];

        pc_data.data.resize(pc_data.depth_max+1);

        for(uint64_t i = pc_data.depth_min;i < pc_data.depth_max;i++){
            pc_data.z_num[i] = pc_struct.z_num[i];
            pc_data.x_num[i] = pc_struct.x_num[i];
            pc_data.y_num[i] = pc_struct.y_num[i];
            pc_data.data[i].resize(pc_struct.z_num[i]*pc_struct.x_num[i]);
        }

        pc_data.z_num[pc_data.depth_max] = pc_struct.org_dims[2];
        pc_data.x_num[pc_data.depth_max] = pc_struct.org_dims[1];
        pc_data.y_num[pc_data.depth_max] = pc_struct.org_dims[0];
        pc_data.data[pc_data.depth_max].resize(pc_data.z_num[pc_data.depth_max]*pc_data.x_num[pc_data.depth_max]);

        particles_int.org_dims = pc_struct.org_dims;

        particles_int.y_num = pc_struct.y_num;

        //first add the layers
        particles_int.depth_max = pc_struct.depth_max + 1;
        particles_int.depth_min = pc_struct.depth_min;

        particles_int.z_num.resize(pc_data.depth_max+1);
        particles_int.x_num.resize(pc_data.depth_max+1);

        particles_int.y_num.resize(pc_data.depth_max+1);

        particles_int.y_num[pc_data.depth_max] = pc_struct.org_dims[0];

        particles_int.data.resize(pc_data.depth_max+1);

        for(uint64_t i = particles_int.depth_min;i <= particles_int.depth_max;i++){
            particles_int.z_num[i] = pc_data.z_num[i];
            particles_int.x_num[i] = pc_data.x_num[i];
            particles_int.y_num[i] = pc_data.y_num[i];
            particles_int.data[i].resize(pc_data.z_num[i]*pc_data.x_num[i]);
        }

        //now initialize the entries of the two data sets, access structure

        //initialize loop variables
        int x_;
        int z_;
        int y_;

        int x_seed;
        int z_seed;
        int y_seed;

        uint64_t j_;

        uint64_t status;
        uint64_t node_val;
        uint16_t node_val_part;

        //next initialize the entries;
        Part_timer timer;
        timer.verbose_flag = false;

        std::vector<uint16_t> temp_exist;
        std::vector<uint16_t> temp_location;

        std::vector<U> temp_int;

        timer.start_timer("intiialize access data structure");

        for(uint64_t i = pc_data.depth_max;i >= pc_data.depth_min;i--){

            const unsigned int x_num = pc_data.x_num[i];
            const unsigned int z_num = pc_data.z_num[i];


            const unsigned int x_num_seed = pc_data.x_num[i-1];
            const unsigned int z_num_seed = pc_data.z_num[i-1];

            temp_exist.resize(pc_data.y_num[i]);
            temp_location.resize(pc_data.y_num[i]);
            temp_int.resize(pc_data.y_num[i]);

#pragma omp parallel for default(shared) private(j_,z_,x_,y_,node_val,status,z_seed,x_seed,node_val_part) firstprivate(temp_exist,temp_location) if(z_num*x_num > 100)
            for(z_ = 0;z_ < z_num;z_++){

                for(x_ = 0;x_ < x_num;x_++){

                    std::fill(temp_exist.begin(), temp_exist.end(), 0);
                    std::fill(temp_location.begin(), temp_location.end(), 0);

                    std::fill(temp_int.begin(), temp_int.end(), 0);

                    if( i < pc_data.depth_max){
                        //access variables
                        const size_t offset_pc_data = x_num*z_ + x_;
                        const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();

                        y_ = 0;

                        //first loop over
                        for(j_ = 0; j_ < j_num;j_++){
                            //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                            node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                            node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];

                            if(!(node_val&1)){
                                //normal node
                                y_++;
                                //create pindex, and create status (0,1,2,3) and type
                                status = (node_val & STATUS_MASK) >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.
                                uint16_t part_offset = (node_val_part & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;

                                if(status > SEED){
                                    temp_exist[y_] = status;
                                    temp_location[y_] = part_offset;
                                }

                            } else {

                                y_ = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_--;
                            }
                        }
                    }

                    x_seed = x_/2;
                    z_seed = z_/2;

                    if( i > pc_data.depth_min){
                        //access variables
                        size_t offset_pc_data = x_num_seed*z_seed + x_seed;
                        const size_t j_num = pc_struct.pc_data.data[i-1][offset_pc_data].size();


                        y_ = 0;

                        //first loop over
                        for(j_ = 0; j_ < j_num;j_++){
                            //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                            node_val_part = pc_struct.part_data.access_data.data[i-1][offset_pc_data][j_];
                            node_val = pc_struct.pc_data.data[i-1][offset_pc_data][j_];

                            if(!(node_val&1)){
                                //normal node
                                y_++;
                                //create pindex, and create status (0,1,2,3) and type
                                status = (node_val & STATUS_MASK) >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.
                                uint16_t part_offset = (node_val_part & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;

                                if(status == SEED){
                                    temp_exist[2*y_] = status;
                                    temp_exist[2*y_+1] = status;

                                    temp_location[2*y_] = part_offset + (z_&1)*4 + (x_&1)*2;
                                    temp_location[2*y_+1] = part_offset + (z_&1)*4 + (x_&1)*2 + 1;

                                }

                            } else {

                                y_ = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                y_--;
                            }
                        }
                    }


                    size_t first_empty = 0;

                    size_t offset_pc_data = x_num*z_ + x_;
                    size_t offset_pc_data_seed = x_num_seed*z_seed + x_seed;
                    size_t curr_index = 0;
                    size_t prev_ind = 0;

                    //first value handle the duplication of the gap node

                    status = temp_exist[0];

                    if((status> 0)){
                        first_empty = 0;
                    } else {
                        first_empty = 1;
                    }

                    size_t part_total= 0;

                    for(y_ = 0;y_ < temp_exist.size();y_++){

                        status = temp_exist[y_];

                        if(status> 0){
                            curr_index+= 1 + prev_ind;
                            prev_ind = 0;
                            part_total++;
                        } else {
                            prev_ind = 1;
                        }
                    }

                    if(curr_index == 0){
                        pc_data.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                    } else {
                        pc_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node

                    }



                    curr_index = 0;
                    prev_ind = 1;
                    size_t prev_coord = 0;

                    size_t part_counter=0;

                    pc_data.data[i][offset_pc_data][0] = 1;
                    pc_data.data[i][offset_pc_data].back() = 1;

                    //initialize particles
                    particles_int.data[i][offset_pc_data].resize(pc_data.data[i][offset_pc_data].size());

                    for(y_ = 0;y_ < temp_exist.size();y_++){

                        status = temp_exist[y_];

                        if((status> 0)){

                            curr_index++;

                            //set starting type
                            if(prev_ind == 1){
                                //gap node
                                //set type

                                pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_GAP;
                                pc_data.data[i][offset_pc_data][curr_index-1] |= (((uint64_t)y_) << NEXT_COORD_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index-1] |= ( prev_coord << PREV_COORD_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);
                                //CHANGE HERE

                                //gap node
                               // pc_data.data[i][offset_pc_data][curr_index-1] = 1; //set type to gap
                               // pc_data.data[i][offset_pc_data][curr_index-1] |= ((y_ - prev_coord) << COORD_DIFF_SHIFT_PARTICLE); //set the coordinate difference

                                curr_index++;
                            }
                            prev_coord = y_;
                            //set type


                            pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_PC;

                            //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                            pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);

                            pc_data.data[i][offset_pc_data][curr_index-1] |= (status << STATUS_SHIFT);

                            //lastly retrieve the intensities
                            if(status == SEED){
                                //seed from up one level
                                particles_int.data[i][offset_pc_data][curr_index-1] = pc_struct.part_data.particle_data.data[i-1][offset_pc_data_seed][temp_location[y_]];
                            }
                            else {
                                //non seed same level
                                particles_int.data[i][offset_pc_data][curr_index-1] = pc_struct.part_data.particle_data.data[i][offset_pc_data][temp_location[y_]];
                            }

                            part_counter++;


                            prev_ind = 0;
                        } else {
                            //store for setting above
                            if(prev_ind == 0){
                                //prev_coord = y_;
                            }

                            prev_ind = 1;

                        }
                    }


                    int stop = 1;



                }

            }
        }

        timer.stop_timer();


        ///////////////////////////////////
        //
        //  Calculate neighbours
        //
        /////////////////////////////////

        //(+y,-y,+x,-x,+z,-z)
        pc_data.set_neighbor_relationships();

        get_part_numbers();

    }

};




#endif //PARTPLAY_APR_HPP
