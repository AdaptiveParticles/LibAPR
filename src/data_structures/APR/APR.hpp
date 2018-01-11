//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "benchmarks/development/Tree/PartCellStructure.hpp"

//#include "benchmarks/development/old_numerics/filter_numerics.hpp"
//#include "benchmarks/development/old_numerics/misc_numerics.hpp"

#include "src/data_structures/APR/PartCellData.hpp"

//#include "benchmarks/development/Tree/PartCellStructure.hpp"

#include "CurrLevel.hpp"

#include "src/io/hdf5functions_blosc.h"

#include "src/data_structures/APR/APR_iterator.hpp"

#include "src/misc/APR_timer.hpp"

#include "src/algorithm/APR_parameters.hpp"

#include <map>
#include <unordered_map>

class APR_parameters;

typedef std::unordered_map<uint16_t,uint16_t> hash_map;
//typedef std::map<uint16_t,uint16_t> hash_map;

template<typename ImageType>
class APR : public APR_iterator<ImageType>{

public:


    //Main internal datastructures

    ExtraPartCellData<ImageType> particles_int; // holds the particles intenisty information

    PartCellData<uint64_t> pc_data; // holds the spatial and neighbours access information and methods

    //used for storing number of paritcles and cells per level for parallel access iterators
    std::vector<float> num_parts;
    std::vector<float> num_elements;
    ExtraPartCellData<uint64_t> num_parts_xy;
    double num_elements_total;

    std::string name;
    APR_parameters parameters;


    //old parameters (depreciated)
    Proc_par pars;

    //Experimental
    ExtraPartCellData<hash_map> random_access;

    APR(){
        this->pc_data_pointer = &pc_data;
    }

    APR(PartCellStructure<float,uint64_t>& pc_struct){
        init_cells(pc_struct);
        this->pc_data_pointer = &pc_data;
    }

    //deprecitated
    ExtraPartCellData<uint16> y_vec;


    void init_cells(PartCellStructure<float,uint64_t>& pc_struct){
        create_pc_data_new(pc_struct);

        this->curr_level.init(pc_data);

    }

    void init_random_access(){


        random_access.initialize_structure_parts_empty(particles_int);

        ExtraPartCellData<std::pair<uint16_t,uint16_t>> hash_init;

        hash_init.initialize_structure_parts_empty(particles_int);

        int counter = 0;

        //create the intiializer lists

        //loop over all particles
        for (this->begin(); this->end() == true; this->it_forward()) {

            hash_init.data[this->depth()][this->curr_level.pc_offset].push_back({this->y(),this->j()});

        }

        //now create the actual hash tables
        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++) {

            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];

            for (uint64_t z_ = 0; z_ < z_num_; z_++) {

                for (uint64_t x_ = 0; x_ < x_num_; x_++) {
                    const uint64_t offset_pc_data = x_num_ * z_ + x_;
                    if(hash_init.data[i][offset_pc_data].size() > 0) {
                        random_access.data[i][offset_pc_data].resize(1);

                        random_access.data[i][offset_pc_data][0].insert(hash_init.data[i][offset_pc_data].begin(),
                                                                        hash_init.data[i][offset_pc_data].end());
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

        pdata_new.initialize_structure_cells(pc_data);

        uint64_t z_,x_,j_,node_val;
        uint64_t part_offset;

        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];

#pragma omp parallel for default(shared) private(z_,x_,j_,node_val)  if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t j_num = pc_data.data[i][offset_pc_data].size();

                    int counter = 0;

                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val = pc_data.data[i][offset_pc_data][j_];

                        if(!(node_val&1)){

                            pdata_new.data[i][offset_pc_data][counter] = pdata_old.data[i][offset_pc_data][j_];

                            counter++;

                        } else {

                        }

                    }

                    pdata_new.data[i][offset_pc_data].resize(counter); //reduce to new size
                }
            }
        }

        std::swap(pdata_new,pdata_old);

    }

    uint64_t begin(){

        return this->curr_level.init_iterate(pc_data);

    }

    uint64_t begin(unsigned int depth){
        return this->curr_level.init_iterate(pc_data,depth);
    }

    uint64_t end(){
        return this->curr_level.counter > 0;
    }

    uint64_t end(unsigned int depth){
        return this->curr_level.counter > 0;
    }

    uint64_t it_forward(){

        this->curr_level.move_to_next_pc(pc_data);

        return this->curr_level.counter;
    }

    uint64_t it_forward(unsigned int depth){

        this->curr_level.move_to_next_pc(pc_data,depth);

        return this->curr_level.counter;
    }



    ////////////////////////
    //
    //  Accessing info when iterating. Does not make sense outisde of a looping structure
    //
    //////////////////////////





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
    void interp_depth_ds(Mesh_data<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        //get depth
        ExtraPartCellData<U> depth_parts;
        depth_parts.initialize_structure_cells(pc_data);

        for (begin(); end() == true ; it_forward()) {
            //
            //  Demo APR iterator
            //

            //access and info
            this->curr_level.get_val(depth_parts) = this->depth();

        }

        Mesh_data<U> temp;

        interp_img(temp,depth_parts);

        down_sample(temp,img,
                    [](U x, U y) { return std::max(x,y); },
                    [](U x) { return x; }, true);

    }

    template<typename U>
    void interp_depth(Mesh_data<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        //get depth
        ExtraPartCellData<U> depth_parts;
        depth_parts.initialize_structure_cells(pc_data);

        for (begin(); end() == true ; it_forward()) {
            //
            //  Demo APR iterator
            //

            //access and info
            this->curr_level.get_val(depth_parts) = this->depth();

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
            this->curr_level.get_val(type_parts) = this->type();

        }

        interp_img(img,type_parts);


    }

    template<typename T>
    void calc_sat_adaptive_y(Mesh_data<T>& input,Mesh_data<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in){
        //
        //  Bevan Cheeseman 2016
        //
        //  Calculates a O(1) recursive mean using SAT.
        //


        const int z_num = input.z_num;
        const int x_num = input.x_num;
        const int y_num = input.y_num;

        std::vector<T> temp_vec;
        temp_vec.resize(y_num,0);

        std::vector<T> offset_vec;
        offset_vec.resize(y_num,0);


        int i, k, index;
        float counter, temp, divisor,offset;

        //need to introduce an offset max to make the algorithm still work, and it also makes sense.
        const int offset_max = offset_max_in;

        float scale = scale_in;

        const unsigned int d_max = this->depth_max();

#pragma omp parallel for default(shared) private(i,k,counter,temp,index,divisor,offset) firstprivate(temp_vec,offset_vec)
        for(int j = 0;j < z_num;j++){
            for(i = 0;i < x_num;i++){

                index = j*x_num*y_num + i*y_num;

                //first update the fixed length scale
                for (k = 0; k < y_num;k++){

                    offset_vec[k] = std::min((T)floor(pow(2,d_max- offset_img.mesh[index + k])/scale),(T)offset_max);

                }


                //first pass over and calculate cumsum
                temp = 0;
                for (k = 0; k < y_num;k++){
                    temp += input.mesh[index + k];
                    temp_vec[k] = temp;
                }

                input.mesh[index] = 0;
                //handling boundary conditions (LHS)
                for (k = 1; k <= (offset_max+1);k++){
                    divisor = 2*offset_vec[k] + 1;
                    offset = offset_vec[k];

                    if(k <= (offset+1)){
                        //emulate the bc
                        input.mesh[index + k] = -temp_vec[0]/divisor;
                    }

                }

                //second pass calculate mean
                for (k = 0; k < y_num;k++){
                    divisor = 2*offset_vec[k] + 1;
                    offset = offset_vec[k];
                    if(k >= (offset+1)){
                        input.mesh[index + k] = -temp_vec[k - offset - 1]/divisor;
                    }

                }


                //second pass calculate mean
                for (k = 0; k < (y_num);k++){
                    divisor = 2*offset_vec[k] + 1;
                    offset = offset_vec[k];
                    if(k < (y_num - offset)) {
                        input.mesh[index + k] += temp_vec[k + offset] / divisor;
                    }
                }


                counter = 0;
                //handling boundary conditions (RHS)
                for (k = ( y_num - offset_max); k < (y_num);k++){

                    divisor = 2*offset_vec[k] + 1;
                    offset = offset_vec[k];

                    if(k >= (y_num - offset)){
                        counter = k - (y_num-offset)+1;

                        input.mesh[index + k]*= divisor;
                        input.mesh[index + k]+= temp_vec[y_num-1];
                        input.mesh[index + k]*= 1.0/(divisor - counter);

                    }

                }

                //handling boundary conditions (LHS), need to rehandle the boundary
                for (k = 1; k < (offset_max + 1);k++){

                    divisor = 2*offset_vec[k] + 1;
                    offset = offset_vec[k];

                    if(k < (offset + 1)){
                        input.mesh[index + k] *= divisor/(1.0*k + offset);
                    }

                }

                //end point boundary condition
                divisor = 2*offset_vec[0] + 1;
                offset = offset_vec[0];
                input.mesh[index] *= divisor/(offset+1);
            }
        }



    }
    template<typename T>
    void calc_sat_adaptive_x(Mesh_data<T>& input,Mesh_data<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in){
        //
        //  Adaptive form of Matteusz' SAT code.
        //
        //

        const int z_num = input.z_num;
        const int x_num = input.x_num;
        const int y_num = input.y_num;

        unsigned int offset_max = offset_max_in;

        std::vector<T> temp_vec;
        temp_vec.resize(y_num*(2*offset_max + 2),0);

        int i,k;
        float temp;
        int index_modulo, previous_modulo, current_index, jxnumynum, offset,forward_modulo,backward_modulo;

        const float scale = scale_in;
        const unsigned int d_max = this->depth_max();


#pragma omp parallel for default(shared) private(i,k,temp,index_modulo, previous_modulo, forward_modulo,backward_modulo,current_index, jxnumynum,offset) \
        firstprivate(temp_vec)
        for(int j = 0; j < z_num; j++) {

            jxnumynum = j * x_num * y_num;

            //prefetching

            for(k = 0; k < y_num ; k++){
                // std::copy ?
                temp_vec[k] = input.mesh[jxnumynum + k];
            }


            for(i = 1; i < 2 * offset_max + 1; i++) {
                for(k = 0; k < y_num; k++) {
                    temp_vec[i*y_num + k] = input.mesh[jxnumynum + i*y_num + k] + temp_vec[(i-1)*y_num + k];
                }
            }


            // LHS boundary

            for(i = 0; i < offset_max + 1; i++){
                for(k = 0; k < y_num; k++) {
                    offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[jxnumynum + i * y_num + k])/scale),(T)offset_max);
                    if(i < (offset + 1)) {
                        input.mesh[jxnumynum + i * y_num + k] = (temp_vec[(i + offset) * y_num + k]) / (i + offset + 1);
                    }
                }
            }

            // middle

            //for(i = offset + 1; i < x_num - offset; i++){

            for(i = 1; i < x_num ; i++){
                // the current cumsum

                for(k = 0; k < y_num; k++) {


                    offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[jxnumynum + i * y_num + k])/scale),(T)offset_max);

                    if((i >= offset_max + 1) & (i < (x_num - offset_max))) {
                        // update the buffers

                        index_modulo = (i + offset_max) % (2 * offset_max + 2);
                        previous_modulo = (i + offset_max - 1) % (2 * offset_max + 2);
                        temp = input.mesh[jxnumynum + (i + offset_max) * y_num + k] + temp_vec[previous_modulo * y_num + k];
                        temp_vec[index_modulo * y_num + k] = temp;

                    }

                    //perform the mean calculation
                    if((i >= offset+ 1) & (i < (x_num - offset))) {
                        // calculate the positions in the buffers
                        forward_modulo = (i + offset) % (2 * offset_max + 2);
                        backward_modulo = (i - offset - 1) % (2 * offset_max + 2);
                        input.mesh[jxnumynum + i * y_num + k] = (temp_vec[forward_modulo * y_num + k] - temp_vec[backward_modulo * y_num + k]) /
                                                                (2 * offset + 1);

                    }
                }

            }

            // RHS boundary //circular buffer

            for(i = x_num - offset_max; i < x_num; i++){

                for(k = 0; k < y_num; k++){

                    offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[jxnumynum + i * y_num + k])/scale),(T)offset_max);

                    if(i >= (x_num - offset)){
                        // calculate the positions in the buffers
                        backward_modulo  = (i - offset - 1) % (2 * offset_max + 2); //maybe the top and the bottom different
                        forward_modulo = (x_num - 1) % (2 * offset_max + 2); //reached the end so need to use that

                        input.mesh[jxnumynum + i * y_num + k] = (temp_vec[forward_modulo * y_num + k] -
                                                                 temp_vec[backward_modulo * y_num + k]) /
                                                                (x_num - i + offset);
                    }
                }
            }
        }


    }


    template<typename T>
    void calc_sat_adaptive_z(Mesh_data<T>& input,Mesh_data<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in){

        // The same, but in place

        const int z_num = input.z_num;
        const int x_num = input.x_num;
        const int y_num = input.y_num;

        int j,k;
        float temp;
        int index_modulo, previous_modulo, current_index, iynum,forward_modulo,backward_modulo,offset;
        int xnumynum = x_num * y_num;

        const int offset_max = offset_max_in;
        const float scale = scale_in;
        const unsigned int d_max = this->depth_max();

        std::vector<T> temp_vec;
        temp_vec.resize(y_num*(2*offset_max + 2),0);

#pragma omp parallel for default(shared) private(j,k,temp,index_modulo, previous_modulo, current_index,backward_modulo,forward_modulo, iynum,offset) \
        firstprivate(temp_vec)
        for(int i = 0; i < x_num; i++) {

            iynum = i * y_num;

            //prefetching

            for(k = 0; k < y_num ; k++){
                // std::copy ?
                temp_vec[k] = input.mesh[iynum + k];
            }

            //(updated z)
            for(j = 1; j < 2 * offset_max+ 1; j++) {
                for(k = 0; k < y_num; k++) {
                    temp_vec[j*y_num + k] = input.mesh[j * xnumynum + iynum + k] + temp_vec[(j-1)*y_num + k];
                }
            }

            // LHS boundary (updated)
            for(j = 0; j < offset_max + 1; j++){
                for(k = 0; k < y_num; k++) {
                    offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[j * xnumynum + iynum + k])/scale),(T)offset_max);
                    if(i < (offset + 1)) {
                        input.mesh[j * xnumynum + iynum + k] = (temp_vec[(j + offset) * y_num + k]) / (j + offset + 1);
                    }
                }
            }

            // middle
            for(j = 1; j < z_num ; j++){

                for(k = 0; k < y_num; k++) {

                    offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[j * xnumynum + iynum + k])/scale),(T)offset_max);

                    //update the buffer
                    if((j >= offset_max + 1) & (j < (z_num - offset_max))) {

                        index_modulo = (j + offset_max) % (2 * offset_max + 2);
                        previous_modulo = (j + offset_max - 1) % (2 * offset_max + 2);

                        // the current cumsum
                        temp = input.mesh[(j + offset_max) * xnumynum + iynum + k] + temp_vec[previous_modulo*y_num + k];
                        temp_vec[index_modulo*y_num + k] = temp;
                    }

                    if((j >= offset+ 1) & (j < (z_num - offset))) {
                        // calculate the positions in the buffers
                        forward_modulo = (j + offset) % (2 * offset_max + 2);
                        backward_modulo = (j - offset - 1) % (2 * offset_max + 2);

                        input.mesh[j * xnumynum + iynum + k] =
                                (temp_vec[forward_modulo * y_num + k] - temp_vec[backward_modulo * y_num + k]) /
                                (2 * offset + 1);

                    }
                }
            }

            // RHS boundary

            for(j = z_num - offset_max; j < z_num; j++){
                for(k = 0; k < y_num; k++){

                    offset = std::min((T)floor(pow(2,d_max- offset_img.mesh[j * xnumynum + iynum + k])/scale),(T)offset_max);

                    if(j >= (z_num - offset)){
                        //calculate the buffer offsets
                        backward_modulo  = (j - offset - 1) % (2 * offset_max + 2); //maybe the top and the bottom different
                        forward_modulo = (z_num - 1) % (2 * offset_max + 2); //reached the end so need to use that

                        input.mesh[j * xnumynum + iynum + k] = (temp_vec[forward_modulo*y_num + k] -
                                                                temp_vec[backward_modulo*y_num + k]) / (z_num - j + offset);

                    }
                }

            }


        }

    }




    template<typename U,typename V>
    void interp_parts_smooth(Mesh_data<U>& out_image,ExtraPartCellData<V>& interp_data,std::vector<float> scale_d = {2,2,2}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        Part_timer timer;
        timer.verbose_flag = false;

        Mesh_data<U> pc_image;
        Mesh_data<uint8_t> k_img;

        unsigned int offset_max = 20;

        interp_img(pc_image,interp_data);

        interp_depth(k_img);

        timer.start_timer("sat");
        //demo
        calc_sat_adaptive_y(pc_image,k_img,scale_d[0],offset_max);

        timer.stop_timer();

        timer.start_timer("sat");

        calc_sat_adaptive_x(pc_image,k_img,scale_d[1],offset_max);

        timer.stop_timer();

        timer.start_timer("sat");

        calc_sat_adaptive_z(pc_image,k_img,scale_d[2],offset_max);

        timer.stop_timer();


        std::swap(pc_image,out_image);

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

        //initialization of the iteration structures
        APR_iterator<ImageType> apr_it(*this); //this is required for parallel access
        uint64_t part;

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it)
        for (part = 0; part < this->num_parts_total; ++part) {
            //needed step for any parallel loop (update to the next part)
            apr_it.set_iterator_to_particle_by_number(part);

            apr_it(parts) = img_by_level[apr_it.depth()](apr_it.y(),apr_it.x(),apr_it.z());

        }


//        int z_,x_,j_,y_;
//
//        for(uint64_t depth = (pc_data.depth_min);depth <= pc_data.depth_max;depth++) {
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = pc_data.x_num[depth];
//            const unsigned int z_num_ = pc_data.z_num[depth];
//
//            const unsigned int x_num_min_ = 0;
//            const unsigned int z_num_min_ = 0;
//
//            CurrentLevel<float, uint64_t> curr_level_l(pc_data);
//            curr_level_l.set_new_depth(depth, pc_data);
//
//            const float step_size = pow(2,curr_level_l.depth_max - curr_level_l.depth);
//
//#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level_l) if(z_num_*x_num_ > 100)
//            for (z_ = z_num_min_; z_ < z_num_; z_++) {
//                //both z and x are explicitly accessed in the structure
//
//                for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                    curr_level_l.set_new_xz(x_, z_, pc_data);
//
//                    for (j_ = 0; j_ < curr_level_l.j_num; j_++) {
//
//                        bool iscell = curr_level_l.new_j(j_, pc_data);
//
//                        if (iscell) {
//                            //Indicates this is a particle cell node
//                            curr_level_l.update_cell(pc_data);
//
//                            curr_level_l.get_val(parts) = img_by_level[depth](curr_level_l.y,curr_level_l.x,curr_level_l.z);
//
//                        } else {
//
//                            curr_level_l.update_gap(pc_data);
//
//                        }
//
//
//                    }
//                }
//            }
//        }




    }


    void get_part_numbers() {
        //
        //  Computes totals of total number of particles, and the total number of elements (PC and gap nodes)
        //

        this->num_parts.resize(pc_data.depth_max + 1);
        this->num_elements.resize(pc_data.depth_max + 1);

        int z_, x_, j_, y_;

        uint64_t counter_parts = 0;
        uint64_t counter_elements = 0;

        for (uint64_t depth = (pc_data.depth_min); depth <= pc_data.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[depth];
            const unsigned int z_num_ = pc_data.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            CurrentLevel<ImageType, uint64_t> curr_level_l(pc_data);
            curr_level_l.set_new_depth(depth, pc_data);

            const float step_size = pow(2, curr_level_l.depth_max - curr_level_l.depth);


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

        this->num_parts_total = 0;
        num_elements_total = 0;

        for (int i = 0; i <= pc_data.depth_max; ++i) {

            num_elements_total += num_elements[i];
        }

        this->num_parts_total += num_parts[pc_data.depth_max];

    }

    void set_part_numbers_xz() {
        //
        //  Computes totals of total number of particles in each xz
        //

        num_parts_xy.initialize_structure_parts_empty(particles_int);

        int z_, x_, j_, y_;

        uint64_t counter_parts = 0;

        for (uint64_t depth = (pc_data.depth_min); depth <= pc_data.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[depth];
            const unsigned int z_num_ = pc_data.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            CurrentLevel<ImageType, uint64_t> curr_level_l(pc_data);
            curr_level_l.set_new_depth(depth, pc_data);

            const float step_size = pow(2, curr_level_l.depth_max - curr_level_l.depth);



            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    curr_level_l.set_new_xz(x_, z_, pc_data);



                    for (j_ = 0; j_ < curr_level_l.j_num; j_++) {

                        bool iscell = curr_level_l.new_j(j_, pc_data);


                        if (iscell) {
                            //Indicates this is a particle cell node
                            curr_level_l.update_cell(pc_data);

                            counter_parts++;

                        } else {

                            curr_level_l.update_gap(pc_data);

                        }


                    }

                    num_parts_xy.data[curr_level_l.depth][curr_level_l.pc_offset].push_back(counter_parts);

                }
            }

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


    void init_from_pulling_scheme(std::vector<Mesh_data<uint8_t>>& layers){
        //
        //
        //  INITIALIZE THE PARTICLE CELL STRUCTURE FORM THE OUTPUT OF THE PULLING SCHEME
        //
        //

        //INITIALIZE THE DOMAIN SIZES

        pc_data.x_num.resize(this->depth_max()+1);
        pc_data.y_num.resize(this->depth_max()+1);
        pc_data.z_num.resize(this->depth_max()+1);

        for(int i = pc_data.depth_min;i < pc_data.depth_max;i++){
            pc_data.x_num[i] = layers[i].x_num;
            pc_data.y_num[i] = layers[i].y_num;
            pc_data.z_num[i] = layers[i].z_num;

        }

        pc_data.y_num[pc_data.depth_max] = pc_data.org_dims[0];
        pc_data.x_num[pc_data.depth_max] = pc_data.org_dims[1];
        pc_data.z_num[pc_data.depth_max] = pc_data.org_dims[2];

        //transfer over data-structure to make the same (re-use of function for read-write)

        std::vector<std::vector<uint8_t>> p_map;
        p_map.resize(pc_data.depth_max);

        for (int k = 0; k < pc_data.depth_max; ++k) {
            std::swap(p_map[k],layers[k].mesh);
        }

        create_partcell_structure(p_map);

    }


    void create_partcell_structure(std::vector<std::vector<uint8_t>>& p_map){
        //
        //  Bevan Cheeseman 2017
        //
        //  Takes an optimal part_map configuration from the pushing scheme and creates an efficient data structure for procesing using V, instead of V_n as in original (needs to be optimized)
        //

        //initialize the structure
        pc_data.data.resize(pc_data.depth_max + 1);

        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

            pc_data.data[i].resize(pc_data.z_num[i]*pc_data.x_num[i]);
        }

        Part_timer timer;
        timer.verbose_flag = false;

        //initialize loop variables
        uint64_t x_;
        uint64_t z_;
        uint64_t y_;

        //next initialize the entries;

        uint64_t curr_index;
        uint64_t status;
        uint64_t prev_ind = 0;

        std::vector<unsigned int> x_num = pc_data.x_num;
        std::vector<unsigned int> y_num = pc_data.y_num;
        std::vector<unsigned int> z_num = pc_data.z_num;

        std::vector<uint64_t> status_temp;


        uint64_t prev_coord = 0;


        timer.start_timer("intiialize part_cells");

        const uint8_t seed_us = 4; //deal with the equivalence optimization

        for(uint64_t i = (pc_data.depth_min+1);i < pc_data.depth_max;i++) {

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            const unsigned int x_num_ds = x_num[i - 1];
            const unsigned int z_num_ds = z_num[i - 1];
            const unsigned int y_num_ds = y_num[i - 1];

#pragma omp parallel for default(shared) private(z_, x_, y_, curr_index, status, prev_ind) if(z_num_*x_num_ > 100)
            for (z_ = 0; z_ < z_num_; z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {
                    const size_t offset_part_map_ds = (x_ / 2) * y_num_ds + (z_ / 2) * y_num_ds * x_num_ds;
                    const size_t offset_part_map = x_ * y_num_ + z_ * y_num_ * x_num_;

                    for (y_ = 0; y_ < y_num_ds; y_++) {

                        status = p_map[i - 1][offset_part_map_ds + y_];

                        if (status == SEED) {
                            p_map[i][offset_part_map + 2 * y_] = seed_us;
                            p_map[i][offset_part_map + 2 * y_ + 1] = seed_us;
                        }
                    }
                }

            }
        }



        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            const unsigned int x_num_ds = x_num[i-1];
            const unsigned int z_num_ds = z_num[i-1];
            const unsigned int y_num_ds = y_num[i-1];

#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    size_t first_empty = 0;
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_part_map_ds = (x_/2)*y_num_ds + (z_/2)*y_num_ds*x_num_ds;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 0;

                    //first value handle the duplication of the gap node

                    if(i == (pc_data.depth_max)) {

                        status = p_map[i-1][ offset_part_map_ds];
                        if(status == SEED){
                            first_empty = 0;
                        } else {
                            first_empty = 1;
                        }

                        for (y_ = 0; y_ < y_num_ds; y_++) {

                            status = p_map[i-1][ offset_part_map_ds + y_];

                            if (status == SEED) {
                                curr_index += 1 + prev_ind;
                                prev_ind = 0;
                                curr_index += 1 + prev_ind;
                            } else {
                                prev_ind = 1;
                            }

                        }

                        if (curr_index == 0) {
                            pc_data.data[i][offset_pc_data].resize(
                                    1); //always first adds an extra entry for intialization and extra info
                        } else {

                            pc_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,
                                                                   0); //gap node to begin, already finishes with a gap node

                        }
                    } else {

                        status = p_map[i][offset_part_map];
                        if((status> 1) & (status < 5)){
                            first_empty = 0;
                        } else {
                            first_empty = 1;
                        }

                        for(y_ = 0;y_ < y_num_;y_++){

                            status = p_map[i][offset_part_map + y_];

                            if((status> 1) & (status < 5)){
                                curr_index+= 1 + prev_ind;
                                prev_ind = 0;
                            } else {
                                prev_ind = 1;
                            }
                        }

                        if(curr_index == 0){
                            pc_data.data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                        } else {

                            pc_data.data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node

                        }

                    }


                }
            }

        }

        prev_coord = 0;


        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            const unsigned int x_num_ds = x_num[i-1];
            const unsigned int z_num_ds = z_num[i-1];
            const unsigned int y_num_ds = y_num[i-1];

#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind,prev_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_part_map_ds = (x_/2)*y_num_ds + (z_/2)*y_num_ds*x_num_ds;
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 1;
                    prev_coord = 0;

                    if(i == pc_data.depth_max){
                        //initialize the first values type
                        pc_data.data[i][offset_pc_data][0] = TYPE_GAP_END;

                        uint64_t y_u;

                        for (y_ = 0; y_ < y_num_ds; y_++) {

                            status = p_map[i - 1][offset_part_map_ds + y_];

                            if (status == SEED) {

                                for (int k = 0; k < 2; ++k) {

                                    y_u = 2*y_ + k;

                                    curr_index++;

                                    //set starting type
                                    if (prev_ind == 1) {
                                        //gap node
                                        //set type
                                        pc_data.data[i][offset_pc_data][curr_index - 1] = TYPE_GAP;
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= (y_u << NEXT_COORD_SHIFT);
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= (prev_coord
                                                << PREV_COORD_SHIFT);
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR
                                                << YP_DEPTH_SHIFT);
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR
                                                << YM_DEPTH_SHIFT);

                                        curr_index++;
                                    }
                                    prev_coord = y_u;
                                    //set type
                                    pc_data.data[i][offset_pc_data][curr_index - 1] = TYPE_PC;

                                    //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);

                                    //set the status
                                    switch (status) {
                                        case SEED: {
                                            pc_data.data[i][offset_pc_data][curr_index - 1] |= SEED_SHIFTED;
                                            break;
                                        }
                                        case BOUNDARY: {
                                            pc_data.data[i][offset_pc_data][curr_index - 1] |= BOUNDARY_SHIFTED;
                                            break;
                                        }
                                        case FILLER: {
                                            pc_data.data[i][offset_pc_data][curr_index - 1] |= FILLER_SHIFTED;
                                            break;
                                        }

                                    }

                                    prev_ind = 0;
                                }
                            } else {
                                //store for setting above
                                if (prev_ind == 0) {
                                    //prev_coord = y_;
                                }

                                prev_ind = 1;

                            }

                        }

                        //Initialize the last value GAP END indicators to no neighbour
                        pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size() - 1] = TYPE_GAP_END;
                        pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YP_DEPTH_SHIFT);
                        pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YM_DEPTH_SHIFT);



                    } else {

                        //initialize the first values type
                        pc_data.data[i][offset_pc_data][0] = TYPE_GAP_END;

                        for (y_ = 0; y_ < y_num_; y_++) {

                            status = p_map[i][offset_part_map + y_];

                            if((status> 1) && (status < 5)) {

                                curr_index++;

                                //set starting type
                                if (prev_ind == 1) {
                                    //gap node
                                    //set type
                                    pc_data.data[i][offset_pc_data][curr_index - 1] = TYPE_GAP;
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (y_ << NEXT_COORD_SHIFT);
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (prev_coord << PREV_COORD_SHIFT);
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                                    pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);

                                    curr_index++;
                                }
                                prev_coord = y_;
                                //set type
                                pc_data.data[i][offset_pc_data][curr_index - 1] = TYPE_PC;

                                //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                                pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                                pc_data.data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);

                                //set the status
                                switch (status) {
                                    case seed_us: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= SEED_SHIFTED;
                                        break;
                                    }
                                    case BOUNDARY: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= BOUNDARY_SHIFTED;
                                        break;
                                    }
                                    case FILLER: {
                                        pc_data.data[i][offset_pc_data][curr_index - 1] |= FILLER_SHIFTED;
                                        break;
                                    }

                                }

                                prev_ind = 0;
                            } else {


                                prev_ind = 1;

                            }
                        }

                        //Initialize the last value GAP END indicators to no neighbour
                        pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size() - 1] = TYPE_GAP_END;
                        pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YP_DEPTH_SHIFT);
                        pc_data.data[i][offset_pc_data][pc_data.data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YM_DEPTH_SHIFT);
                    }
                }
            }

        }

        timer.stop_timer();


        ///////////////////////////////////
        //
        //  Calculate neighbours
        //
        /////////////////////////////////

        timer.start_timer("set_up_neigh");

        //(+y,-y,+x,-x,+z,-z)
        pc_data.set_neighbor_relationships();

        timer.stop_timer();

    }

    ///////////////////////
    ///
    /// Random Access Structures (Experimental) Cheeseman 2018
    ///
    ///
    ///////////////////////

    int random_access_pc(uint64_t depth,uint16_t y,uint64_t x,uint64_t z){
        //
        //  Random access check for valid x,z, any given y, returns the index of the stored Particle Intensity.
        //

        int j;

        uint64_t pc_offset = pc_data.x_num[depth]*z + x;

        if(random_access.data[depth][pc_offset].size() > 0) {
            hash_map::iterator pc = random_access.data[depth][pc_offset][0].find(y);

            if(pc != random_access.data[depth][pc_offset][0].end()){
                j = pc->second;
            } else {
                return -1;
            }

        } else {
            return -1;

        }

        return j;

    }

    //////////////////////////
    ///
    /// Experimental random access neighbours.
    ///
    /// \tparam S data type of the particles
    /// \param face the neighbour direction (+y,-y,+x,-x,+z,-z)
    /// \param parts the particles data structure
    /// \param neigh_val vector returning the particles values of the neighbours
    ////////////////////////

    template<typename S>
    void get_neigh_random(unsigned int face,ExtraPartCellData<S>& parts,std::vector<S>& neigh_val){
        //
        //  Get APR face neighbours relying on random access through a map, or unordered map structure for y
        //

        const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
        const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
        const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

        constexpr uint8_t neigh_child_dir[6][3] = {{4,2,2},{4,2,2},{0,4,4},{0,4,4},{0,2,2},{0,2,2}};

        constexpr uint8_t child_offsets[3][3] = {{0,1,1},{1,0,1},{1,1,0}};

        //first try on same depth
        int z_ = this->z() + dir_z[face];
        int x_ = this->x() + dir_x[face];
        int y_ = this->y() + dir_y[face];
        int depth_ = this->depth();

        uint16_t j=0;

        uint64_t pc_offset = pc_data.x_num[depth_]*z_ + x_;
        bool found = false;

        neigh_val.resize(0);

        if((x_ < 0) | (x_ >= pc_data.x_num[depth_]) | (z_ < 0) | (z_ >= pc_data.z_num[depth_]) ){
            //out of bounds
            return;
        }

        if(random_access.data[depth_][pc_offset].size() > 0) {
            hash_map::iterator pc = random_access.data[depth_][pc_offset][0].find(y_);

            if(pc != random_access.data[depth_][pc_offset][0].end()){
                j = pc->second;
                found = true;
            }
        }

        if(!found){
            //
            //  Find parents
            //

            unsigned int depth_p = depth_ - 1;
            unsigned int x_p = x_/2;
            unsigned int y_p = y_/2;
            unsigned int z_p = z_/2;

            pc_offset = pc_data.x_num[depth_p]*z_p + x_p;

            if(random_access.data[depth_p][pc_offset].size() > 0) {
                hash_map::iterator pc = random_access.data[depth_p][pc_offset][0].find(y_p);

                if(pc != random_access.data[depth_p][pc_offset][0].end()){
                    j = pc->second;
                    found = true;
                }
            }

            if(!found) {

                if(depth_ < pc_data.depth_max) {
                    // get the potentially 4 children
                    unsigned int depth_c = depth_ + 1;
                    unsigned int x_c = (x_ + dir_x[face])*2 + (dir_x[face]<0);
                    unsigned int y_c = (y_ + dir_y[face])*2 + (dir_y[face]<0);
                    unsigned int z_c = (z_ + dir_z[face])*2 + (dir_z[face]<0);

                    unsigned int dir = face/2;

                    for (int i = 0; i < 2; ++i) {
                        for (int k = 0; k < 2; ++k) {
                            y_ = y_c + (child_offsets[dir][0])*i + (child_offsets[dir][0])*k;
                            x_ = x_c + (child_offsets[dir][1])*i + (child_offsets[dir][1])*k;
                            z_ = z_c + (child_offsets[dir][2])*i + (child_offsets[dir][2])*k;

                            //add of they exist
                            if((x_ < 0) | (x_ >= pc_data.x_num[depth_c]) | (z_ < 0) | (z_ >= pc_data.z_num[depth_]) ){
                                //out of bounds

                            } else {

                                pc_offset = pc_data.x_num[depth_c]*z_ + x_;

                                if (random_access.data[depth_c][pc_offset].size() > 0) {
                                    hash_map::iterator pc = random_access.data[depth_c][pc_offset][0].find(y_);

                                    if (pc != random_access.data[depth_c][pc_offset][0].end()) {
                                        j = pc->second;
                                        neigh_val.push_back(parts.data[depth_c][pc_offset][j]);
                                    }
                                }

                            }



                        }
                    }



                }

            } else{
                neigh_val.push_back(parts.data[depth_p][pc_offset][j]);
            }

        } else{

            neigh_val.push_back(parts.data[depth_][pc_offset][j]);

        }


    }


    void read_apr(std::string file_name)
    {

        //hdf5 inits
        hid_t fid, pr_groupid, obj_id,attr_id;
        H5G_info_t info;

        //need to register the filters so they work properly
        register_bosc();

        int num_parts,num_cells;

        fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);

        //Get the group you want to open

        pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
        H5Gget_info( pr_groupid, &info );

        //Getting an attribute
        obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);

        //Load the attributes


        /////////////////////////////////////////////
        //  Get metadata
        //
        //////////////////////////////////////////////

        hid_t       aid, atype, attr;

        aid = H5Screate(H5S_SCALAR);

        name.reserve(100);

        //std::string string_out;

        //std::vector<char> string_out;
        //string_out.resize(80);

        //atype = H5Tcopy (H5T_C_S1);

        char string_out[100];

        for (int j = 0; j < 100; ++j) {
            string_out[j] = 0;
        }

        attr_id = 	H5Aopen(pr_groupid,"name",H5P_DEFAULT);

        atype = H5Aget_type(attr_id);

        hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);

        H5Aread(attr_id,atype_mem,string_out) ;
        H5Aclose(attr_id);

        name= string_out;
        pars.name = string_out;

        pc_data.org_dims.resize(3);
        pc_data.depth_max = 0;
        pc_data.depth_min = 0;

        attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&pc_data.org_dims[0]) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&pc_data.org_dims[1]) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&pc_data.org_dims[2]) ;
        H5Aclose(attr_id);

//        attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
//        H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
//        H5Aclose(attr_id);
//
//        attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
//        H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
//        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"depth_max",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&pc_data.depth_max) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"depth_min",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&pc_data.depth_min) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"lambda",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.lambda ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"sigma_th",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.sigma_th ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"sigma_th_max",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.sigma_th_max ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"I_th",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.Ip_th ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"dx",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.dx ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"dy",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.dy ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"dz",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.dz ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"psfx",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.psfx ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"psfy",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.psfy ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"psfz",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.psfz ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"rel_error",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.rel_error ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"background_intensity_estimate",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.background_intensity_estimate ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"noise_sd_estimate",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&parameters.noise_sd_estimate ) ;
        H5Aclose(attr_id);

        //std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;

        std::vector<std::vector<uint8_t>> p_map_load;
        std::vector<std::vector<uint16_t>> Ip;

        pc_data.depth_max = pc_data.depth_max;

        p_map_load.resize(pc_data.depth_max);
        Ip.resize(pc_data.depth_max+1);

        std::string name;

        pc_data.x_num.resize(pc_data.depth_max+1);
        pc_data.z_num.resize(pc_data.depth_max+1);
        pc_data.y_num.resize(pc_data.depth_max+1);

        for(int i = pc_data.depth_min;i < pc_data.depth_max; i++){

            //get the info

            int x_num;
            name = "p_map_x_num_"+std::to_string(i);

            attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
            H5Aread(attr_id,H5T_NATIVE_INT,&x_num) ;
            H5Aclose(attr_id);

            int y_num;
            name = "p_map_y_num_"+std::to_string(i);

            attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
            H5Aread(attr_id,H5T_NATIVE_INT,&y_num) ;
            H5Aclose(attr_id);

            int z_num;
            name = "p_map_z_num_"+std::to_string(i);

            attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
            H5Aread(attr_id,H5T_NATIVE_INT,&z_num) ;
            H5Aclose(attr_id);


            p_map_load[i].resize(x_num*y_num*z_num);


            if(p_map_load[i].size() > 0){
                name = "p_map_"+std::to_string(i);
                //Load the data then update the particle dataset
                hdf5_load_data_blosc(obj_id,H5T_NATIVE_UINT8,p_map_load[i].data(),name.c_str());
            }


            pc_data.x_num[i] = x_num;
            pc_data.y_num[i] = y_num;
            pc_data.z_num[i] = z_num;

        }

        for(int i = pc_data.depth_min;i <= pc_data.depth_max; i++){

            //get the info

            int Ip_num;
            name = "Ip_size_"+std::to_string(i);

            attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
            H5Aread(attr_id,H5T_NATIVE_INT,&Ip_num);
            H5Aclose(attr_id);


            Ip[i].resize(Ip_num);

            if(Ip[i].size()>0){
                name = "Ip_"+std::to_string(i);
                hdf5_load_data_blosc(obj_id,H5T_NATIVE_UINT16,Ip[i].data(),name.c_str());
            }

        }

        pc_data.y_num[pc_data.depth_max] = pc_data.org_dims[0];
        pc_data.x_num[pc_data.depth_max] = pc_data.org_dims[1];
        pc_data.z_num[pc_data.depth_max] = pc_data.org_dims[2];


        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        //
        //
        //  Transfer back the intensities
        //
        //
        create_partcell_structure(p_map_load);

        particles_int.initialize_structure_cells(pc_data);

        for (int depth = this->depth_min(); depth <= this->depth_max(); ++depth) {

            uint64_t counter = 0;

            for (this->begin(depth); this->end(depth) != 0 ; this->it_forward(depth)) {

                this->curr_level.get_val(particles_int) = Ip[depth][counter];

                counter++;

            }
        }


    }

    void write_apr(std::string save_loc,std::string file_name,bool predict = false){
        //
        //
        //  Bevan Cheeseman 2018
        //
        //  Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
        //
        //

        APR_timer write_timer;

        write_timer.verbose_flag = false;

        //initialize
        uint64_t node_val_part;
        uint64_t y_coord;
        int x_;
        int z_;

        uint64_t j_;
        uint64_t status;
        uint64_t curr_key=0;
        uint64_t part_offset=0;


        //Neighbour Routine Checking

        uint64_t p;

        register_bosc();

        std::string hdf5_file_name = save_loc + file_name + "_apr.h5";

        file_name = file_name + "_apr";

        hdf5_create_file_blosc(hdf5_file_name);

        //hdf5 inits
        hid_t fid, pr_groupid, obj_id;
        H5G_info_t info;

        hsize_t     dims_out[2];

        hsize_t rank = 1;

        hsize_t dims;
        hsize_t dim_a=1;

        fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

        //Get the group you want to open

        //////////////////////////////////////////////////////////////////
        //
        //  Write meta-data to the file
        //
        //
        //
        ///////////////////////////////////////////////////////////////////////
        dims = 1;

        pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        H5Gget_info( pr_groupid, &info );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &pc_data.org_dims[1] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &pc_data.org_dims[0] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &pc_data.org_dims[2] );


        obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

        dims_out[0] = 1;
        dims_out[1] = 1;

        //just an identifier in here for the reading of the parts

        int num_parts = this->num_parts_total;
        int num_cells = this->num_elements_total;

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

        // New parameter and background data

        if(pars.name.size() == 0){
            pars.name = "no_name";
            name = "no_name";
        }

        hdf5_write_string_blosc(pr_groupid,"name",pars.name);

        std::string git_hash = exec_blosc("git rev-parse HEAD");

        hdf5_write_string_blosc(pr_groupid,"githash",git_hash);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"lambda",1,dims_out, &parameters.lambda );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"sigma_th",1,dims_out, &parameters.sigma_th );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"sigma_th_max",1,dims_out, &parameters.sigma_th_max );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"I_th",1,dims_out, &parameters.Ip_th );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"dx",1,dims_out, &parameters.dx );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"dy",1,dims_out, &parameters.dy );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"dz",1,dims_out, &parameters.dz );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"psfx",1,dims_out, &parameters.psfx );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"psfy",1,dims_out, &parameters.psfy );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"psfz",1,dims_out, &parameters.psfz );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"rel_error",1,dims_out, &parameters.rel_error);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"noise_sd_estimate",1,dims_out, &parameters.noise_sd_estimate);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"background_intensity_estimate",1,dims_out, &parameters.background_intensity_estimate);

        //////////////////////////////////////////////////////////////////
        //
        //  Write data to the file
        //
        //
        //
        ///////////////////////////////////////////////////////////////////////

        write_timer.start_timer("intensities");

        uint64_t depth_min = pc_data.depth_min;

        std::vector<uint16_t> Ip;

        ExtraPartCellData<ImageType> temp_int;
        temp_int.initialize_structure_cells(pc_data);
        temp_int.data = particles_int.data;

        shift_particles_from_cells(temp_int);

        write_timer.stop_timer();

        write_timer.start_timer("write int");


        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            const unsigned int y_num_ = pc_data.y_num[i];

            //write the vals
            Ip.resize(0);

            uint64_t total_p = 0;

            for(z_ = 0;z_ < z_num_;z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {

                    const size_t offset_pc_data = x_num_ * z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    total_p += j_num;
                }
            }

            Ip.resize(total_p);

            total_p = 0;

            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    std::copy(temp_int.data[i][offset_pc_data].begin(),temp_int.data[i][offset_pc_data].end(),Ip.begin() + total_p);

                    total_p += j_num;

                }

            }

            dims = Ip.size();

            std::string name = "Ip_size_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);

            if(Ip.size() > 0){
                //write the parts

                name = "Ip_"+std::to_string(i);

                hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, Ip.data());

            }

        }

        write_timer.stop_timer();


        write_timer.start_timer("pc_data");


        for(uint64_t i = pc_data.depth_min;i < pc_data.depth_max;i++){


            unsigned int x_num_ = pc_data.x_num[i];
            unsigned int z_num_ = pc_data.z_num[i];
            unsigned int y_num_ = pc_data.y_num[i];

            uint64_t node_val;

            std::vector<uint8_t> p_map;

            p_map.resize(x_num_*z_num_*y_num_,0);

            std::fill(p_map.begin(), p_map.end(), 0);

            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;

                    const size_t j_num = pc_data.data[i][offset_pc_data].size();

                    y_coord = 0;

                    for(j_ = 0;j_ < j_num;j_++){

                        node_val = pc_data.data[i][offset_pc_data][j_];

                        if (!(node_val&1)){
                            //get the index gap node
                            y_coord++;

                            status = pc_data.get_status(node_val);

                            if(status > 1) {

                                p_map[offset_p_map + y_coord] = status;

                            }

                        } else {

                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure

                        }

                    }

                }

            }


            x_num_ = pc_data.x_num[i+1];
            z_num_ = pc_data.z_num[i+1];
            y_num_ = pc_data.y_num[i+1];

            int x_num_d = pc_data.x_num[i];
            int z_num_d = pc_data.z_num[i];
            int y_num_d = pc_data.y_num[i];

#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_p_map = y_num_d*x_num_d*(z_/2) + y_num_d*(x_/2);

                    const size_t j_num = pc_data.data[i+1][offset_pc_data].size();

                    y_coord = 0;

                    for(j_ = 0;j_ < j_num;j_++){

                        node_val = pc_data.data[i+1][offset_pc_data][j_];

                        if (!(node_val&1)){
                            //get the index gap node
                            y_coord++;

                            status = pc_data.get_status(node_val);

                            if(status == 1) {

                                p_map[offset_p_map + y_coord/2] = status;

                            }

                        } else {

                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                        }

                    }

                }

            }


            dims = p_map.size();
            name = "p_map_"+std::to_string(i);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, p_map.data());

            name = "p_map_x_num_"+std::to_string(i);
            hsize_t attr = x_num_d;
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

            attr = y_num_d;
            name = "p_map_y_num_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

            attr = z_num_d;
            name = "p_map_z_num_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

        }

        write_timer.stop_timer();

        hsize_t attr = pc_data.depth_min;
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &pc_data.depth_max );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );

        // output the file size
        hsize_t file_size;
        H5Fget_filesize(fid, &file_size);

        std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        std::cout << "Writing Complete" << std::endl;


    }



    template<typename S>
    void write_particles_only(std::string save_loc,std::string file_name,ExtraPartCellData<S>& parts_extra){
        //
        //
        //  Bevan Cheeseman 2018
        //
        //  Writes only the particle data, requires the same APR to be read in correctly.
        //
        //  #FIX_ME Extend me.
        //
        //

        std::cout << "Either uint8, uint16, or float" << std::endl;

        //initialize
        uint64_t node_val_part;
        uint64_t y_coord;
        int x_;
        int z_;

        uint64_t j_;
        uint64_t status;
        uint64_t curr_key=0;
        uint64_t part_offset=0;


        //Neighbour Routine Checking

        uint64_t p;

        register_bosc();

        std::string hdf5_file_name = save_loc + file_name + "_apr_extra_parts.h5";

        file_name = file_name + "_apr_extra_parts";

        hdf5_create_file_blosc(hdf5_file_name);

        //hdf5 inits
        hid_t fid, pr_groupid, obj_id;
        H5G_info_t info;

        hsize_t     dims_out[2];

        hsize_t rank = 1;

        hsize_t dims;
        hsize_t dim_a=1;

        fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

        //Get the group you want to open

        //////////////////////////////////////////////////////////////////
        //
        //  Write meta-data to the file
        //
        //
        //
        ///////////////////////////////////////////////////////////////////////
        dims = 1;

        pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        H5Gget_info( pr_groupid, &info );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &pc_data.org_dims[1] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &pc_data.org_dims[0] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &pc_data.org_dims[2] );


        obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

        dims_out[0] = 1;
        dims_out[1] = 1;

        //just an identifier in here for the reading of the parts

        int num_parts = this->num_parts_total;
        int num_cells = this->num_elements_total;

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

        // New parameter and background data

        if(pars.name.size() == 0){
            pars.name = "no_name";
            name = "no_name";
        }

        hdf5_write_string_blosc(pr_groupid,"name",pars.name);

        std::string git_hash = exec_blosc("git rev-parse HEAD");

        hdf5_write_string_blosc(pr_groupid,"githash",git_hash);


        S val = 0;
        hid_t type = get_type_blosc(val);
        int type_id = type;

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"data_type",1,dims_out, &type_id);

        //////////////////////////////////////////////////////////////////
        //
        //  Write data to the file
        //
        //
        //
        ///////////////////////////////////////////////////////////////////////

        uint64_t depth_min = pc_data.depth_min;

        std::vector<S> Ip;

        ExtraPartCellData<S> temp_int;
        temp_int.initialize_structure_cells(pc_data);
        temp_int.data = parts_extra.data;

        shift_particles_from_cells(temp_int);


        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

            const unsigned int x_num_ = pc_data.x_num[i];
            const unsigned int z_num_ = pc_data.z_num[i];
            const unsigned int y_num_ = pc_data.y_num[i];

            //write the vals

            Ip.resize(0);

            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    uint64_t curr_size = Ip.size();
                    Ip.resize(curr_size+ j_num);

                    std::copy(temp_int.data[i][offset_pc_data].begin(),temp_int.data[i][offset_pc_data].end(),Ip.begin() + curr_size);

                }

            }

            dims = Ip.size();

            std::string name = "Ip_size_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);

            if(Ip.size() > 0){
                //write the parts

                name = "Ip_"+std::to_string(i);

                S val = 0;

                hdf5_write_data_blosc(obj_id, get_type_blosc(val), name.c_str(), rank, &dims, Ip.data());

            }

        }

        hsize_t attr = pc_data.depth_min;
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &pc_data.depth_max );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );

        // output the file size
        hsize_t file_size;
        H5Fget_filesize(fid, &file_size);

        std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        std::cout << "Writing ExtraPartCellData Complete" << std::endl;

    }

    template<typename T>
    void read_parts_only(std::string file_name,ExtraPartCellData<T>& extra_parts)
    {

        //hdf5 inits
        hid_t fid, pr_groupid, obj_id,attr_id;
        H5G_info_t info;

        //need to register the filters so they work properly
        register_bosc();

        int num_parts,num_cells;

        fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);

        //Get the group you want to open

        pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
        H5Gget_info( pr_groupid, &info );

        //Getting an attribute
        obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);

        //Load the attributes


        /////////////////////////////////////////////
        //  Get metadata
        //
        //////////////////////////////////////////////

        hid_t       aid, atype, attr;

        aid = H5Screate(H5S_SCALAR);

        int data_type;

        attr_id = 	H5Aopen(pr_groupid,"data_type",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&data_type ) ;
        H5Aclose(attr_id);

        hid_t hdf5_data_type = data_type;

        std::vector<std::vector<T>> Ip;

        Ip.resize(pc_data.depth_max+1);

        for(int i = pc_data.depth_min;i <= pc_data.depth_max; i++){

            //get the info

            int Ip_num;
            name = "Ip_size_"+std::to_string(i);

            attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
            H5Aread(attr_id,H5T_NATIVE_INT,&Ip_num);
            H5Aclose(attr_id);

            Ip[i].resize(Ip_num);

            if(Ip[i].size()>0){
                name = "Ip_"+std::to_string(i);
                hdf5_load_data_blosc(obj_id,hdf5_data_type,Ip[i].data(),name.c_str());
            }

        }

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        extra_parts.initialize_structure_cells(pc_data);

        for (int depth = this->depth_min(); depth <= this->depth_max(); ++depth) {

            uint64_t counter = 0;

            for (this->begin(depth); this->end() == true ; this->it_forward(depth)) {

                this->curr_level.get_val(extra_parts) = Ip[depth][counter];

                counter++;

            }
        }


    }




    template<typename S>
    void predict_intensities(ExtraPartCellData<S>& parts){
        //
        //  Reduce the entropy of the saved sequence by predicting the intensity and storing \delta
        //

        //init temp data
        ExtraPartCellData<S> delta;
        delta.initialize_structure_cells(pc_data);

        std::vector<S> neigh_vec;

        std::vector<unsigned int> dir = {1,3};

        //loops from lowest level to highest
        for (this->begin(); this->end() != 0 ; this->it_forward()) {

            //get the minus neighbours (1,3,5)

            float pred = 0;
            float counter = 0;

            for (int j = 0; j < dir.size(); ++j) {

                get_neigh_dir(parts, neigh_vec, dir[j]);

                if (neigh_vec.size() > 0) {
                    //check if the depth is less
                    if (pc_data.pc_key_get_depth(this->curr_level.neigh_part_keys.neigh_face[dir[j]][0]) <= this->depth()) {
                        for (int i = 0; i < neigh_vec.size(); ++i) {
                            pred += neigh_vec[i];
                            counter++;
                        }
                    }
                }
            }

            if(counter > 0){
                pred = floor(pred/counter);
            }

            this->curr_level.get_val(delta) = this->curr_level.get_val(parts) - pred;


        }

        std::swap(parts,delta);

    }





//
//    cudaCompress::util::u2f((uint16_t*)dpImage, dpBuffer, sizeX * sizeY);
//    // variance stabilization
//    cudaCompress::util::vst(dpBuffer, dpBuffer, sizeX * sizeY, bgLevel, conversion, readNoise);
//    // scale with quantization step
//    cudaCompress::util::multiply(dpBuffer, dpBuffer, 1 / quantStep, sizeX * sizeY);
//    // run  quantization first then prediction
//    cudaCompress::util::f2u(dpBuffer, (uint16_t*)dpScratch, sizeX * sizeY);
//    //cudaMemcpy(dpImage, dpScratch, sizeX*sizeY * sizeof(int16_t), cudaMemcpyDeviceToDevice);
//    cudaCompress::util::predictor7_tiles((int16_t*)dpScratch, dpImage, sizeX * sizeof(int16_t), sizeX, sizeY, tileSize);
//    cudaCompress::util::symbolize(dpSymbols, dpImage, sizeX, sizeY, sizeZ);

};




#endif //PARTPLAY_APR_HPP
