//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APRRECONSTRUCTION_HPP
#define PARTPLAY_APRRECONSTRUCTION_HPP

#include "src/data_structures/APR/APR.hpp"

class APRReconstruction {
public:


    template<typename U,typename V,typename S>
    void interp_img(APR<S>& apr, Mesh_data<U>& img,ExtraPartCellData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        img.initialize( apr.orginal_dimensions(0), apr.orginal_dimensions(1), apr.orginal_dimensions(2),0);

        int z_,x_,j_,y_;

        for(uint64_t depth = (apr.depth_min());depth <= apr.depth_max();depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ =  apr.spatial_index_x_max(depth);
            const unsigned int z_num_ = apr.spatial_index_z_max(depth);

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            CurrentLevel<float, uint64_t> curr_level_l(apr.pc_data);
            curr_level_l.set_new_depth(depth, apr.pc_data);

            const float step_size = pow(2,curr_level_l.depth_max - curr_level_l.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level_l) if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    curr_level_l.set_new_xz(x_, z_, apr.pc_data);

                    for (j_ = 0; j_ < curr_level_l.j_num; j_++) {

                        bool iscell = curr_level_l.new_j(j_, apr.pc_data);

                        if (iscell) {
                            //Indicates this is a particle cell node
                            curr_level_l.update_cell(apr.pc_data);

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

                            curr_level_l.update_gap(apr.pc_data);

                        }


                    }
                }
            }
        }

    }



    template<typename U,typename S>
    void interp_depth_ds(APR<S>& apr,Mesh_data<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        //get depth
        ExtraPartCellData<U> depth_parts(apr);

        for (apr.begin(); apr.end() != 0 ; apr.it_forward()) {
            //
            //  Demo APR iterator
            //

            //access and info
            apr.curr_level.get_val(depth_parts) =apr.depth();

        }

        Mesh_data<U> temp;

        interp_img(apr,temp,depth_parts);

        down_sample(temp,img,
                    [](U x, U y) { return std::max(x,y); },
                    [](U x) { return x; }, true);

    }

    template<typename U,typename S>
    void interp_depth(APR<S>& apr,Mesh_data<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        //get depth
        ExtraPartCellData<U> depth_parts(apr);


        for (apr.begin(); apr.end() == true ; apr.it_forward()) {
            //
            //  Demo APR iterator
            //

            //access and info
            apr.curr_level.get_val(depth_parts) = apr.depth();

        }

        interp_img(apr,img,depth_parts);

    }

    template<typename U,typename S>
    void interp_type(APR<S>& apr,Mesh_data<U>& img){

        //get depth
        ExtraPartCellData<U> type_parts(apr);


        for (apr.begin(); apr.end() == true ; apr.it_forward()) {
            //
            //  Demo APR iterator
            //

            //access and info
            apr.curr_level.get_val(type_parts) = apr.type();

        }

        interp_img(apr,img,type_parts);


    }

    template<typename T>
    void calc_sat_adaptive_y(Mesh_data<T>& input,Mesh_data<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in,const unsigned int d_max){
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

        //const unsigned int d_max = this->depth_max();

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
    void calc_sat_adaptive_x(Mesh_data<T>& input,Mesh_data<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in,const unsigned int d_max){
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
        //const unsigned int d_max = this->depth_max();


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
    void calc_sat_adaptive_z(Mesh_data<T>& input,Mesh_data<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in,const unsigned int d_max ){

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
        //const unsigned int d_max = this->depth_max();

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




    template<typename U,typename V,typename S>
    void interp_parts_smooth(APR<S>& apr,Mesh_data<U>& out_image,ExtraPartCellData<V>& interp_data,std::vector<float> scale_d = {2,2,2}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        Part_timer timer;
        timer.verbose_flag = false;

        Mesh_data<U> pc_image;
        Mesh_data<uint8_t> k_img;

        unsigned int offset_max = 20;

        interp_img(apr,pc_image,interp_data);

        interp_depth(apr,k_img);

        timer.start_timer("sat");
        //demo
        calc_sat_adaptive_y(pc_image,k_img,scale_d[0],offset_max,apr.depth_max());

        timer.stop_timer();

        timer.start_timer("sat");

        calc_sat_adaptive_x(pc_image,k_img,scale_d[1],offset_max,apr.depth_max());

        timer.stop_timer();

        timer.start_timer("sat");

        calc_sat_adaptive_z(pc_image,k_img,scale_d[2],offset_max,apr.depth_max());

        timer.stop_timer();

        std::swap(pc_image,out_image);

    }



};


#endif //PARTPLAY_APRRECONSTRUCTION_HPP
