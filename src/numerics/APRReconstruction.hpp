//
// Created by cheesema on 16.01.18.
//

#ifndef PARTPLAY_APRRECONSTRUCTION_HPP
#define PARTPLAY_APRRECONSTRUCTION_HPP

#include "../data_structures/APR/APR.hpp"
#include "../data_structures/APR/APRIterator.hpp"


struct recon_patch{
    int x_begin=0;
    int x_end=-1;
    int y_begin=0;
    int y_end=-1;
    int z_begin=0;
    int z_end=-1;
    int level_delta=0;
};

class APRReconstruction {
public:


    template<typename U,typename V,typename S>
    void interp_img(APR<S>& apr, MeshData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        APRIterator<S> apr_iterator(apr);
        uint64_t particle_number;

        img.init(apr.orginal_dimensions(0), apr.orginal_dimensions(1), apr.orginal_dimensions(2), 0);

        for (uint64_t level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

            const float step_size = pow(2,apr_iterator.level_max() - level);

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
            for (particle_number = apr_iterator.particles_level_begin(level); particle_number <  apr_iterator.particles_level_end(level); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                int dim1 = apr_iterator.y() * step_size;
                int dim2 = apr_iterator.x() * step_size;
                int dim3 = apr_iterator.z() * step_size;

                float temp_int;
                //add to all the required rays

                temp_int = parts[apr_iterator];

                const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                    for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                        for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                            img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] = temp_int;
                        }
                    }
                }
            }
        }

    }

    template<typename U,typename V,typename S>
    void interp_image_patch(APR<S>& apr, MeshData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        APRIterator<S> apr_iterator(apr);
        uint64_t particle_number;

        img.init(apr.orginal_dimensions(0), apr.orginal_dimensions(1), apr.orginal_dimensions(2), 0);

        for (uint64_t level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

            const float step_size = pow(2,apr_iterator.level_max() - level);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
            for (particle_number = apr_iterator.particles_level_begin(level); particle_number <  apr_iterator.particles_level_end(level); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                int dim1 = apr_iterator.y() * step_size;
                int dim2 = apr_iterator.x() * step_size;
                int dim3 = apr_iterator.z() * step_size;

                float temp_int;
                //add to all the required rays

                temp_int = parts[apr_iterator];

                const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                    for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                        for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                            img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] = temp_int;
                        }
                    }
                }
            }
        }

    }





    template<typename U,typename S>
    void interp_depth_ds(APR<S>& apr,MeshData<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        //get depth
        ExtraParticleData<U> depth_parts(apr);

        APRIterator<S> apr_iterator(apr);
        uint64_t particle_number;

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            //access and info
            depth_parts[apr_iterator] =apr_iterator.level();

        }

        MeshData<U> temp;

        interp_img(apr,temp,depth_parts);

        downsample(temp, img,
                   [](const U &x, const U &y) -> U { return std::max(x, y); },
                   [](const U &x) -> U { return x; }, true);

    }

    template<typename U,typename S>
    void interp_level(APR<S> &apr, MeshData<U> &img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        //get depth
        ExtraParticleData<U> level_parts(apr);

        APRIterator<S> apr_iterator(apr);
        uint64_t particle_number;

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            //
            //  Demo APR iterator
            //

            //access and info
            level_parts[apr_iterator] = apr_iterator.level();

        }

        interp_img(apr,img,level_parts);

    }

    template<typename U,typename S>
    void interp_type(APR<S>& apr,MeshData<U>& img){

        //get depth
        ExtraParticleData<U> type_parts(apr);


        APRIterator<S> apr_iterator(apr);
        uint64_t particle_number;

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            //
            //  Demo APR iterator
            //

            //access and info
            apr_iterator(type_parts) = apr_iterator.type();

        }

        interp_img(apr,img,type_parts);


    }

    template<typename T>
    void calc_sat_adaptive_y(MeshData<T>& input,MeshData<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in,const unsigned int d_max){
        //
        //  Bevan Cheeseman 2016
        //
        //  Calculates a O(1) recursive mean using SAT.
        //


        const int64_t z_num = input.z_num;
        const int64_t x_num = input.x_num;
        const int64_t y_num = input.y_num;

        std::vector<T> temp_vec;
        temp_vec.resize(y_num,0);

        std::vector<T> offset_vec;
        offset_vec.resize(y_num,0);


        int64_t i, k, index;
        float counter, temp, divisor,offset;

        //need to introduce an offset max to make the algorithm still work, and it also makes sense.
        const int offset_max = offset_max_in;

        float scale = scale_in;

        //const unsigned int d_max = this->depth_max();

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,k,counter,temp,index,divisor,offset) firstprivate(temp_vec,offset_vec)
#endif
        for(int64_t j = 0;j < z_num;j++){
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
    void calc_sat_adaptive_x(MeshData<T>& input,MeshData<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in,const unsigned int d_max){
        //
        //  Adaptive form of Matteusz' SAT code.
        //
        //

        const int64_t z_num = input.z_num;
        const int64_t x_num = input.x_num;
        const int64_t y_num = input.y_num;

        unsigned int offset_max = offset_max_in;

        std::vector<T> temp_vec;
        temp_vec.resize(y_num*(2*offset_max + 2),0);

        int64_t i,k;
        float temp;
        int64_t index_modulo, previous_modulo, jxnumynum, offset,forward_modulo,backward_modulo;

        const float scale = scale_in;
        //const unsigned int d_max = this->depth_max();


#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,k,temp,index_modulo, previous_modulo, forward_modulo,backward_modulo, jxnumynum,offset) \
        firstprivate(temp_vec)
#endif
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
    void calc_sat_adaptive_z(MeshData<T>& input,MeshData<uint8_t>& offset_img,float scale_in,unsigned int offset_max_in,const unsigned int d_max ){

        // The same, but in place

        const int64_t z_num = input.z_num;
        const int64_t x_num = input.x_num;
        const int64_t y_num = input.y_num;

        int64_t j,k;
        float temp;
        int64_t index_modulo, previous_modulo, iynum,forward_modulo,backward_modulo,offset;
        int64_t xnumynum = x_num * y_num;

        const int offset_max = offset_max_in;
        const float scale = scale_in;
        //const unsigned int d_max = this->depth_max();

        std::vector<T> temp_vec;
        temp_vec.resize(y_num*(2*offset_max + 2),0);

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(j,k,temp,index_modulo, previous_modulo,backward_modulo,forward_modulo, iynum,offset) \
        firstprivate(temp_vec)
#endif
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
    void interp_parts_smooth(APR<S>& apr,MeshData<U>& out_image,ExtraParticleData<V>& interp_data,std::vector<float> scale_d = {2,2,2}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        APRTimer timer;
        timer.verbose_flag = false;

        MeshData<U> pc_image;
        MeshData<uint8_t> k_img;

        unsigned int offset_max = 20;

        interp_img(apr,pc_image,interp_data);

        interp_level(apr, k_img);

        timer.start_timer("sat");
        //demo
        calc_sat_adaptive_y(pc_image,k_img,scale_d[0],offset_max,apr.level_max());

        timer.stop_timer();

        timer.start_timer("sat");

        calc_sat_adaptive_x(pc_image,k_img,scale_d[1],offset_max,apr.level_max());

        timer.stop_timer();

        timer.start_timer("sat");

        calc_sat_adaptive_z(pc_image,k_img,scale_d[2],offset_max,apr.level_max());

        timer.stop_timer();

        pc_image.swap(out_image);
    }



};


#endif //PARTPLAY_APRRECONSTRUCTION_HPP
