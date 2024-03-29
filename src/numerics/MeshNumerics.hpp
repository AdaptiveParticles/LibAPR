//
// Created by cheesema on 31.01.18.
//

#ifndef PARTPLAY_MESHNUMERICS_HPP
#define PARTPLAY_MESHNUMERICS_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "misc/APRTimer.hpp"
#include "io/TiffUtils.hpp"


template<typename T>
T min_nz(T x,T y){
    if(x>0){
        return std::min(x,y);
    } else {
        return y;
    }
}

class MeshNumerics {

public:

    template<typename T>
    void generate_smooth_stencil(std::vector<PixelData<T>>& stencils){

        unsigned int dim = 3;
//        unsigned int order = 1;
//        unsigned int padd = 1;

        std::vector<int> derivative = {0,0,0};

        PixelData<float> stencil_c;

        stencil_c.initWithValue(dim,dim,dim,1.0f/(27.0f));

       // GenerateStencils generateStencils;

        //generateStencils.solve_for_stencil(stencil_c,dim,order,padd,derivative);

        stencils.resize(2);

        stencils[0].init(stencil_c);
        stencils[0].copyFromMesh(stencil_c);

        //generateStencils.solve_for_stencil(stencil_c,dim,1,padd,derivative);


        stencils[1].init(stencil_c);
        stencils[1].copyFromMesh(stencil_c);

    }


    template<typename T,typename R>
    void apply_stencil(PixelData<T>& input,PixelData<R>& stencil){

        PixelData<T> output;
        output.init(input);

        int stencil_y_half = (stencil.y_num-1)/2;
        int stencil_x_half = (stencil.x_num-1)/2;
        int stencil_z_half = (stencil.z_num-1)/2;

        int i=0;
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(i)
#endif
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.x_num; ++j) {
                for (int k = 0; k < input.y_num; ++k) {
                    double neigh_sum = 0;
                    double counter = 0;

                    int stencil_z_b = std::max((-stencil_z_half+i),0);
                    int stencil_z_e = std::min((stencil_z_half+i),(input.z_num-1));

                    int stencil_x_b = std::max((-stencil_x_half+j),0);
                    int stencil_x_e = std::min((stencil_x_half+j),(input.x_num-1));

                    int stencil_y_b = std::max((-stencil_y_half+k),0);
                    int stencil_y_e = std::min((stencil_y_half+k),(input.y_num-1));


                    for (int l = stencil_z_b; l <= stencil_z_e; ++l) {
                        for (int q = stencil_x_b; q <= stencil_x_e; ++q) {
                            for (int w = stencil_y_b; w <= stencil_y_e; ++w) {

                                neigh_sum +=input.at(w, q, l);
                                counter++;
                            }
                        }
                    }

                    output.at(k, j, i) = neigh_sum/counter;

                }
            }
        }

        std::swap(output,input);

    }


    template<typename T>
    void erode_sol(PixelData<T>& input){
        PixelData<uint8_t> output;
        output.initWithValue(input.y_num,input.x_num,input.z_num,0);

        int i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(i)
#endif
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.x_num; ++j) {
                for (int k = 0; k < input.y_num; ++k) {

                    uint64_t counter=0;
                    if(input.at(k , j , i)>0) {
                        for (int l = -1; l < (1 + 1); ++l) {
                            for (int q = -1; q < (1 + 1); ++q) {
                                for (int w = -1; w < (1 + 1); ++w) {

                                    if (input(k + w, j + q, i + l) > 0) {
                                        counter++;
                                    }
                                }
                            }
                        }

                        if(counter== 27){
                            output.at(k, j, i) = 1;
                        } else {
                            output.at(k, j, i) = 0;
                        }
                    }

                }
            }
        }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(i)
#endif
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.x_num; ++j) {
                for (int k = 0; k < input.y_num; ++k) {

                    uint64_t counter=0;
                    if(input.at(k , j , i)>0) {
                        for (int l = -1; l < (1 + 1); ++l) {
                            for (int q = -1; q < (1 + 1); ++q) {
                                for (int w = -1; w < (1 + 1); ++w) {

                                    if (output(k + w, j + q, i + l) > 0) {
                                        counter++;
                                    }
                                }
                            }
                        }

                        if((counter > 0) && (output.at(k, j, i)==0)){
                            input.at(k , j , i) = 0;
                        }
                    }

                }
            }
        }

    }


    template<typename T>
    void smooth_sol(PixelData<T>& input){

        PixelData<T> output;
        output.init(input);

        int i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(i)
#endif
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.x_num; ++j) {
                for (int k = 0; k < input.y_num; ++k) {
                    float neigh_sum=0;
                    uint64_t counter=0;

                    for (int l = -1; l < (1+1); ++l) {
                        for (int q = -1; q < (1+1); ++q) {
                            for (int w = -1; w <(1+1); ++w) {

                                if(input(k + w, j + q, i+l) >0) {
                                    neigh_sum += input(k + w, j + q, i + l);

                                    counter++;
                                }
                            }
                        }
                    }
                    if(counter>0) {
                        output.at(k, j, i) = neigh_sum / (1.0f * counter);
                    } else {
                        output.at(k, j, i) = 0;
                    }


                }
            }
        }

        std::swap(output,input);

    }

    template<typename T>
    void smooth_mesh(PixelData<T>& input){

        PixelData<T> output;
        output.init(input);

        int i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(i)
#endif
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.x_num; ++j) {
                for (int k = 0; k < input.y_num; ++k) {
                    float neigh_sum=0;
                    uint64_t counter=0;

                    for (int l = -1; l < (1+1); ++l) {
                        for (int q = -1; q < (1+1); ++q) {
                            for (int w = -1; w <(1+1); ++w) {


                                neigh_sum +=  input(k + w, j + q, i+l);

                                counter++;
                            }
                        }
                    }

                    output.at(k,j,i) = neigh_sum/(1.0f*counter);


                }
            }
        }

        std::swap(output,input);

    }



    template<typename T>
    void fill_tree_vals(PixelData<T>& input,unsigned int level_min,unsigned int level_max){

        std::vector<PixelData<T>> temp_imgs;
        temp_imgs.resize(level_max);

        temp_imgs.back().swap(input);

        //upwards pass
        for (int i = (level_max-1); i >= level_min; --i) {

            smooth_sol(temp_imgs[i]);

            auto max_op = [](const float x, const float y) -> float { return std::max(x,y);};
            auto nothing = [](const float x) -> float { return x; };

            downsample(temp_imgs[i], temp_imgs[i-1], max_op, nothing, true);

        }

        //filling in the gaps
        double avg = 0;
        uint64_t counter = 0;
        for (int j = 0; j < temp_imgs[level_min-1].mesh.size(); ++j) {
            if(temp_imgs[level_min-1].mesh[j]>0){
                avg+=temp_imgs[level_min-1].mesh[j];
                counter++;
            }
        }
        avg = avg/(counter*1.0);

        int j = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
        for (j = 0; j < temp_imgs[level_min-1].mesh.size(); ++j) {
            if(temp_imgs[level_min-1].mesh[j]==0){
                temp_imgs[level_min-1].mesh[j]=avg;
            }
        }


        //downwards pass
        for (int i = level_min; i < level_max; ++i) {
            PixelData<T> up_sampled;
            up_sampled.init(temp_imgs[i]);

            const_upsample_img(up_sampled,temp_imgs[i-1]);

            int j = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
            for (j = 0; j < temp_imgs[i].mesh.size(); ++j) {
                if(temp_imgs[i].mesh[j]==0){
                    temp_imgs[i].mesh[j]=up_sampled.mesh[j];
                }
            }

            smooth_sol(temp_imgs[i]);

        }

        temp_imgs.back().swap(input);


    }


    template<typename T>
    void fill_vals(PixelData<T>& input){

        PixelData<T> output;
        output.initWithValue(input.y_num,input.x_num,input.z_num,1000);

        PixelData<float> counter;
        counter.initWithValue(input.y_num,input.x_num,input.z_num,1);


        //yloop

        int i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(i)
#endif
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.x_num; ++j) {
                float prev_val = 0;

                //forward
                for (int k = 0; k < input.y_num; ++k) {

                    float curr_val = input.at(k,j,i);

                    if((curr_val == 0) && (prev_val >0)){
                        output.at(k,j,i) += prev_val;
                        counter.at(k,j,i)++;

                    } else if(curr_val > 0) {
                        prev_val = curr_val;
                        output.at(k,j,i) = curr_val;
                        counter.at(k,j,i)=1;
                    }

                }
                prev_val = 0;
                //backward
                for (int k = (input.y_num-1); k >= 0; --k) {

                    float curr_val = input.at(k,j,i);

                    if((curr_val == 0) && (prev_val >0)){
                        output.at(k,j,i) += prev_val;
                        counter.at(k,j,i)++;

                    } else if(curr_val > 0) {
                        prev_val = curr_val;
                    }
                }
            }
        }


        //xloop

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(i)
#endif
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.y_num; ++j) {
                float prev_val = 0;

                //forward
                for (int k = 0; k < input.x_num; ++k) {

                    float curr_val = input.at(j,k,i);

                    if((curr_val == 0) && (prev_val >0)){
                        output.at(j,k,i) += prev_val;
                        counter.at(j,k,i)++;

                    } else if(curr_val > 0) {
                        prev_val = curr_val;
                        output.at(j,k,i) = curr_val;
                        counter.at(j,k,i)=1;
                    }

                }
                prev_val = 0;
                //backward
                for (int k = (input.x_num-1); k >= 0; --k) {

                    float curr_val = input.at(j,k,i);

                    if((curr_val == 0) && (prev_val >0)){
                        output.at(j,k,i) += prev_val;
                        counter.at(j,k,i)++;

                    } else if(curr_val > 0) {
                        prev_val = curr_val;
                    }
                }
            }
        }

        //zloop

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(i)
#endif
        for (i = 0; i < input.x_num; ++i) {
            for (int j = 0; j < input.y_num; ++j) {
                float prev_val = 0;

                //forward
                for (int k = 0; k < input.z_num; ++k) {

                    float curr_val = input.at(j,i,k);

                    if((curr_val == 0) && (prev_val >0)){
                        output.at(j,i,k) += prev_val;
                        counter.at(j,i,k)++;

                    } else if(curr_val > 0) {
                        prev_val = curr_val;
                        output.at(j,i,k) = curr_val;
                        counter.at(j,i,k)=1;
                    }

                }
                prev_val = 0;
                //backward
                for (int k = (input.z_num-1); k >= 0; --k) {

                    float curr_val = input.at(j,i,k);

                    if((curr_val == 0) && (prev_val >0)){
                        output.at(j,i,k) += prev_val;
                        counter.at(j,i,k)++;

                    } else if(curr_val > 0) {
                        prev_val = curr_val;
                    }
                }
            }
        }




        for (int l = 0; l < output.mesh.size(); ++l) {
                output.mesh[l]=output.mesh[l]/counter.mesh[l];
        }


        output.swap(input);

    }


    template<typename U,typename V>
    static float compute_gradient(const PixelData<U>& input_data,std::vector<PixelData<V>>& output_data,std::vector<float> delta = {1.0f,1.0f,1.0f}){
        //
        //  Computes gradient magnitude using finite differences
        //

        //#TODO add delta, do in each direction and compute the gradient, then later will compute the magnitude.
        //#TODO also add the delta for the gradient particle magnitude
        //#TODO also get the ray cast mesh going
        //#TODO add the options and mesh input for the two examples

        const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
        const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
        const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

        output_data.resize(3);
        output_data[0].init(input_data);
        output_data[1].init(input_data);
        output_data[2].init(input_data);

        uint64_t x_num = input_data.x_num;
        uint64_t y_num = input_data.y_num;
        uint64_t z_num = input_data.z_num;

        APRTimer timer;
        timer.verbose_flag = true;
        timer.start_timer("compute gradient mesh");

        size_t j = 0;
        size_t k = 0;
        size_t i = 0;

        const std::vector<std::vector<uint8_t>> group_directions = {{0,1},{2,3},{4,5}}; // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
        const std::vector<float> sign = {1.0f,-1.0f};

        int64_t j_n = 0;
        int64_t k_n = 0;
        int64_t i_n = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n)
#endif
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                for(k = 0;k < y_num;k++){

                    float current_intensity = input_data.mesh[j*x_num*y_num + i*y_num + k];


                    for (int dimension = 0; dimension < 3; ++dimension) {
                        float counter_dir = 0;
                        float gradient_estimate= 0;
                        for (int n = 0; n < 2; ++n) {

                            float intensity_sum = 0;
                            float count_neighbours = 0;

                            const uint8_t direction = group_directions[dimension][n];

                            i_n = i + dir_x[direction];
                            k_n = k + dir_y[direction];
                            j_n = j + dir_z[direction];

                            //check boundary conditions
                            if ((i_n >= 0) & (i_n < (int64_t)x_num)) {
                                if ((j_n >= 0) & (j_n < (int64_t)z_num)) {
                                    if ((k_n >= 0) & (k_n < (int64_t)y_num)) {
                                        intensity_sum += input_data.mesh[j_n * x_num * y_num + i_n * y_num + k_n];
                                        count_neighbours++;
                                    }
                                }
                            }

                            if(count_neighbours > 0) {
                                gradient_estimate += sign[n] * (current_intensity - intensity_sum / count_neighbours) /
                                                     delta[dimension]; //calculates the one sided finite difference in each direction using the average of particles
                                counter_dir++;
                            }
                        }

                        output_data[dimension].mesh[j*x_num*y_num + i*y_num + k] = gradient_estimate/counter_dir;
                    }
                }
            }
        }


        timer.stop_timer();
        float elapsed_seconds = (float)(timer.t2 - timer.t1);

        return (elapsed_seconds);
    }


    template<typename T, typename S>
    static void find_boundary(PixelData<T>& input, PixelData<S>& output) {

        output.initWithResize(input.y_num, input.x_num, input.z_num);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) default(none) shared(input, output)
#endif
        for(int z = 0; z < input.z_num; ++z) {
            for(int x = 0; x < input.x_num; ++x) {
                for(int y = 0; y < input.y_num; ++y) {
                    output.at(y, x, z) = 0;
                    const auto val = input.at(y, x, z);

                    if(val && (find_neighbor(input, y, x, z, val) != val)) {
                        output.at(y, x, z) = val;
                    }
                }
            }
        }
    }


    template<typename T, typename S>
    static void dilate(PixelData<T>& input, PixelData<S>& output) {

        output.initWithResize(input.y_num, input.x_num, input.z_num);
        output.copyFromMesh(input);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) default(none) shared(input, output)
#endif
        for (int z = 0; z < input.z_num; ++z) {
            for (int x = 0; x < input.x_num; ++x) {
                for (int y = 0; y < input.y_num; ++y) {
                    if (input.at(y, x, z) == 0) {
                        output.at(y, x, z) = find_neighbor(input, y, x, z);
                    }
                }
            }
        }
    }
};


template<typename T>
inline T find_neighbor(const PixelData<T>& input, const int y, const int x, const int z, const T comp_val = 0) {
    if(y > 0 && (input.at(y-1, x, z) != comp_val)) {
        return input.at(y-1, x, z);
    }
    if(y < input.y_num-1 && (input.at(y+1, x, z) != comp_val)) {
        return input.at(y+1, x, z);
    }
    if(x > 0 && (input.at(y, x-1, z) != comp_val)) {
        return input.at(y, x-1, z);
    }
    if(x < input.x_num-1 && (input.at(y, x+1, z) != comp_val)) {
        return input.at(y, x+1, z);
    }
    if(z > 0 && (input.at(y, x, z-1) != comp_val)) {
        return input.at(y, x, z-1);
    }
    if(z < input.z_num-1 && (input.at(y, x, z+1) != comp_val)) {
        return input.at(y, x, z+1);
    }
    return comp_val;
}


#endif //PARTPLAY_MESHNUMERICS_HPP
