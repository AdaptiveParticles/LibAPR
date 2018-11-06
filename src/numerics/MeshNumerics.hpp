//
// Created by cheesema on 31.01.18.
//

#ifndef PARTPLAY_MESHNUMERICS_HPP
#define PARTPLAY_MESHNUMERICS_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "misc/APRTimer.hpp"

#include "../src/Numerics/GenerateStencils.hpp"

class MeshNumerics {

public:

    template<typename T>
    void generate_smooth_stencil(std::vector<PixelData<T>>& stencils){

        unsigned int dim = 3;
        unsigned int order = 1;
        unsigned int padd = 1;

        std::vector<int> derivative = {0,0,0};

        PixelData<float> stencil_c;

        GenerateStencils generateStencils;

        generateStencils.solve_for_stencil(stencil_c,dim,order,padd,derivative);

        stencils.resize(2);

        stencils[0].init(stencil_c);
        stencils[0].copyFromMesh(stencil_c);

        generateStencils.solve_for_stencil(stencil_c,dim,1,padd,derivative);

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
#pragma omp parallel for default(shared) private(i)
        for (i = 0; i < input.z_num; ++i) {
            for (int j = 0; j < input.x_num; ++j) {
                for (int k = 0; k < input.y_num; ++k) {
                    double neigh_sum = 0;

                    for (int l = -stencil_z_half; l <= stencil_z_half; ++l) {
                        for (int q = -stencil_x_half; q <= stencil_x_half; ++q) {
                            for (int w = -stencil_y_half; w <= stencil_y_half; ++w) {

                                neigh_sum += stencil.at(l+stencil_y_half,q+stencil_x_half,w+stencil_z_half)*input(k + w, j + q, i + l);
                            }
                        }
                    }

                    output.at(k, j, i) = neigh_sum;

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

                    for (int l = -1; l < 1+1; ++l) {
                        for (int q = -1; q < 1+1; ++q) {
                            for (int w = -1; w <1+1; ++w) {


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
};


#endif //PARTPLAY_MESHNUMERICS_HPP
