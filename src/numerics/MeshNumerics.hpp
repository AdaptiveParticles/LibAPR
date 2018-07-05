//
// Created by cheesema on 31.01.18.
//

#ifndef PARTPLAY_MESHNUMERICS_HPP
#define PARTPLAY_MESHNUMERICS_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "misc/APRTimer.hpp"

class MeshNumerics {

public:

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
