//
// Created by msusik on 24.08.16.
//

#ifndef PARTPLAY_K_HPP
#define PARTPLAY_K_HPP

#include "../data_structures/particle_map.hpp"
#include "variance.hpp"


static inline uint32_t asmlog_2(const uint32_t x){
    if(x == 0) return 0;
    return (31 - __builtin_clz (x));
}

template< typename T>
void compute_k_for_array(Mesh_data<T>& input,float k_factor,float rel_error){
    //
    //  Takes the sqrt of the grad vector to caluclate the magnitude
    //
    //  Bevan Cheeseman 2016
    //

    float mult_const = k_factor/rel_error;

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    int i,k;

#pragma omp parallel for default(shared) private (i,k) if(z_num*x_num*y_num > 100000)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

#pragma omp simd
            for (k = 0; k < (y_num);k++){

                input.mesh[j*x_num*y_num + i*y_num + k] = asmlog_2(input.mesh[j*x_num*y_num + i*y_num + k]*mult_const);
            }

        }
    }

}


template< typename T>
void get_level_3D(Mesh_data<T> &var, Mesh_data<T> &grad_input, Part_rep &p_rep, Particle_map<T> &part_map,
                  Mesh_data<T> &grad){
    //
    //
    //  Calculate the local resolution estimates k
    //
    //


    Part_timer timer;
    //timer.verbose_flag = true;

    //first step is to perform a max down sample of the gradient
    timer.start_timer("level_downsample_main");

    down_sample(grad_input,grad,
                [](T x, T y) { return std::max(x,y); },
                [](T x) { return x; });

    timer.stop_timer();
    timer.start_timer("level_divide");
    //First need to divide the two grad and var, do this on the cpu

#pragma omp parallel for default(shared)
    for(int i = 0; i < grad.mesh.size(); i++)
    {
        grad.mesh[i] /= var.mesh[i];
    }

    timer.stop_timer();


    float k_factor;

    float min_dim = std::min(p_rep.pars.dy,std::min(p_rep.pars.dx,p_rep.pars.dz));

    k_factor = pow(2,p_rep.pl_map.k_max+1)*min_dim;

    timer.start_timer("level_kcalc");

    //calculate the k value at each time step

    compute_k_for_array(grad,k_factor,p_rep.pars.rel_error);

    timer.stop_timer();

    timer.start_timer("level_partmap");




    part_map.initialize(p_rep.org_dims);

    Mesh_data<T> test_a_ds;

    //first do k_max
    part_map.fill(p_rep.pl_map.k_max,grad);

    timer.stop_timer();
    timer.start_timer("level_loop");


    for(int k_ = p_rep.pl_map.k_max - 1; k_ >= p_rep.pl_map.k_min; k_--){

        //down sample the resolution level k, using a max reduction
        down_sample(grad,test_a_ds,
                    [](T x, T y) { return std::max(x,y); },
                    [](T x) { return x; }, true);
        //for those value of level k, add to the hash table
        part_map.fill(k_,test_a_ds);
        //assign the previous mesh to now be resampled.
        std::swap(grad, test_a_ds);

    }

    timer.stop_timer();

}

template< typename T>
void get_level_2D(Mesh_data<T> &var, Mesh_data<T> &grad_input, Part_rep &p_rep, Particle_map<T> &part_map,
                  Mesh_data<T> &grad){
    //
    //
    //  Calculate the local resolution estimates k
    //
    //


    Part_timer timer;
    //timer.verbose_flag = true;

    //first step is to perform a max down sample of the gradient
    timer.start_timer("level_downsample_main");

    down_sample(grad_input,grad,
                [](T x, T y) { return std::max(x,y); },
                [](T x) { return x; });

    timer.stop_timer();
    timer.start_timer("level_divide");
    //First need to divide the two grad and var, do this on the cpu

#pragma omp parallel for default(shared)
    for(int i = 0; i < grad.mesh.size(); i++)
    {
        grad.mesh[i] /= var.mesh[i];
    }

    timer.stop_timer();




    float k_factor;

    float min_dim = std::min(p_rep.pars.dy,std::min(p_rep.pars.dx,p_rep.pars.dz));

    k_factor = pow(2,p_rep.pl_map.k_max+1)*min_dim;

    timer.start_timer("level_kcalc");

    //calculate the k value at each time step

    compute_k_for_array(grad,k_factor,p_rep.pars.rel_error);

    timer.stop_timer();


    debug_write(grad,"kinput");

    timer.start_timer("level_partmap");


    part_map.initialize(p_rep.org_dims);

    Mesh_data<T> test_a_ds;

    //first do k_max
    part_map.fill(p_rep.pl_map.k_max,grad);

    timer.stop_timer();
    timer.start_timer("level_loop");


    for(int k_ = p_rep.pl_map.k_max - 1; k_ >= p_rep.pl_map.k_min; k_--){

        //down sample the resolution level k, using a max reduction
        down_sample(grad,test_a_ds,
                    [](T x, T y) { return std::max(x,y); },
                    [](T x) { return x; }, true);
        //for those value of level k, add to the hash table
        part_map.fill(k_,test_a_ds);
        //assign the previous mesh to now be resampled.
        std::swap(grad, test_a_ds);

    }

    timer.stop_timer();

}

template< typename T>
void get_level_3D_ground_truth(Mesh_data<T> &grad_input, Part_rep &p_rep, Particle_map<T> &part_map,
                  Mesh_data<T> &grad){
    //
    //
    //  Calculate the local resolution estimates k
    //
    //

    Part_timer timer;
    //timer.verbose_flag = true;

    //first step is to perform a max down sample of the gradient
    timer.start_timer("level_downsample_main");

    down_sample(grad_input,grad,
                [](T x, T y) { return std::max(x,y); },
                [](T x) { return x; });

    timer.stop_timer();

    float k_factor;

    float min_dim = std::min(p_rep.pars.dy,std::min(p_rep.pars.dx,p_rep.pars.dz));

    k_factor = pow(2,p_rep.pl_map.k_max+1)*min_dim;

    timer.start_timer("level_kcalc");

    //calculate the k value at each time step

    compute_k_for_array(grad,k_factor,p_rep.pars.rel_error);

    timer.stop_timer();

    timer.start_timer("level_partmap");


    part_map.initialize(p_rep.org_dims);

    Mesh_data<T> test_a_ds;

    //first do k_max
    part_map.fill(p_rep.pl_map.k_max,grad);

    timer.stop_timer();
    timer.start_timer("level_loop");


    for(int k_ = p_rep.pl_map.k_max - 1; k_ >= p_rep.pl_map.k_min; k_--){

        //down sample the resolution level k, using a max reduction
        down_sample(grad,test_a_ds,
                    [](T x, T y) { return std::max(x,y); },
                    [](T x) { return x; }, true);
        //for those value of level k, add to the hash table
        part_map.fill(k_,test_a_ds);
        //assign the previous mesh to now be resampled.
        std::swap(grad, test_a_ds);

    }

    timer.stop_timer();

}


#endif //PARTPLAY_K_HPP
