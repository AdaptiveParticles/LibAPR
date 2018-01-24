//
// Created by cheesema on 24.01.18.
//

#ifndef PARTPLAY_APRBENCHMARK_HPP
#define PARTPLAY_APRBENCHMARK_HPP


#include <benchmarks/analysis/AnalysisData.hpp>
#include "src/data_structures/APR/APR.hpp"

class APRBenchmark {

public:

    APRBenchmark(){

    }

    AnalysisData analysis_data;

    template<typename U,typename V>
    void pixels_linear_neighbour_access(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats);

    template<typename U,typename V>
    void apr_linear_neighbour_access(APR<U> apr,float num_repeats);

};

template<typename U,typename V>
void APRBenchmark::apr_linear_neighbour_access(APR<U> apr,float num_repeats){

    APRTimer timer;

    ExtraParticleData<V> output(apr);
    APRIterator<U> apr_iterator(apr);
    APRIterator<U> neighbour_iterator(apr);

    uint64_t particle_number;

    timer.start_timer("APR parallel iterator neighbour loop");

    for(int r = 0;r < num_repeats;r++) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator, neighbour_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)

            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            U neigh_sum = 0;
            U counter = 0;

            //loop over all the neighbours and set the neighbour iterator to it
            for (int direction = 0; direction < 6; ++direction) {
                apr_iterator.find_neighbours_in_direction(direction);
                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                    if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                        //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
                        neigh_sum += apr.particles_intensities[neighbour_iterator];
                        counter++;
                    }

                }
            }

            output[apr_iterator] = neigh_sum/(counter*1.0);

        }
    }


    timer.stop_timer();

    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds/num_repeats;

    std::cout << "APR Linear Neigh: " << (apr.total_number_particles()) << " took: " << time << std::endl;
    std::cout << "per 1000000 pixel took: " << (time)/((1.0*apr.total_number_particles())/1000000.0) << std::endl;

    analysis_data.add_float_data("neigh_pixel_linear_total",time);
    analysis_data.add_float_data("neigh_pixel_linear_perm",(time)/((1.0*apr.total_number_particles())/1000000.0));

}

template<typename U,typename V>
void APRBenchmark::pixels_linear_neighbour_access(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    MeshData<U> input_data;
    MeshData<V> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

    APRTimer timer;
    timer.verbose_flag = false;
    timer.start_timer("full pixel neighbour access");

    int j = 0;
    int k = 0;
    int i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    //float neigh_sum = 0;

    for(int r = 0;r < num_repeats;r++){

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n)
#endif
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                for(k = 0;k < y_num;k++){
                    U neigh_sum = 0;
                    U counter = 0;

                    for(int  d  = 0;d < 6;d++){

                        i_n = i + dir_x[d];
                        k_n = k + dir_y[d];
                        j_n = j + dir_z[d];

                        //check boundary conditions
                        if((i_n >=0) & (i_n < x_num) ){
                            if((j_n >=0) & (j_n < z_num) ){
                                if((k_n >=0) & (k_n < y_num) ){
                                    neigh_sum += input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                                    counter++;
                                }
                            }
                        }
                    }

                    output_data.mesh[j*x_num*y_num + i*y_num + k] = neigh_sum/(counter*1.0);

                }
            }
        }

    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds/num_repeats;

    std::cout << "Pixel Linear Neigh: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    std::cout << "per 1000000 pixel took: " << (time)/((1.0*x_num*y_num*z_num)/1000000.0) << std::endl;

    analysis_data.add_float_data("neigh_pixel_linear_total",time);
    analysis_data.add_float_data("neigh_pixel_linear_perm",(time)/((1.0*x_num*y_num*z_num)/1000000.0));

}


#endif //PARTPLAY_APRBENCHMARK_HPP
