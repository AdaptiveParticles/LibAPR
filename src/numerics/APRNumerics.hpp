//
// Created by cheesema on 30.01.18.
//

#ifndef PARTPLAY_APRNUMERICS_HPP
#define PARTPLAY_APRNUMERICS_HPP

#include "data_structures/APR/APR.hpp"


class APRNumerics {

public:

    template<typename U>
    static void compute_part_level(APR &apr,ParticleData<U>& parts_level){
        parts_level.data.resize(apr.total_number_particles());

        auto apr_iterator = apr.iterator();

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        parts_level[apr_iterator] = level;

                    }
                }
            }
        }

    }


    template<typename T>
    static void compute_gradient_vector(APR &apr,ParticleData<T>& particleData,ParticleData<std::vector<float>>& gradient,const bool normalize = true,const std::vector<float> delta = {1.0f,1.0f,1.0f}){


        APRTimer timer;
        timer.verbose_flag = true;

        std::vector<float> init_val = {0,0,0};

        gradient.data.resize(apr.total_number_particles(),init_val);

        auto apr_iterator = apr.iterator();
        auto neighbour_iterator = apr.iterator();


        const std::vector<std::vector<uint8_t>> group_directions = {{0,1},{2,3},{4,5}}; // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
        const std::vector<float> sign = {1.0,-1.0};

        timer.start_timer("Calculate the gradient in each direction for the APR");

        //
        //  Calculates an estimate of the gradient in each direciton, using an average of two one sided FD of the gradient using the average of particles for children.
        //

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator, neighbour_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float current_intensity = particleData[apr_iterator];

                        //loop over all the neighbours and set the neighbour iterator to it
                        for (int dimension = 0; dimension < 3; ++dimension) {
                            float gradient_estimate = 0;

                            float counter_dir = 0;

                            for (int i = 0; i < 2; ++i) {
                                float intensity_sum = 0;
                                float count_neighbours = 0;

                                const uint8_t direction = group_directions[dimension][i];

                                apr_iterator.find_neighbours_in_direction(direction);

                                const float distance_between_particles =
                                        0.5f * pow(2.0f, (float) (apr_iterator.level_max() - apr_iterator.level())) +
                                        0.5f *
                                        pow(2.0f, (float) (apr_iterator.level_max() - neighbour_iterator.level())) *
                                        delta[dimension]; //in pixels

                                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                                for (int index = 0;
                                     index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                                    if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                        intensity_sum += particleData[neighbour_iterator];
                                        count_neighbours++;
                                    }
                                }
                                if (count_neighbours > 0) {
                                    gradient_estimate +=
                                            sign[i] * (current_intensity - intensity_sum / count_neighbours) /
                                            distance_between_particles; //calculates the one sided finite difference in each direction using the average of particles
                                    counter_dir++;
                                }
                            }
                            //store the estimate of the gradient
                            gradient[apr_iterator][dimension] = gradient_estimate / counter_dir;
                        }

                        if (normalize) {

                            float gradient_mag = sqrt(gradient[apr_iterator][0] * gradient[apr_iterator][0] +
                                                      gradient[apr_iterator][1] * gradient[apr_iterator][1] +
                                                      gradient[apr_iterator][2] * gradient[apr_iterator][2]);

                            if(gradient_mag > 0) {
                                gradient[apr_iterator][0] /= gradient_mag;
                                gradient[apr_iterator][1] /= gradient_mag;
                                gradient[apr_iterator][2] /= gradient_mag;
                            }
                        }

                    }
                }
            }
        }

        timer.stop_timer();

    }

    template<typename S,typename U>
    void seperable_smooth_filter(APR &apr,const ParticleData<S>& input_data,ParticleData<U>& output_data,const std::vector<float>& filter,unsigned int repeats = 1){

        output_data.init(apr.total_number_particles());

        ParticleData<U> output_data_2(apr.total_number_particles());
        output_data_2.copy_parts(apr,input_data);

        for (unsigned int i = 0; i < repeats; ++i) {
            face_neighbour_filter(apr,output_data_2,output_data,filter,0);
            face_neighbour_filter(apr,output_data,output_data_2,filter,1);
            face_neighbour_filter(apr,output_data_2,output_data,filter,2);
            std::swap(output_data_2.data,output_data.data);
        }

        std::swap(output_data_2.data,output_data.data);
    }


    template<typename S,typename U>
    void face_neighbour_filter(APR &apr,ParticleData<S>& input_data,ParticleData<U>& output_data,const std::vector<float>& filter,const int direction) {

        std::vector<uint8_t> faces;
        if (direction == 0) {
            faces = {0, 1};
        } else if (direction == 1) {
            faces = {2, 3};
        } else {
            faces = {4, 5};
        }

        auto apr_iterator = apr.iterator();
        auto neighbour_iterator = apr.iterator();


        const std::vector<float> filter_t = {filter[2], filter[0]};

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator, neighbour_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float current_intensity = input_data[apr_iterator];
                        output_data[apr_iterator] = current_intensity * filter[1];

                        for (int i = 0; i < 2; ++i) {
                            float intensity_sum = 0;
                            float count_neighbours = 0;

                            const uint8_t direction = faces[i];

                            apr_iterator.find_neighbours_in_direction(direction);

                            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                            for (int index = 0;
                                 index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                                if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                    intensity_sum += input_data[neighbour_iterator];
                                    count_neighbours++;
                                }
                            }
                            if (count_neighbours > 0) {
                                output_data[apr_iterator] += filter_t[i] * intensity_sum / count_neighbours;
                            } else {
                                output_data[apr_iterator] += filter_t[i] * current_intensity;
                            }
                        }
                    }
                }
            }
        }
    }
};


#endif //PARTPLAY_APRNUMERICS_HPP
