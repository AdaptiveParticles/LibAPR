//
// Created by cheesema on 23.02.18.
//

#ifndef APR_TIME_APRCOMPUTEHELPER_HPP
#define APR_TIME_APRCOMPUTEHELPER_HPP

#include <data_structures/APR/APRTree.hpp>
#include <numerics/APRTreeNumerics.hpp>
#include "APRNumerics.hpp"
#include <algorithm/APRConverter.hpp>

template<typename ImageType>
class APRComputeHelper {
public:

    APRComputeHelper(APR<ImageType>& apr){
        apr_tree.init(apr);
    }

    APRComputeHelper(){
    }

    APRTree<ImageType> apr_tree;
    ExtraParticleData<ImageType> adaptive_max;
    ExtraParticleData<ImageType> adaptive_min;

    void init_tree(APR<ImageType>& apr){
        apr_tree.init(apr);
    }

    template<typename T>
    void compute_local_scale(APR<ImageType>& apr,ExtraParticleData<T>& local_intensity_scale,unsigned int smooth_iterations = 3,unsigned int smooth_seed = 7){

        if(apr_tree.total_number_parent_cells()==0) {
            apr_tree.init(apr);
        }

        APRTimer timer;
        timer.verbose_flag = true;

        timer.start_timer("smooth");

        APRNumerics aprNumerics;
        ExtraParticleData<uint16_t> smooth(apr);
        std::vector<float> filter = {0.1f, 0.8f, 0.1f}; // << Feel free to play with these
        //aprNumerics.seperable_smooth_filter(apr, apr.particles_intensities, smooth, filter, smooth_iterations);
        for (int i = 0; i < smooth_iterations; ++i) {
            aprNumerics.weight_neighbours(apr,apr.particles_intensities,smooth,0.5);
            std::swap(smooth.data,apr.particles_intensities.data);
        }

        std::swap(smooth.data,apr.particles_intensities.data);

        timer.stop_timer();

        unsigned int smoothing_steps_local = smooth_seed;

        timer.start_timer("adaptive min");

        APRTreeNumerics::calculate_adaptive_min(apr,apr_tree,smooth,adaptive_min,smoothing_steps_local);

        timer.stop_timer();



        timer.start_timer("adaptive max");
        APRTreeNumerics::calculate_adaptive_max(apr,apr_tree,smooth,adaptive_max,smoothing_steps_local);

        timer.stop_timer();

        local_intensity_scale.init(apr);
        adaptive_max.zip(apr,adaptive_min,local_intensity_scale, [](const uint16_t &a, const uint16_t &b) { return abs(a-b); });

        MeshData<uint16_t> boundary;
        apr.interp_img(boundary,adaptive_min);
        std::string image_file_name = apr.parameters.input_dir +  "min_seed4.tif";
        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);

    }
    template<typename T>
    void compute_local_scale_alternative(APR<ImageType>& apr,ExtraParticleData<T>& local_intensity_scale,unsigned int smooth_iterations = 3){

        if(apr_tree.total_number_parent_cells()==0) {
            apr_tree.init(apr);
        }

        APRTimer timer;
        timer.verbose_flag = true;

        timer.start_timer("adaptive min");

        APRTreeNumerics::calculate_adaptive_min_2(apr,apr_tree,apr.particles_intensities,adaptive_min,smooth_iterations);

        timer.stop_timer();

        timer.start_timer("adaptive max");
        APRTreeNumerics::calculate_adaptive_max_2(apr,apr_tree,apr.particles_intensities,adaptive_max,smooth_iterations);

        timer.stop_timer();

        local_intensity_scale.init(apr);
        adaptive_max.zip(apr,adaptive_min,local_intensity_scale, [](const uint16_t &a, const uint16_t &b) { return abs(a-b); });

    }

    template<typename T,typename U,typename V>
    void compute_apr_edge_energy(APR<ImageType>& apr,ExtraParticleData<T>& edge_energy,ExtraParticleData<V>& input_particles,ExtraParticleData<U>& local_intensity_scale,float scale_factor,float min_var = 1,float Ip_th = 0,std::vector<float> delta = {1,1,1}){

        ExtraParticleData<T> gradient; //vector for holding the derivative in the three directions, initialized to have the same number of elements as particles.

        APRNumerics::compute_gradient_magnitude(apr,input_particles,gradient,delta);

        edge_energy.init(apr);

        APRIterator<uint16_t> apr_iterator(apr);

        uint64_t particle_number = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            if(input_particles[apr_iterator] > Ip_th) {
                edge_energy[apr_iterator] = scale_factor * gradient[apr_iterator] /
                                            (std::max(local_intensity_scale[apr_iterator] * 1.0f, min_var));
            } else {
                edge_energy[apr_iterator] = 0;
            }
            //edge_energy[apr_iterator] = scale_factor*gradient[apr_iterator];
        }


    }

    template<typename T,typename U,typename V>
    void compute_apr_interior_energy(APR<ImageType>& apr,ExtraParticleData<T>& interior_energy,ExtraParticleData<V>& input_particles,ExtraParticleData<U>& local_intensity_scale,float scale_factor,float min_var,float Ip_th){
        //assumes you have computed apr_min with this function

        interior_energy.init(apr);

        APRIterator<uint16_t> apr_iterator(apr);

        uint64_t particle_number = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            if(input_particles[apr_iterator] > Ip_th) {
                interior_energy[apr_iterator] =
                        scale_factor * (input_particles[apr_iterator] - adaptive_min[apr_iterator]) /
                        (std::max(local_intensity_scale[apr_iterator] * 1.0f, min_var));
            } else {
                interior_energy[apr_iterator] = 0;
            }

        }
    }

    void compute_local_scale_smooth_propogate(APR<ImageType>& apr,MeshData<ImageType>& input_image,ExtraParticleData<ImageType>& local_intensity_scale){

        APRConverter<ImageType> aprConverter;
        unsigned int smooth_factor = 15;

        MeshData<float> local_scale_temp;

        MeshData<float> local_scale_temp2;

        downsample(input_image, local_scale_temp,
                   [](const float &x, const float &y) -> float { return x + y; },
                   [](const float &x) -> float { return x / 8.0; },true);

        local_scale_temp2.init(local_scale_temp);

        aprConverter.get_local_intensity_scale(local_scale_temp,local_scale_temp2);

        APRTreeIterator<uint16_t> aprTreeIterator(apr_tree);
        uint64_t parent_number;

        local_intensity_scale.init(apr);

        ExtraParticleData<uint16_t> local_intensity_scale_tree(apr_tree);

        for (parent_number = aprTreeIterator.particles_level_begin(aprTreeIterator.level_max());
             parent_number < aprTreeIterator.particles_level_end(aprTreeIterator.level_max()); ++parent_number) {

            aprTreeIterator.set_iterator_to_particle_by_number(parent_number);

            local_intensity_scale_tree[aprTreeIterator] = (ImageType)local_scale_temp.at(aprTreeIterator.y_nearest_pixel(),aprTreeIterator.x_nearest_pixel(),aprTreeIterator.z_nearest_pixel());

        }

        for (int i = 0; i < smooth_factor; ++i) {
            //smooth step
            for (unsigned int level = (aprTreeIterator.level_max());
                 level >= aprTreeIterator.level_min(); --level) {
                for (parent_number = aprTreeIterator.particles_level_begin(level);
                     parent_number < aprTreeIterator.particles_level_end(level); ++parent_number) {

                    aprTreeIterator.set_iterator_to_particle_by_number(parent_number);

                    float temp = local_intensity_scale_tree[aprTreeIterator];
                    float counter = 1;

                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        if (aprTreeIterator.find_neighbours_same_level(direction)) {

                            if (aprTreeIterator.set_neighbour_iterator(aprTreeIterator, direction, 0)) {

                                if (local_intensity_scale_tree[aprTreeIterator] > 0) {
                                    temp += local_intensity_scale_tree[aprTreeIterator];
                                    counter++;
                                }
                            }
                        }
                    }

                    local_intensity_scale_tree[aprTreeIterator] = temp / counter;

                }
            }
        }

        uint64_t particle_number;
        APRIterator<uint16_t> apr_iterator(apr);

        APRTreeIterator<uint16_t> parent_iterator(apr_tree);

        //Now set the highest level particle cells.
        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max());
             particle_number <
             apr_iterator.particles_level_end(apr_iterator.level_max()); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            parent_iterator.set_iterator_to_parent(apr_iterator);
            local_intensity_scale[apr_iterator] = local_intensity_scale_tree[parent_iterator];

        }


        APRTreeIterator<uint16_t> neighbour_tree_iterator(apr_tree);


        ExtraParticleData<uint16_t> boundary_type(apr);

        //spread solution

        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max() - 1);
             particle_number <
             apr_iterator.particles_level_end(apr_iterator.level_max() - 1); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            //now we only update the neighbours, and directly access them through a neighbour iterator

            if (apr_iterator.type() == 2) {

                float temp = 0;
                float counter = 0;

                float counter_neigh = 0;

                aprTreeIterator.set_particle_cell_no_search(apr_iterator);

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    // Neighbour aprTreeIterator Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    if (aprTreeIterator.find_neighbours_same_level(direction)) {

                        if (neighbour_tree_iterator.set_neighbour_iterator(aprTreeIterator, direction, 0)) {
                            temp += local_intensity_scale_tree[neighbour_tree_iterator];
                            counter++;

                        }
                    }
                }
                if(counter>0) {
                    local_intensity_scale[apr_iterator] = temp/counter;
                    boundary_type[apr_iterator] = apr_iterator.level_max();
                }

            }
        }

        APRIterator<uint16_t> neigh_iterator(apr);


        for (int level = (apr_iterator.level_max()-1); level >= apr_iterator.level_min() ; --level) {

            bool still_empty = true;
            while(still_empty) {
                still_empty = false;
                for (particle_number = apr_iterator.particles_level_begin(level);
                     particle_number <
                     apr_iterator.particles_level_end(level); ++particle_number) {
                    //This step is required for all loops to set the iterator by the particle number
                    apr_iterator.set_iterator_to_particle_by_number(particle_number);

                    if (local_intensity_scale[apr_iterator] == 0) {

                        float counter = 0;
                        float temp = 0;

                        //loop over all the neighbours and set the neighbour iterator to it
                        for (int direction = 0; direction < 6; ++direction) {
                            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                            if (apr_iterator.find_neighbours_in_direction(direction)) {

                                if (neigh_iterator.set_neighbour_iterator(apr_iterator, direction, 0)) {

                                    if(boundary_type[neigh_iterator]>=(level+1)) {
                                        counter++;
                                        temp += local_intensity_scale[neigh_iterator];
                                    }
                                }
                            }
                        }

                        if (counter > 0) {
                            local_intensity_scale[apr_iterator] = temp / counter;
                            boundary_type[apr_iterator] = level;
                        } else {
                            still_empty = true;
                        }
                    } else {
                        boundary_type[apr_iterator]=level+1;
                    }
                }
            }
        }


    }

    void compute_local_min_max_apr(APRConverter<ImageType>& apr_converter,APR<ImageType>& apr,APR<ImageType>& apr_new){

        TiffUtils::TiffInfo inputTiff(apr_converter.par.input_dir + apr_converter.par.input_image_name);
        MeshData<uint16_t> input_img= TiffUtils::getMesh<uint16_t>(inputTiff);

        ExtraParticleData<uint16_t> local_intensity_scale;

        this->apr_tree.init(apr);

        this->compute_local_scale_alternative(apr,local_intensity_scale,2);

        /*
        *  Compute APR using custom local intensity scale
        *
         *  Using the below approach you can use your own methods for calculating the gradient and local intensity scale, and then
         *  use them to compute the APR.
         *
        * */

        MeshData<uint16_t> custom_local_scale;
        MeshData<uint16_t> custom_grad; //if it is not initialized, the classic approach to using the gradient will be used.
        apr.interp_img(custom_local_scale,local_intensity_scale);

        //compute apr with alternative local intensity scale
        apr_converter.get_apr_method_custom_gradient_and_scale(apr_new,input_img,custom_grad,custom_local_scale);

    }


};


#endif //APR_TIME_APRCOMPUTEHELPER_HPP
