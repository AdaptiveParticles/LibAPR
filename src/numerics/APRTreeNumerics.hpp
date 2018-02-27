//
// Created by cheesema on 15.02.18.
//

#ifndef LIBAPR_APRTREENUMERICS_HPP
#define LIBAPR_APRTREENUMERICS_HPP

#include "src/data_structures/APR/APRTree.hpp"
#include "src/data_structures/APR/APRTreeIterator.hpp"
#include "APRNumerics.hpp"

class APRTreeNumerics {


public:
    template<typename T,typename S,typename U,typename BinaryOperation>
    static void fill_tree_from_particles(APR<T>& apr,APRTree<T>& apr_tree,ExtraParticleData<S>& particle_data,ExtraParticleData<U>& tree_data,BinaryOperation op,const bool normalize = false) {

        tree_data.init_tree(apr_tree);

        std::fill(tree_data.data.begin(),tree_data.data.end(),0);

        APRTreeIterator<T> treeIterator(apr_tree);
        APRTreeIterator<T> parentIterator(apr_tree);

        APRIterator<T> apr_iterator(apr);

        ExtraParticleData<uint8_t> child_counter;

        if(normalize){
            child_counter.init_tree(apr_tree);
        }

        uint64_t particle_number = 0;
        uint64_t parent_number = 0;

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,parentIterator)
        for (particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            //set parent
            parentIterator.set_iterator_to_parent(apr_iterator);

            tree_data[parentIterator] = op((U) particle_data[apr_iterator], (U) tree_data[parentIterator]);

            if(normalize){
                child_counter[parentIterator]++;
            }

        }


        //then do the rest of the tree where order matters
        for (unsigned int level = treeIterator.level_max(); level >= treeIterator.level_min(); --level) {
#pragma omp parallel for schedule(static) private(parent_number) firstprivate(treeIterator,parentIterator)
            for (parent_number = treeIterator.particles_level_begin(level);
                 parent_number < treeIterator.particles_level_end(level); ++parent_number) {

                treeIterator.set_iterator_to_particle_by_number(parent_number);

                if(parentIterator.set_iterator_to_parent(treeIterator)) {

                    tree_data[parentIterator] = op((U) tree_data[treeIterator], (U) tree_data[parentIterator]);
                    if(normalize){
                        child_counter[parentIterator]++;
                    }
                }

            }
        }

        if(normalize){
            for (unsigned int level = treeIterator.level_max(); level >= treeIterator.level_min(); --level) {
#pragma omp parallel for schedule(static) private(parent_number) firstprivate(treeIterator)
                for (parent_number = treeIterator.particles_level_begin(level);
                     parent_number < treeIterator.particles_level_end(level); ++parent_number) {

                    treeIterator.set_iterator_to_particle_by_number(parent_number);

                    tree_data[treeIterator]/=(1.0*child_counter[treeIterator]);

                }
            }
        }




    }

    template<typename T,typename S,typename U>
    static void pull_down_tree_to_particles(APR<T>& apr,APRTree<T>& apr_tree,ExtraParticleData<S>& particle_data,ExtraParticleData<U>& tree_data,uint8_t level_offset) {
        //
        //  Retrieves a value "level_offset" values up the tree and returns them as Particle data
        //

        particle_data.init(apr);

        APRTreeIterator<T> parentIterator(apr_tree);

        APRIterator<T> apr_iterator(apr);

        uint64_t particle_number = 0;
        uint64_t parent_number = 0;

        for (particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            //set parent
            if(parentIterator.set_iterator_to_parent(apr_iterator)) {

                uint8_t current_level_offset = 1;

                while ((parentIterator.level() > parentIterator.level_min()) & (current_level_offset < level_offset)) {

                    parentIterator.set_iterator_to_parent(parentIterator);
                    current_level_offset++;

                }

                particle_data[apr_iterator] = tree_data[parentIterator];
            }

        }

    };

    template<typename T,typename S>
    static void calculate_adaptive_min(APR<T>& apr,APRTree<T>& apr_tree,ExtraParticleData<S>& intensities,ExtraParticleData<S>& adaptive_min,unsigned int smooth_factor = 7){


        APRTimer timer;
        timer.verbose_flag = false;
        timer.start_timer("fill");

        ExtraParticleData<float> mean_tree;
        APRTreeNumerics::fill_tree_from_particles(apr,apr_tree,intensities,mean_tree,[] (const float& a,const float& b) {return a+b;},true);
        timer.stop_timer();

        timer.start_timer("init");
        APRTreeIterator<uint16_t> apr_tree_iterator(apr_tree);

        APRTreeIterator<uint16_t> neighbour_tree_iterator(apr_tree);
        APRIterator<uint16_t> apr_iterator(apr);
        APRIterator<uint16_t> neigh_iterator(apr);

        ExtraParticleData<uint16_t> boundary_type(apr);

        ExtraParticleData<uint64_t> child_counter(apr_tree);
        ExtraParticleData<uint64_t> child_counter_temp(apr_tree);
        ExtraParticleData<double> tree_min(apr_tree);
        ExtraParticleData<double> tree_min_temp(apr_tree);

        timer.stop_timer();

        timer.start_timer("first loop");

        //Basic serial iteration over all particles
        uint64_t particle_number;
        //Basic serial iteration over all particles
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,neigh_iterator,apr_tree_iterator,neighbour_tree_iterator)
        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max()-1);
             particle_number <
             apr_iterator.particles_level_end(apr_iterator.level_max()-1); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            //now we only update the neighbours, and directly access them through a neighbour iterator

            if(apr_iterator.type() == 2) {

                float counter = 1;
                float temp = intensities[apr_iterator];

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    if(apr_iterator.find_neighbours_same_level(direction)) {

                        if (neigh_iterator.set_neighbour_iterator(apr_iterator, direction, 0)) {
                            counter++;
                            temp+=intensities[neigh_iterator];

                        }
                    }
                }


                temp= temp/counter;
                counter=0;
                apr_tree_iterator.set_particle_cell_no_search(apr_iterator);

                float counter_total=0;

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    if(apr_tree_iterator.find_neighbours_same_level(direction)) {

                        if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {
                            if(temp < mean_tree[neighbour_tree_iterator]){
                                counter++;
                            }

                            counter_total++;

                        }
                    }
                }

                if(counter > 0) {
                    //counter = 1.0;

                    if (counter/counter_total ==1) {
                        boundary_type[apr_iterator] = 1;

                        if (apr_tree_iterator.set_iterator_to_parent(apr_iterator)) {

                            tree_min[apr_tree_iterator] += temp;
                            child_counter[apr_tree_iterator]++;
                            child_counter_temp[apr_tree_iterator]=child_counter[apr_tree_iterator];
                            tree_min_temp[apr_tree_iterator] = tree_min[apr_tree_iterator];

                        }

                    } else {
                        boundary_type[apr_iterator] = 0;
                    }
                }
            }
        }

        timer.stop_timer();

        //MeshData<uint16_t> boundary;
//        apr.interp_img(boundary,boundary_type);
//        std::string image_file_name = apr.parameters.input_dir +  "boundary_type.tif";
//        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);
//
//        apr.interp_img(boundary,boundary_type);
//        image_file_name = apr.parameters.input_dir +  "boundary_int.tif";
//        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);

        timer.start_timer("loop 2");

        APRTreeIterator<uint16_t> parent_it(apr_tree);

        uint64_t parent_number;
        //then do the rest of the tree where order matters

        for (unsigned int level = (apr_tree_iterator.level_max()-1); level > apr_tree_iterator.level_min(); --level) {

#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator,neighbour_tree_iterator)
            //two loops first spread
            for (parent_number = apr_tree_iterator.particles_level_begin(level);
                 parent_number < apr_tree_iterator.particles_level_end(level); ++parent_number) {

                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

                //maybe spread first, then normalize, then push upwards..

                if (child_counter[apr_tree_iterator] > 1) {
                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        if(apr_tree_iterator.find_neighbours_same_level(direction)) {

                            if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {

                                tree_min_temp[neighbour_tree_iterator]+=tree_min[apr_tree_iterator];
                                child_counter_temp[neighbour_tree_iterator]+=child_counter[apr_tree_iterator];
                            }
                        }
                    }
                }
            }

#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator,parent_it)
            //then average and push up
            for (parent_number = apr_tree_iterator.particles_level_begin(level);
                 parent_number < apr_tree_iterator.particles_level_end(level); ++parent_number) {

                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);
                //maybe spread first, then normalize, then push upwards..

                if(child_counter_temp[apr_tree_iterator]>1){
                    tree_min[apr_tree_iterator] = tree_min_temp[apr_tree_iterator]/(child_counter_temp[apr_tree_iterator]*1.0f);
                    //tree_min[apr_tree_iterator] = tree_min[apr_tree_iterator]/(child_counter[apr_tree_iterator]*1.0f);
                    child_counter[apr_tree_iterator]=1;
                } else {
                    tree_min[apr_tree_iterator] = 0;
                    child_counter[apr_tree_iterator] = 0;
                }

                parent_it.set_iterator_to_parent(apr_tree_iterator);

                if(tree_min[apr_tree_iterator] > 0){
                    tree_min[parent_it]+=tree_min[apr_tree_iterator];
                    child_counter[parent_it]++;

                    child_counter_temp[parent_it]=child_counter[parent_it];
                    tree_min_temp[parent_it] = tree_min[parent_it];
                }
            }
        }

        timer.stop_timer();

        timer.start_timer("loop 3");

#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator,parent_it)
        for (parent_number = apr_tree_iterator.particles_level_begin(apr_tree_iterator.level_max());
             parent_number < apr_tree_iterator.particles_level_end(apr_tree_iterator.level_max()); ++parent_number) {

            apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

            parent_it.set_iterator_to_parent(apr_tree_iterator);

            while((parent_it.level() > parent_it.level_min()) && (child_counter[parent_it] == 0)){
                parent_it.set_iterator_to_parent(parent_it);
            }

            tree_min[apr_tree_iterator] = tree_min[parent_it];

        }

        timer.stop_timer();

        timer.start_timer("loop 3b");


        for (int i = 0; i < smooth_factor; ++i) {
            //smoothing step

#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator,neighbour_tree_iterator)
            for (parent_number = apr_tree_iterator.particles_level_begin(apr_tree_iterator.level_max());
                 parent_number <
                 apr_tree_iterator.particles_level_end(apr_tree_iterator.level_max()); ++parent_number) {

                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

                float temp = tree_min[apr_tree_iterator];
                float counter = 1;

                for (int direction = 0; direction < 6; ++direction) {
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    if (apr_tree_iterator.find_neighbours_same_level(direction)) {

                        if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {
                            temp += tree_min[neighbour_tree_iterator];
                            counter++;

                        }
                    }
                }

                tree_min_temp[apr_tree_iterator] = temp / counter;

            }
            std::swap(tree_min.data,tree_min_temp.data);
        }


        timer.stop_timer();

        timer.start_timer("loop 4");

        adaptive_min.init(apr);

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,parent_it)
        //Now set the highest level particle cells.
        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max());
                 particle_number <
                 apr_iterator.particles_level_end(apr_iterator.level_max()); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            parent_it.set_iterator_to_parent(apr_iterator);
            adaptive_min[apr_iterator] = tree_min[parent_it];

        }

        timer.stop_timer();

        MeshData<uint16_t> boundary;
        apr.interp_img(boundary,adaptive_min);
        std::string image_file_name = apr.parameters.input_dir +  "min_seed.tif";
        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);

        //spread solution

        std::fill(boundary_type.data.begin(),boundary_type.data.end(),0);

        timer.start_timer("loop 5");
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,apr_tree_iterator,neighbour_tree_iterator)
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

                apr_tree_iterator.set_particle_cell_no_search(apr_iterator);

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    if (apr_tree_iterator.find_neighbours_same_level(direction)) {

                        if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {
                            temp += tree_min[neighbour_tree_iterator];
                            counter++;

                        }
                    }
                }
                if(counter>0) {
                    adaptive_min[apr_iterator] = temp/counter;
                    boundary_type[apr_iterator] = apr_iterator.level_max();
                }

            }
        }


        apr.interp_img(boundary,adaptive_min);
        image_file_name = apr.parameters.input_dir +  "min_seed2.tif";
        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);

        uint64_t loop_counter = 1;
        timer.stop_timer();

        timer.start_timer("loop final");
        int maximum_iteration = 20;

        for (int level = (apr_iterator.level_max()-1); level >= apr_iterator.level_min() ; --level) {


            uint64_t empty_counter = 0;
            bool still_empty = true;
            while(still_empty & empty_counter < maximum_iteration) {
                still_empty = false;
                empty_counter++;

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,neigh_iterator) reduction(||:still_empty)
                for (particle_number = apr_iterator.particles_level_begin(level);
                     particle_number <
                     apr_iterator.particles_level_end(level); ++particle_number) {
                    //This step is required for all loops to set the iterator by the particle number
                    apr_iterator.set_iterator_to_particle_by_number(particle_number);

                    if (boundary_type[apr_iterator] == 0) {

                        float counter = 0;
                        float temp = 0;

                        //loop over all the neighbours and set the neighbour iterator to it
                        for (int direction = 0; direction < 6; ++direction) {
                            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                            if (apr_iterator.find_neighbours_in_direction(direction)) {

                                if (neigh_iterator.set_neighbour_iterator(apr_iterator, direction, 0)) {

                                    float n_l = neigh_iterator.level();
                                    float n_t = boundary_type[neigh_iterator];

                                    if(boundary_type[neigh_iterator]>=(level+1)) {
                                        counter++;
                                        temp += adaptive_min[neigh_iterator];

                                    }
                                }
                            }
                        }

                        if (counter > 0) {
                            adaptive_min[apr_iterator] = temp / counter;
                            boundary_type[apr_iterator] = (uint16_t)level;
                        } else {

                            still_empty = true;

                        }
                    } else {
                        boundary_type[apr_iterator]=(uint16_t)(level+1);
                    }
                }
            }

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,parent_it)
            for (particle_number = apr_iterator.particles_level_begin(level);
                 particle_number <
                 apr_iterator.particles_level_end(level); ++particle_number) {
                //This step is required for all loops to set the iterator by the particle number
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                if(boundary_type[apr_iterator] == 0){

                    parent_it.set_iterator_to_parent(apr_iterator);

                    while((parent_it.level() > parent_it.level_min()) && (child_counter[parent_it] == 0)){
                        parent_it.set_iterator_to_parent(parent_it);
                    }

                    adaptive_min[apr_iterator] = tree_min[parent_it];
                    boundary_type[apr_iterator] = (uint16_t)(level + 1);
                }

            }
        }

        timer.stop_timer();

        apr.interp_img(boundary,adaptive_min);
        image_file_name = apr.parameters.input_dir +  "min_seed3.tif";
        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);

    }

    template<typename T,typename S>
    static void calculate_adaptive_max(APR<T>& apr,APRTree<T>& apr_tree,ExtraParticleData<S>& intensities,ExtraParticleData<S>& adaptive_max,unsigned int smooth_factor = 7) {

        ExtraParticleData<float> mean_tree;
        APRTreeNumerics::fill_tree_from_particles(apr, apr_tree, intensities, mean_tree,
                                                  [](const float &a, const float &b) { return a + b; }, true);

        //APRTreeNumerics::fill_tree_from_particles(apr, apr_tree, intensities, mean_tree,
                                                //  [](const float &a, const float &b) { return std::max(a,b); }, false);

        APRTreeIterator<uint16_t> apr_tree_iterator(apr_tree);
        APRTreeIterator<uint16_t> parent_iterator(apr_tree);

        APRTreeIterator<uint16_t> neighbour_tree_iterator(apr_tree);
        APRIterator<uint16_t> apr_iterator(apr);
        APRIterator<uint16_t> neigh_iterator(apr);

        ExtraParticleData<uint16_t> boundary_type(apr);

        ExtraParticleData<float> max_spread(apr_tree);
        ExtraParticleData<uint64_t> max_counter(apr_tree);

        ExtraParticleData<float> max_spread_temp(apr_tree);
        ExtraParticleData<uint64_t> max_counter_temp(apr_tree);

        //Basic serial iteration over all particles
        uint64_t particle_number;
        //Basic serial iteration over all particles


        uint64_t parent_number;
//        float weight = 0.5;
//
//        ExtraParticleData<float> mean_tree_temp(apr_tree);
//
//        unsigned int num_rep_smooth = 5;
//
//        for (int j = 0; j < num_rep_smooth; ++j) {
//            for (unsigned int level = (apr_tree_iterator.level_max());
//                 level >= apr_tree_iterator.level_min(); --level) {
//#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator, neighbour_tree_iterator, parent_iterator)
//                for (parent_number = apr_tree_iterator.particles_level_begin(level);
//                     parent_number <
//                     apr_tree_iterator.particles_level_end(level); ++parent_number) {
//
//                    apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);
//
//                    float temp = 0;
//                    float counter = 0;
//                    float counter_neigh = 0;
//
//
//                    float val = 0;
//
//                    //loop over all the neighbours and set the neighbour iterator to it
//                    for (int direction = 0; direction < 6; ++direction) {
//                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//                        if (apr_tree_iterator.find_neighbours_in_direction(direction)) {
//
//                            if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {
//
//                                val += (mean_tree[neighbour_tree_iterator]);
//                                counter++;
//                            }
//                        }
//                    }
//
//                    mean_tree_temp[apr_iterator] = (weight) * mean_tree[apr_iterator] + (val / counter) * (1 - weight);
//                }
//            }
//
//            std::swap(mean_tree.data,mean_tree_temp.data);
//        }
//        std::swap(mean_tree.data,mean_tree_temp.data);

       unsigned int level = apr_tree_iterator.level_max();

#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator, neighbour_tree_iterator, parent_iterator)
        for (parent_number = apr_tree_iterator.particles_level_begin(level);
             parent_number <
             apr_tree_iterator.particles_level_end(level); ++parent_number) {

            apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

            float temp = 0;
            float counter = 0;
            float counter_neigh = 0;

            float val = mean_tree[apr_tree_iterator];

            //loop over all the neighbours and set the neighbour iterator to it
            for (int direction = 0; direction < 6; ++direction) {
                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                if (apr_tree_iterator.find_neighbours_same_level(direction)) {

                    if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {

                        if (mean_tree[neighbour_tree_iterator] < val) {
                            counter++;
                        }
                        counter_neigh++;
                    }
                }
            }

            if (counter > 0) {
                //counter = 1.0;

                if (counter / counter_neigh == 1) {

                    parent_iterator.set_iterator_to_parent(apr_tree_iterator);
                    max_spread[apr_tree_iterator] = 2;

                    max_spread[parent_iterator] += mean_tree[apr_tree_iterator];
                    max_counter[parent_iterator]++;

                    max_spread_temp[parent_iterator] = max_spread[parent_iterator];
                    max_counter_temp[parent_iterator] = max_counter[parent_iterator];
                }
            }

        }



        //then do the rest of the tree where order matters
        for (unsigned int level = (apr_tree_iterator.level_max() - 1);
             level >= apr_tree_iterator.level_min(); --level) {

#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator,neighbour_tree_iterator,parent_iterator)
            //two loops first spread
            for (parent_number = apr_tree_iterator.particles_level_begin(level);
                 parent_number < apr_tree_iterator.particles_level_end(level); ++parent_number) {

                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

                //maybe spread first, then normalize, then push upwards..


                if (max_counter[apr_tree_iterator] > 0) {
                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        if (apr_tree_iterator.find_neighbours_same_level(direction)) {

                            if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {

                                max_spread_temp[neighbour_tree_iterator] += max_spread[apr_tree_iterator];
                                max_counter_temp[neighbour_tree_iterator] += max_counter[apr_tree_iterator];
                            }
                        }
                    }
                }
            }

#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator,neighbour_tree_iterator,parent_iterator)
            //then average and push up
            for (parent_number = apr_tree_iterator.particles_level_begin(level);
                 parent_number < apr_tree_iterator.particles_level_end(level); ++parent_number) {

                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);


                //maybe spread first, then normalize, then push upwards..

                if (max_counter_temp[apr_tree_iterator] > 0) {
                    max_spread[apr_tree_iterator] =
                            max_spread_temp[apr_tree_iterator] / (max_counter_temp[apr_tree_iterator] * 1.0f);
                    //tree_min[apr_tree_iterator] = tree_min[apr_tree_iterator]/(child_counter[apr_tree_iterator]*1.0f);
                    max_counter[apr_tree_iterator] = 1;
                } else {
                    max_spread[apr_tree_iterator] = 0;
                }

                if (level > apr_tree_iterator.level_min()) {
                    parent_iterator.set_iterator_to_parent(apr_tree_iterator);

                    if (max_spread[apr_tree_iterator] > 0) {
                        max_spread[parent_iterator] += max_spread[apr_tree_iterator];
                        max_counter[parent_iterator]++;

                        max_counter_temp[parent_iterator] = max_counter[parent_iterator];
                        max_spread_temp[parent_iterator] = max_spread[parent_iterator];
                    }
                }
            }
        }


#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,parent_iterator)
        //Now set the highest level particle cells.
        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max());
             particle_number <
             apr_iterator.particles_level_end(apr_iterator.level_max()); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            parent_iterator.set_iterator_to_parent(apr_iterator);
            boundary_type[apr_iterator] = max_spread[parent_iterator];

        }

        MeshData<uint16_t> boundary;
        apr.interp_img(boundary,boundary_type);
        std::string image_file_name = apr.parameters.input_dir +  "max_type.tif";
        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);

        adaptive_max.init(apr);


        for (unsigned int level = (apr_tree_iterator.level_max());
             level >= apr_tree_iterator.level_max(); --level) {
//#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator, parent_iterator)
            for (parent_number = apr_tree_iterator.particles_level_begin(level);
                 parent_number <
                 apr_tree_iterator.particles_level_end(level); ++parent_number) {

                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

                parent_iterator.set_iterator_to_parent(apr_tree_iterator);

                while ((parent_iterator.level() > parent_iterator.level_min()) && (max_spread[parent_iterator] == 0)) {
                    parent_iterator.set_iterator_to_parent(parent_iterator);
                }

                //float t = max_spread[parent_iterator];

                max_spread[apr_tree_iterator] = max_spread[parent_iterator];

            }
        }




        for (int i = 0; i < smooth_factor; ++i) {
            //smooth step
            for (unsigned int level = (apr_tree_iterator.level_max());
                 level >= apr_tree_iterator.level_min(); --level) {
#pragma omp parallel for schedule(static) private(parent_number) firstprivate(apr_tree_iterator,neighbour_tree_iterator)
                for (parent_number = apr_tree_iterator.particles_level_begin(level);
                     parent_number < apr_tree_iterator.particles_level_end(level); ++parent_number) {

                    apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

                    float temp = max_spread[apr_tree_iterator];
                    float counter = 1;

                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        if (apr_tree_iterator.find_neighbours_same_level(direction)) {

                            if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {

                                if (max_spread[neighbour_tree_iterator] > 0) {
                                    temp += max_spread[neighbour_tree_iterator];
                                    counter++;
                                }
                            }
                        }
                    }

                    max_spread_temp[apr_tree_iterator] = temp / counter;

                }
            }

            std::swap(max_spread_temp.data,max_spread.data);
        }


#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,parent_iterator)
        //Now set the highest level particle cells.
        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max());
             particle_number <
             apr_iterator.particles_level_end(apr_iterator.level_max()); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            parent_iterator.set_iterator_to_parent(apr_iterator);
            adaptive_max[apr_iterator] = max_spread[parent_iterator];

        }



        apr.interp_img(boundary,adaptive_max);
        image_file_name = apr.parameters.input_dir +  "max_seed.tif";
        TiffUtils::saveMeshAsTiffUint16(image_file_name, boundary);

        std::fill(boundary_type.data.begin(),boundary_type.data.end(),0);

        //spread solution
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,apr_tree_iterator,neighbour_tree_iterator)
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

                apr_tree_iterator.set_particle_cell_no_search(apr_iterator);

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    if (apr_tree_iterator.find_neighbours_same_level(direction)) {

                        if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {
                            temp += max_spread[neighbour_tree_iterator];
                            counter++;

                        }
                    }
                }
                if(counter>0) {
                    adaptive_max[apr_iterator] = temp/counter;
                    boundary_type[apr_iterator] = apr_iterator.level_max();
                }

            }
        }

        int maximum_iteration = 20;


        for (int level = (apr_iterator.level_max()-1); level >= apr_iterator.level_min() ; --level) {
            uint64_t empty_counter = 0;
            bool still_empty = true;
            while(still_empty && (empty_counter < maximum_iteration)) {
                empty_counter++;
                still_empty = false;
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,neigh_iterator) reduction(||:still_empty)
                for (particle_number = apr_iterator.particles_level_begin(level);
                     particle_number <
                     apr_iterator.particles_level_end(level); ++particle_number) {
                    //This step is required for all loops to set the iterator by the particle number
                    apr_iterator.set_iterator_to_particle_by_number(particle_number);

                    if (boundary_type[apr_iterator] == 0) {

                        float counter = 0;
                        float temp = 0;

                        //loop over all the neighbours and set the neighbour iterator to it
                        for (int direction = 0; direction < 6; ++direction) {
                            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                            if (apr_iterator.find_neighbours_in_direction(direction)) {

                                if (neigh_iterator.set_neighbour_iterator(apr_iterator, direction, 0)) {

                                    if(boundary_type[neigh_iterator]>=(level+1)) {
                                        counter++;
                                        temp += adaptive_max[neigh_iterator];
                                    }
                                }
                            }
                        }

                        if (counter > 0) {
                            adaptive_max[apr_iterator] = temp / counter;
                            boundary_type[apr_iterator] = level;
                        } else {
                            still_empty = true;
                        }
                    } else {
                        boundary_type[apr_iterator]=level+1;
                    }
                }
            }

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator,parent_iterator)
            for (particle_number = apr_iterator.particles_level_begin(level);
                 particle_number <
                 apr_iterator.particles_level_end(level); ++particle_number) {
                //This step is required for all loops to set the iterator by the particle number
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                if(boundary_type[apr_iterator] == 0){

                    parent_iterator.set_iterator_to_parent(apr_iterator);

                    while ((parent_iterator.level() > parent_iterator.level_min()) && (max_spread[parent_iterator] == 0)) {
                        parent_iterator.set_iterator_to_parent(parent_iterator);
                    }

                    adaptive_max[apr_iterator] = max_spread[parent_iterator];
                    boundary_type[apr_iterator] = (uint16_t)(level + 1);
                }

            }
        }



    }






};


#endif //LIBAPR_APRTREENUMERICS_HPP
