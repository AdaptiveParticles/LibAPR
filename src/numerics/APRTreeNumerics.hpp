//
// Created by cheesema on 15.02.18.
//

#ifndef LIBAPR_APRTREENUMERICS_HPP
#define LIBAPR_APRTREENUMERICS_HPP

#include "src/data_structures/APR/APRTree.hpp"
#include "src/data_structures/APR/APRTreeIterator.hpp"

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
    static void calculate_adaptive_min(APR<T>& apr,APRTree<T>& apr_tree,ExtraParticleData<S>& adaptive_min){


        ExtraParticleData<float> mean_tree;
        APRTreeNumerics::fill_tree_from_particles(apr,apr_tree,apr.particles_intensities,mean_tree,[] (const float& a,const float& b) {return a+b;},true);

        APRTreeIterator<uint16_t> apr_tree_iterator(apr_tree);

        APRTreeIterator<uint16_t> neighbour_tree_iterator(apr_tree);
        APRIterator<uint16_t> apr_iterator(apr);

        ExtraParticleData<uint16_t> boundary_type(apr);

        //Basic serial iteration over all particles
        uint64_t particle_number;
        //Basic serial iteration over all particles
        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max()-1);
             particle_number <
             apr_iterator.particles_level_end(apr_iterator.level_max()-1); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            //now we only update the neighbours, and directly access them through a neighbour iterator

            float counter = 0;
            float temp = 0;

            if(apr_iterator.type() == 2) {

                apr_tree_iterator.set_particle_cell_no_search(apr_iterator);

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    if(apr_tree_iterator.find_neighbours_same_level(direction)) {

                        if (neighbour_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, 0)) {

                            float type = neighbour_tree_iterator.type();

                            if (neighbour_tree_iterator.type() == 8) {
                                // temp = std::max((float) mean_tree[neighbour_tree_iterator], temp);
                                temp += mean_tree[neighbour_tree_iterator];
                                counter++;
                            }

                        }
                    }
                }

                if(counter > 0) {
                    //counter = 1.0;
                    temp = temp / (counter * 1.0f);

                    if (apr.particles_intensities[apr_iterator] < temp) {
                        boundary_type[apr_iterator] = 1;
                    } else {
                        boundary_type[apr_iterator] = 2;
                    }
                }
            }
        }

        ExtraParticleData<uint64_t> child_counter(apr_tree);
        ExtraParticleData<uint64_t> child_counter_temp(apr_tree);
        ExtraParticleData<double> tree_min(apr_tree);
        ExtraParticleData<double> tree_min_temp(apr_tree);

        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max()-1);
             particle_number <
             apr_iterator.particles_level_end(apr_iterator.level_max()-1); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            if (boundary_type[apr_iterator] == 1) {

                if (apr_tree_iterator.set_iterator_to_parent(apr_iterator)) {

                    tree_min[apr_tree_iterator] += apr.particles_intensities[apr_iterator];
                    child_counter[apr_tree_iterator]++;
                    //child_counter_temp[apr_tree_iterator]=child_counter[apr_tree_iterator];
                    //tree_min_temp[apr_tree_iterator] = tree_min[apr_tree_iterator];

                }
            }
        }

        APRTreeIterator<uint16_t> parent_it(apr_tree);

        uint64_t parent_number;
        //then do the rest of the tree where order matters
        for (unsigned int level = (apr_tree_iterator.level_max()-1); level > apr_tree_iterator.level_min(); --level) {

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

                                //tree_min_temp[neighbour_tree_iterator]+=tree_min[apr_tree_iterator];
                                //child_counter_temp[neighbour_tree_iterator]+=child_counter[apr_tree_iterator];
                            }
                        }
                    }
                }
            }


            //then average and push up
            for (parent_number = apr_tree_iterator.particles_level_begin(level);
                 parent_number < apr_tree_iterator.particles_level_end(level); ++parent_number) {

                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);


                //maybe spread first, then normalize, then push upwards..

                if(child_counter[apr_tree_iterator]>1){
                    //tree_min[apr_tree_iterator] = tree_min_temp[apr_tree_iterator]/(child_counter_temp[apr_tree_iterator]*1.0f);
                    tree_min[apr_tree_iterator] = tree_min[apr_tree_iterator]/(child_counter[apr_tree_iterator]*1.0f);
                    child_counter[apr_tree_iterator]=1;
                } else {
                    tree_min[apr_tree_iterator] = 0;
                }

                parent_it.set_iterator_to_parent(apr_tree_iterator);


                if(apr_tree_iterator.level() != level){
                    int stop = 1;
                }


                if(tree_min[apr_tree_iterator] > 0){
                    tree_min[parent_it]+=tree_min[apr_tree_iterator];
                    child_counter[parent_it]++;

                    //child_counter_temp[parent_it]=child_counter[parent_it];
                    //tree_min_temp[parent_it] = tree_min[parent_it];
                }

            }
        }


//        //then do the rest of the tree where order matters
//        for (unsigned int level = apr_tree_iterator.level_max(); level >= apr_tree_iterator.level_min(); --level) {
//            for (parent_number = apr_tree_iterator.particles_level_begin(level);
//                 parent_number < apr_tree_iterator.particles_level_end(level); ++parent_number) {
//
//                apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);
//
//                if(child_counter[apr_tree_iterator]>1){
//                    tree_min[apr_tree_iterator] = tree_min[apr_tree_iterator]/(child_counter[apr_tree_iterator]*1.0f);
//                } else {
//                    tree_min[apr_tree_iterator] = 0;
//                }
//            }
//        }


        for (parent_number = apr_tree_iterator.particles_level_begin(apr_tree_iterator.level_max());
             parent_number < apr_tree_iterator.particles_level_end(apr_tree_iterator.level_max()); ++parent_number) {

            apr_tree_iterator.set_iterator_to_particle_by_number(parent_number);

            parent_it.set_iterator_to_parent(apr_tree_iterator);

            while((parent_it.level() > parent_it.level_min()) && (tree_min[parent_it] == 0)){
                parent_it.set_iterator_to_parent(parent_it);
            }

            tree_min[apr_tree_iterator] = tree_min[parent_it];

        }

        adaptive_min.init(apr);

        //Now set the highest level particle cells.
        for (particle_number = apr_iterator.particles_level_begin(apr_iterator.level_max());
                 particle_number <
                 apr_iterator.particles_level_end(apr_iterator.level_max()); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            parent_it.set_iterator_to_parent(apr_iterator);
            adaptive_min[apr_iterator] = tree_min[parent_it];

        }


    }

    void calculate_adaptive_max(){





    }





};


#endif //LIBAPR_APRTREENUMERICS_HPP
