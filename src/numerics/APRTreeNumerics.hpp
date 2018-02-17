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
    static void fill_tree_from_particles(APR<T>& apr,APRTree<T>& apr_tree,ExtraParticleData<S>& particle_data,ExtraParticleData<U>& tree_data,BinaryOperation op) {

        tree_data.init_tree(apr_tree);

        std::fill(tree_data.data.begin(),tree_data.data.end(),0);

        APRTreeIterator<T> treeIterator(apr_tree);
        APRTreeIterator<T> parentIterator(apr_tree);

        APRIterator<T> apr_iterator(apr);

        uint64_t particle_number = 0;
        uint64_t parent_number = 0;

        for (particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            //set parent
            parentIterator.set_iterator_to_parent(apr_iterator);

            tree_data[parentIterator] = std::max((U) particle_data[apr_iterator], (U) tree_data[parentIterator]);

        }


        //then do the rest of the tree where order matters
        for (unsigned int level = treeIterator.level_max(); level >= treeIterator.level_min(); --level) {
            for (parent_number = treeIterator.particles_level_begin(level);
                 parent_number < treeIterator.particles_level_end(level); ++parent_number) {

                treeIterator.set_iterator_to_particle_by_number(parent_number);

                if(parentIterator.set_iterator_to_parent(treeIterator)) {

                    tree_data[parentIterator] = std::max((U) tree_data[treeIterator], (U) tree_data[parentIterator]);
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


};


#endif //LIBAPR_APRTREENUMERICS_HPP
