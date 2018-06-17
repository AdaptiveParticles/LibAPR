//
// Created by cheesema on 13.02.18.
//

#ifndef LIBAPR_APRTREE_HPP
#define LIBAPR_APRTREE_HPP

#include "APR.hpp"

#define INTERIOR_PARENT 9

template<typename imageType>
class APRTree {

    template<typename S> friend class APRIterator;
    template<typename S> friend class APRTreeIterator;

public:

    APRTree(){};

    APRTree(APR<imageType>& apr){
        initialize_apr_tree(apr);
    }

    void init(APR<imageType>& apr){
        initialize_apr_tree(apr);
    }

    inline uint64_t total_number_parent_cells() const { return (tree_access).total_number_particles; }

private:
    APRAccess tree_access;

    void initialize_apr_tree(APR<imageType>& apr,bool type_full = false){

        APRIterator<imageType> apr_iterator(apr);

        APRTimer timer;
        timer.verbose_flag = true;

        ///////////////////////
        ///
        /// Set the iterator using random access by particle cell spatial index (x,y,z) and particle cell level
        ///
        ////////////////////////
        std::vector<PixelData<uint8_t>> particle_cell_parent_tree;
        uint64_t l_max = apr.level_max()-1;
        uint64_t l_min = apr.level_min()-1; // extend one extra level

        tree_access.level_min = l_min;
        tree_access.level_max = l_max;


        timer.start_timer("tree - init structure");

        particle_cell_parent_tree.resize(l_max + 1);

        for (uint64_t l = l_min; l < (l_max + 1) ;l ++){

            particle_cell_parent_tree[l].init((int)ceil((1.0*apr.orginal_dimensions(0))/pow(2.0,1.0*l_max - l + 1)),

                                                    (int)ceil((1.0*apr.orginal_dimensions(1))/pow(2.0,1.0*l_max - l + 1)),
                                                    (int)ceil((1.0*apr.orginal_dimensions(2))/pow(2.0,1.0*l_max - l + 1)), (uint8_t)0);
        }

        timer.stop_timer();



        timer.start_timer("tree - insert vals");


        //note the use of the dynamic OpenMP schedule.
        for (unsigned int level = apr.level_max(); level >= apr_iterator.level_min(); --level) {

            int z = 0;
            int x = 0;

            if(level < apr.level_max()) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
                for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                    for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                        for (apr_iterator.set_new_lzx(level, z, x);
                             apr_iterator.global_index() < apr_iterator.particles_zx_end(level, z,
                                                                                         x); apr_iterator.set_iterator_to_particle_next_particle()) {

                            size_t y_p = apr_iterator.y() / 2;
                            size_t x_p = apr_iterator.x() / 2;
                            size_t z_p = apr_iterator.z() / 2;

                            int current_level = apr_iterator.level() - 1;


                            if (particle_cell_parent_tree[current_level](y_p, x_p, z_p) == INTERIOR_PARENT) {
                                particle_cell_parent_tree[current_level](y_p, x_p, z_p) = 1;
                            } else {
                                particle_cell_parent_tree[current_level](y_p, x_p, z_p)++;
                            }

                            while (current_level > l_min) {
                                current_level--;
                                y_p = y_p / 2;
                                x_p = x_p / 2;
                                z_p = z_p / 2;


                                if (particle_cell_parent_tree[current_level](y_p, x_p, z_p) == 0) {
                                    particle_cell_parent_tree[current_level](y_p, x_p, z_p) = INTERIOR_PARENT;

                                } else {
                                    //already covered
                                    break;
                                }
                            }

                        }
                    }
                }
            } else {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
                for (z = 0; z < apr_iterator.spatial_index_z_max(level-1); z++) {
                    for (x = 0; x < apr_iterator.spatial_index_x_max(level-1); ++x) {
                        for (apr_iterator.set_new_lzx(level, 2*z, 2*x);
                             apr_iterator.global_index() < apr_iterator.particles_zx_end(level, 2*z,
                                                                                         2*x); apr_iterator.set_iterator_to_particle_next_particle()) {

                            if(apr_iterator.y()%2 == 0) {
                                size_t y_p = apr_iterator.y() / 2;
                                size_t x_p = apr_iterator.x() / 2;
                                size_t z_p = apr_iterator.z() / 2;

                                int current_level = apr_iterator.level() - 1;


                                if (particle_cell_parent_tree[current_level](y_p, x_p, z_p) == INTERIOR_PARENT) {
                                    particle_cell_parent_tree[current_level](y_p, x_p, z_p) = 1;
                                } else {
                                    particle_cell_parent_tree[current_level](y_p, x_p, z_p)++;

                                }

                                while (current_level > l_min) {
                                    current_level--;
                                    y_p = y_p / 2;
                                    x_p = x_p / 2;
                                    z_p = z_p / 2;


                                    if (particle_cell_parent_tree[current_level](y_p, x_p, z_p) == 0) {
                                        particle_cell_parent_tree[current_level](y_p, x_p, z_p) = INTERIOR_PARENT;

                                    } else {
                                        //already covered
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();


        timer.start_timer("tree - init tree");
        this->tree_access.initialize_tree_access(apr,particle_cell_parent_tree);
        timer.stop_timer();

    }


};


#endif //LIBAPR_APRTREE_HPP
