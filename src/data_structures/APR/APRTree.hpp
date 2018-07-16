//
// Created by cheesema on 13.02.18.
//

#ifndef LIBAPR_APRTREE_HPP
#define LIBAPR_APRTREE_HPP

//#include "numerics/APRTreeNumerics.hpp"
#include "APRTreeIterator.hpp"
#include "APR.hpp"



#define INTERIOR_PARENT 9

class APRTreeIterator;
class APRIterator;

template<typename ImageType>
class APRTree {
    friend class APRIterator;
    friend class APRTreeIterator;
    friend class APRWriter;

public:

    APRTree() {};
    APRTree(APR<ImageType> &apr) { initialize_apr_tree(apr); APROwn = &apr; }


    void init(APR<ImageType> &apr) { initialize_apr_tree(apr); APROwn = &apr;}

    inline uint64_t total_number_parent_cells() const { return tree_access.total_number_particles; }

    ExtraParticleData<ImageType> particles_ds_tree; //down-sampled tree intensities

    operator uint64_t() { return total_number_parent_cells(); }


    APRAccess tree_access;

    APRTreeIterator tree_iterator() {
        return APRTreeIterator(APROwn->apr_access,tree_access);
    }

    template<typename S>
    void fill_tree_mean_downsample(ExtraParticleData<S>& input_particles){
        this->fill_tree_mean(*APROwn,*this,input_particles,particles_ds_tree); //down-sampled tree intensities
    }

private:

    APR<ImageType>* APROwn;

    void initialize_apr_tree(APR<ImageType>& apr, bool type_full = false) {

        APRTimer timer(true);

        auto apr_iterator = apr.iterator();

        // --------------------------------------------------------------------
        // Init APR tree memory
        // --------------------------------------------------------------------

        // extend one extra level
        uint64_t l_max = apr.level_max() - 1;
        uint64_t l_min = apr.level_min() - 1;

        tree_access.l_min = l_min;
        tree_access.l_max = l_max;

        std::vector<PixelData<uint8_t>> particle_cell_parent_tree(l_max);

        timer.start_timer("tree - init structure");
        for (uint64_t l = l_min; l < l_max; ++l) {
            double cellSize = pow(2.0, l_max - l + 1);
            particle_cell_parent_tree[l].init(ceil(apr.orginal_dimensions(0) / cellSize),
                                              ceil(apr.orginal_dimensions(1) / cellSize),
                                              ceil(apr.orginal_dimensions(2) / cellSize),
                                              0);
        }
        timer.stop_timer();


        // --------------------------------------------------------------------
        // Insert values to the APR tree
        // --------------------------------------------------------------------
        timer.start_timer("tree - insert vals");


        //note the use of the dynamic OpenMP schedule.

        for (unsigned int level = (apr_iterator.level_max()); level >= apr_iterator.level_min(); --level) {
            int z = 0;
            int x = 0;
            if (level < (apr.level_max())) {
                #ifdef HAVE_OPENMP
                #pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
                #endif
                for ( z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                    for ( x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                        for (apr_iterator.set_new_lzx(level, z, x);
                             apr_iterator.global_index() < apr_iterator.end_index;
                             apr_iterator.set_iterator_to_particle_next_particle()) {

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
            else {
                #ifdef HAVE_OPENMP
                #pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
                #endif
                for ( z = 0; z < apr_iterator.spatial_index_z_max(level-1); z++) {
                    for ( x = 0; x < apr_iterator.spatial_index_x_max(level-1); ++x) {
                        for (apr_iterator.set_new_lzx(level, 2*z, 2*x);
                             apr_iterator.global_index() < apr_iterator.end_index;
                             apr_iterator.set_iterator_to_particle_next_particle()) {

                            if (apr_iterator.y()%2 == 0) {
                                size_t y_p = apr_iterator.y() / 2;
                                size_t x_p = apr_iterator.x() / 2;
                                size_t z_p = apr_iterator.z() / 2;
                                int current_level = apr_iterator.level() - 1;

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
        tree_access.initialize_tree_access(apr.apr_access,particle_cell_parent_tree);
        timer.stop_timer();
    }
public:
    template<typename T,typename S,typename U>
    static void fill_tree_mean(APR<T>& apr,APRTree<T>& apr_tree,ExtraParticleData<S>& particle_data,ExtraParticleData<U>& tree_data) {

        APRTimer timer;
        timer.verbose_flag = true;

        timer.start_timer("ds-init");
        tree_data.init(apr_tree.total_number_parent_cells());


        APRTreeIterator treeIterator = apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr_tree.tree_iterator();

        APRIterator apr_iterator = apr.iterator();

        int z_d;
        int x_d;
        timer.stop_timer();
        timer.start_timer("ds-1l");

        for (unsigned int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_iterator, parentIterator)
#endif
            for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < parentIterator.spatial_index_x_max(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)-1); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);

                            //dealing with boundary conditions
                            float scale_factor_xz =
                                    (((2 * parentIterator.spatial_index_x_max(level - 1) != apr.spatial_index_x_max(level)) &&
                                      ((x / 2) == (parentIterator.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * parentIterator.spatial_index_z_max(level - 1) != apr.spatial_index_z_max(level)) &&
                                      (z / 2) == (parentIterator.spatial_index_z_max(level - 1) - 1))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.spatial_index_y_max(level - 1) != apr.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }


                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator.global_index() <
                                 apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }


                                if (parentIterator.y() == (parentIterator.spatial_index_y_max(level - 1) - 1)) {
                                    tree_data[parentIterator] =
                                            scale_factor_yxz * apr.particles_intensities[apr_iterator] / 8.0f +
                                            tree_data[parentIterator];
                                } else {

                                    tree_data[parentIterator] =
                                            scale_factor_xz * apr.particles_intensities[apr_iterator] / 8.0f +
                                            tree_data[parentIterator];
                                }

                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();
        timer.start_timer("ds-2l");

        //then do the rest of the tree where order matters
        for (unsigned int level = treeIterator.level_max(); level > treeIterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d,z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < treeIterator.spatial_index_x_max(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.spatial_index_x_max(level)-1); ++x) {

                            parentIterator.set_new_lzx(level - 1, z/2, x/2);

                            float scale_factor_xz =
                                    (((2 * parentIterator.spatial_index_x_max(level - 1) != parentIterator.spatial_index_x_max(level)) &&
                                      ((x / 2) == (parentIterator.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * parentIterator.spatial_index_z_max(level - 1) != parentIterator.spatial_index_z_max(level)) &&
                                      ((z / 2) == (parentIterator.spatial_index_z_max(level - 1) - 1)))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.spatial_index_y_max(level - 1) != parentIterator.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }

                            for (treeIterator.set_new_lzx(level, z, x);
                                 treeIterator.global_index() <
                                 treeIterator.end_index; treeIterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != treeIterator.y() / 2) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                if (parentIterator.y() == (parentIterator.spatial_index_y_max(level - 1) - 1)) {
                                    tree_data[parentIterator] = scale_factor_yxz * tree_data[treeIterator] / 8.0f +
                                                                tree_data[parentIterator];
                                } else {
                                    tree_data[parentIterator] = scale_factor_xz * tree_data[treeIterator] / 8.0f +
                                                                tree_data[parentIterator];
                                }

                            }
                        }
                    }
                }
            }
        }
        timer.stop_timer();
    }


};


#endif //LIBAPR_APRTREE_HPP
