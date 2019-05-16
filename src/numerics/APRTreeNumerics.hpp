//
// Created by cheesema on 15.02.18.
//

#ifndef LIBAPR_APRTREENUMERICS_HPP
#define LIBAPR_APRTREENUMERICS_HPP

#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "APRNumerics.hpp"

class APRTreeNumerics {

    template<typename T, typename S, typename U>
    static void fill_tree_mean_internal(APR<T> &apr, APRTree<T> &apr_tree, ExtraParticleData<S> &particle_data,ExtraParticleData<U> &tree_data,ExtraPartCellData<S> &particle_data_pc,ExtraPartCellData<U> &tree_data_pc,const bool pc_data) {

        APRTimer timer;
        timer.verbose_flag = false;

        timer.start_timer("ds-init");
        tree_data.init(apr_tree.total_number_parent_cells());

        std::fill(tree_data.data.begin(), tree_data.data.end(), 0);

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
            for (z_d = 0; z_d < apr.spatial_index_z_max(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr.spatial_index_x_max(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);

                            //dealing with boundary conditions
                            float scale_factor_xz =
                                    (((2 * apr.spatial_index_x_max(level - 1) != apr.spatial_index_x_max(level)) &&
                                      ((x / 2) == (apr.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * apr.spatial_index_z_max(level - 1) != apr.spatial_index_z_max(level)) &&
                                      (z / 2) == (apr.spatial_index_z_max(level - 1) - 1))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * apr.spatial_index_y_max(level - 1) != apr.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }


                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator.global_index() <
                                 apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                S part_val;

                                if(pc_data){
                                    part_val =  particle_data_pc[apr_iterator.get_pcd_key()];
                                } else {
                                    part_val =  particle_data[apr_iterator];
                                }

                                if (parentIterator.y() == (apr.spatial_index_y_max(level - 1) - 1)) {
                                    tree_data[parentIterator] =
                                            scale_factor_yxz * part_val/ 8.0f +
                                            tree_data[parentIterator];
                                } else {

                                    tree_data[parentIterator] =
                                            scale_factor_xz * part_val / 8.0f +
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
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < apr.spatial_index_z_max(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr.spatial_index_x_max(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);

                            float scale_factor_xz =
                                    (((2 * apr.spatial_index_x_max(level - 1) != apr.spatial_index_x_max(level)) &&
                                      ((x / 2) == (apr.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * apr.spatial_index_z_max(level - 1) != apr.spatial_index_z_max(level)) &&
                                      ((z / 2) == (apr.spatial_index_z_max(level - 1) - 1)))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * apr.spatial_index_y_max(level - 1) != apr.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }

                            for (treeIterator.set_new_lzx(level, z, x);
                                 treeIterator.global_index() <
                                 treeIterator.end_index; treeIterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != treeIterator.y() / 2) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                if (parentIterator.y() == (apr.spatial_index_y_max(level - 1) - 1)) {
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

    template<typename T, typename S, typename U>
    static void fill_tree_max(APR<T> &apr, APRTree<T> &apr_tree, ExtraParticleData<S> &particle_data,ExtraParticleData<U> &tree_data,ExtraPartCellData<S> &particle_data_pc,ExtraPartCellData<U> &tree_data_pc,const bool pc_data) {

        APRTimer timer;
        timer.verbose_flag = false;

        timer.start_timer("ds-init");
        tree_data.init(apr_tree.total_number_parent_cells());

        std::fill(tree_data.data.begin(), tree_data.data.end(), 0);

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
            for (z_d = 0; z_d < apr.spatial_index_z_max(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr.spatial_index_x_max(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);


                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator.global_index() <
                                 apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                S part_val;

                                if(pc_data){

                                    part_val =  particle_data_pc[apr_iterator.get_pcd_key()];
                                } else {
                                    part_val =  particle_data[apr_iterator];
                                }

                                tree_data[parentIterator] = std::max((U)part_val,tree_data[parentIterator]);

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
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < apr.spatial_index_z_max(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr.spatial_index_x_max(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);


                            for (treeIterator.set_new_lzx(level, z, x);
                                 treeIterator.global_index() <
                                 treeIterator.end_index; treeIterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != treeIterator.y() / 2) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                tree_data[parentIterator] =  std::max(tree_data[treeIterator],tree_data[parentIterator]);
                            }
                        }
                    }
                }
            }
        }
        timer.stop_timer();
    }



public:

    template<typename T, typename S, typename U>
    static void fill_tree_mean(APR<T> &apr, APRTree<T> &apr_tree, ExtraPartCellData<S> &particle_data_pc,ExtraParticleData<U> &tree_data) {

        ExtraParticleData<S> particle_data_empty;
        ExtraPartCellData<U> tree_data_pc_empty;
        fill_tree_mean_internal(apr, apr_tree, particle_data_empty,tree_data,particle_data_pc,tree_data_pc_empty,true);

    }

    template<typename T, typename S, typename U>
    static void fill_tree_mean(APR<T> &apr, APRTree<T> &apr_tree, ExtraParticleData<S> &particle_data,ExtraParticleData<U> &tree_data) {

        ExtraPartCellData<S> particle_data_pc_empty;
        ExtraPartCellData<U> tree_data_pc_empty;
        fill_tree_mean_internal(apr, apr_tree, particle_data,tree_data,particle_data_pc_empty,tree_data_pc_empty,false);

    }

    template<typename T, typename S, typename U>
    static void fill_tree_max(APR<T> &apr, APRTree<T> &apr_tree, ExtraPartCellData<S> &particle_data_pc,ExtraParticleData<U> &tree_data) {

        ExtraParticleData<S> particle_data_empty;
        ExtraPartCellData<U> tree_data_pc_empty;
        fill_tree_max(apr, apr_tree, particle_data_empty,tree_data,particle_data_pc,tree_data_pc_empty,true);

    }

    template<typename T, typename S, typename U>
    static void fill_tree_max(APR<T> &apr, APRTree<T> &apr_tree, ExtraParticleData<S> &particle_data,ExtraParticleData<U> &tree_data) {

        ExtraPartCellData<S> particle_data_pc_empty;
        ExtraPartCellData<U> tree_data_pc_empty;
        fill_tree_max(apr, apr_tree, particle_data,tree_data,particle_data_pc_empty,tree_data_pc_empty,false);

    }



    template<typename T, typename U>
    static void fill_tree_max_level(APR<T> &apr, APRTree<T> &apr_tree,ExtraParticleData<U> &tree_data) {

        APRTimer timer;
        timer.verbose_flag = false;

        timer.start_timer("ds-init");
        tree_data.init(apr_tree.total_number_parent_cells());

        std::fill(tree_data.data.begin(), tree_data.data.end(), 0);

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
            for (z_d = 0; z_d < apr.spatial_index_z_max(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr.spatial_index_x_max(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);


                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator.global_index() <
                                 apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                tree_data[parentIterator] = std::max((U)apr_iterator.level(),tree_data[parentIterator]);

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
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < apr.spatial_index_z_max(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr.spatial_index_x_max(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);

                            for (treeIterator.set_new_lzx(level, z, x);
                                 treeIterator.global_index() <
                                 treeIterator.end_index; treeIterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != treeIterator.y() / 2) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                tree_data[parentIterator] =  std::max(tree_data[treeIterator],tree_data[parentIterator]);
                            }
                        }
                    }
                }
            }
        }
        timer.stop_timer();
    }

};

#endif