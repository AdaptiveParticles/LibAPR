//
// Created by cheesema on 15.02.18.
//

#ifndef LIBAPR_APRTREENUMERICS_HPP
#define LIBAPR_APRTREENUMERICS_HPP

#include "data_structures/APR/PartCellData.hpp"
#include "data_structures/APR/ParticleData.hpp"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "APRNumerics.hpp"

class APRTreeNumerics {

    template<typename S, typename U>
    static void fill_tree_mean_internal(APR &apr, ParticleData<S> &particle_data,ParticleData<U> &tree_data,PartCellData<S> &particle_data_pc,PartCellData<U> &tree_data_pc,const bool pc_data) {

        APRTimer timer;
        timer.verbose_flag = false;

        timer.start_timer("ds-init");
        tree_data.init(apr.total_number_parent_cells());

        std::fill(tree_data.data.begin(), tree_data.data.end(), 0);

        auto treeIterator = apr.tree_iterator();
        auto parentIterator = apr.tree_iterator();

        auto apr_iterator = apr.iterator();

//        auto treeIterator = apr.tree_iterator();
//        auto parentIterator = apr.tree_iterator();
//
//        auto apr_iterator = apr.iterator();

        int z_d;
        int x_d;
        timer.stop_timer();

        timer.start_timer("ds-1l");

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_iterator, parentIterator)
#endif
            for (z_d = 0; z_d < parentIterator.z_num(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr_iterator.z_num(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < parentIterator.x_num(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level)-1); ++x) {

                            parentIterator.begin(level - 1, z / 2, x / 2);

                            //dealing with boundary conditions
                            float scale_factor_xz =
                                    (((2 * parentIterator.x_num(level - 1) != apr_iterator.x_num(level)) &&
                                      ((x / 2) == (parentIterator.x_num(level - 1) - 1))) +
                                     ((2 * parentIterator.z_num(level - 1) != apr_iterator.z_num(level)) &&
                                      (z / 2) == (parentIterator.z_num(level - 1) - 1))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.y_num(level - 1) != apr_iterator.y_num(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }


                            for (apr_iterator.begin(level, z, x);
                                 apr_iterator <
                                 apr_iterator.end(); apr_iterator++) {

                                while ((parentIterator.y() != (apr_iterator.y() / 2)) && (parentIterator < parentIterator.end())) {
                                    parentIterator++;
                                }

                                U part_val;

                                if(pc_data){
                                    //part_val =  particle_data_pc[apr_iterator.get_pcd_key()]; #TODO update this
                                } else {
                                    part_val =  particle_data[apr_iterator];
                                }


                                if (parentIterator.y() == (parentIterator.y_num(level - 1) - 1)) {
                                    tree_data[parentIterator] =
                                            scale_factor_yxz * part_val / 8.0f +
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
        for (int level = treeIterator.level_max(); level > treeIterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d,z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < treeIterator.z_num(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.z_num(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < treeIterator.x_num(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.x_num(level)-1); ++x) {

                            parentIterator.begin(level - 1, z/2, x/2);

                            float scale_factor_xz =
                                    (((2 * parentIterator.x_num(level - 1) != parentIterator.x_num(level)) &&
                                      ((x / 2) == (parentIterator.x_num(level - 1) - 1))) +
                                     ((2 * parentIterator.z_num(level - 1) != parentIterator.z_num(level)) &&
                                      ((z / 2) == (parentIterator.z_num(level - 1) - 1)))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.y_num(level - 1) != parentIterator.y_num(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }

                            for (treeIterator.begin(level, z, x);
                                 treeIterator <
                                 treeIterator.end(); treeIterator++) {

                                while ((parentIterator.y() != treeIterator.y() / 2) && (parentIterator < parentIterator.end())) {
                                    parentIterator++;
                                }

                                if (parentIterator.y() == (parentIterator.y_num(level - 1) - 1)) {
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

    template<typename S, typename U>
    static void fill_tree_max(APR &apr, ParticleData<S> &particle_data,ParticleData<U> &tree_data,PartCellData<S> &particle_data_pc,PartCellData<U> &tree_data_pc,const bool pc_data) {

        APRTimer timer;
        timer.verbose_flag = false;

        timer.start_timer("ds-init");
        tree_data.init(apr.total_number_parent_cells());

        std::fill(tree_data.data.begin(), tree_data.data.end(), 0);

        auto treeIterator = apr.random_tree_iterator();
        auto parentIterator = apr.random_tree_iterator();

        auto apr_iterator = apr.random_iterator();

        int z_d;
        int x_d;
        timer.stop_timer();
        timer.start_timer("ds-1l");

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_iterator, parentIterator)
#endif
            for (z_d = 0; z_d < apr_iterator.z_num(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_iterator.z_num(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr_iterator.x_num(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);


                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator <
                                 apr_iterator.end(); apr_iterator.set_iterator_to_particle_next_particle()) {

                                while ((parentIterator.y() != (apr_iterator.y() / 2)) && (parentIterator < parentIterator.end())) {
                                    parentIterator++;
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
        for (int level = treeIterator.level_max(); level > treeIterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < apr_iterator.z_num(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_iterator.z_num(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr_iterator.x_num(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level)); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);


                            for (treeIterator.set_new_lzx(level, z, x);
                                 treeIterator <
                                 treeIterator.end(); treeIterator.set_iterator_to_particle_next_particle()) {

                                while ((parentIterator.y() != treeIterator.y() / 2) && (parentIterator < parentIterator.end())) {
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

    template<typename S, typename U>
    static void fill_tree_mean(APR &apr, PartCellData<S> &particle_data_pc,ParticleData<U> &tree_data) {

        ParticleData<S> particle_data_empty;
        PartCellData<U> tree_data_pc_empty;

        fill_tree_mean_internal(apr, particle_data_empty,tree_data,particle_data_pc,tree_data_pc_empty,true);

    }

    template<typename S, typename U>
    static void fill_tree_mean(APR &apr, ParticleData<S> &particle_data,ParticleData<U> &tree_data) {

        PartCellData<S> particle_data_pc_empty;
        PartCellData<U> tree_data_pc_empty;
        fill_tree_mean_internal(apr, particle_data,tree_data,particle_data_pc_empty,tree_data_pc_empty,false);

        //fill_tree_mean_(apr,particle_data,tree_data);
    }

    template< typename S, typename U>
    static void fill_tree_max(APR &apr, PartCellData<S> &particle_data_pc,ParticleData<U> &tree_data) {

        ParticleData<S> particle_data_empty;
        PartCellData<U> tree_data_pc_empty;
        fill_tree_max(apr, particle_data_empty,tree_data,particle_data_pc,tree_data_pc_empty,true);

    }

    template<typename S, typename U>
    static void fill_tree_max(APR &apr, ParticleData<S> &particle_data,ParticleData<U> &tree_data) {

        PartCellData<S> particle_data_pc_empty;
        PartCellData<U> tree_data_pc_empty;
        fill_tree_max(apr, particle_data,tree_data,particle_data_pc_empty,tree_data_pc_empty,false);

    }


    template<typename U>
    static void fill_tree_max_level(APR &apr,ParticleData<U> &tree_data) {

        APRTimer timer;
        timer.verbose_flag = false;

        timer.start_timer("ds-init");
        tree_data.init(apr.total_number_parent_cells());

        std::fill(tree_data.data.begin(), tree_data.data.end(), 0);

        auto treeIterator = apr.tree_iterator();
        auto parentIterator = apr.tree_iterator();

        auto apr_iterator = apr.iterator();

        int z_d;
        int x_d;
        timer.stop_timer();
        timer.start_timer("ds-1l");

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_iterator, parentIterator)
#endif
            for (z_d = 0; z_d < apr_iterator.z_num(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_iterator.z_num(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr_iterator.x_num(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level)); ++x) {

                            parentIterator.begin(level - 1, z / 2, x / 2);


                            for (apr_iterator.begin(level, z, x);
                                 apr_iterator <
                                 apr_iterator.end(); apr_iterator++) {

                                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                                    parentIterator++;
                                }

                                tree_data[parentIterator] = std::max((U)level,tree_data[parentIterator]);

                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();
        timer.start_timer("ds-2l");

        //then do the rest of the tree where order matters
        for (int level = treeIterator.level_max(); level > treeIterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < apr_iterator.z_num(level) / 2; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_iterator.z_num(level)); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr_iterator.x_num(level) / 2; ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level)); ++x) {

                            parentIterator.begin(level - 1, z / 2, x / 2);

                            for (treeIterator.begin(level, z, x);
                                 treeIterator <
                                 treeIterator.end(); treeIterator++) {

                                while (parentIterator.y() != treeIterator.y() / 2) {
                                    parentIterator++;
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