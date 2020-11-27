//
// Created by cheesema on 15.02.18.
//

#ifndef LIBAPR_APRTREENUMERICS_HPP
#define LIBAPR_APRTREENUMERICS_HPP

#include "data_structures/APR/particles/PartCellData.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/iterators/APRTreeIterator.hpp"

namespace APRTreeNumerics {

    /**
     * Computes the value of each interior (tree) node as the average of its child nodes
     */
    template<typename PartDataType,typename PartDataTypeTree>
    void fill_tree_mean(APR &apr, const PartDataType& particle_data, PartDataTypeTree& tree_data);


    template< typename PartDataType,typename TreeDataType, typename BinaryOp>
    void fill_tree_op(APR& apr, const ParticleData<PartDataType>& particle_data, ParticleData<TreeDataType>& tree_data,
                      BinaryOp op, TreeDataType init_val = 0);

    /**
     * Computes the value of each interior (tree) node as the maximum of its child nodes
     */
    template<typename S, typename U>
    void fill_tree_max(APR& apr, const ParticleData<S>& particle_data, ParticleData<U>& tree_data) {
        fill_tree_op(apr, particle_data, tree_data,
                     [](const U& a, const U& b){ return std::max(a, b); },
                     std::numeric_limits<U>::min());
    }

    /**
     * Computes the value of each interior (tree) node as the minimum of its child nodes
     */
    template<typename S, typename U>
    void fill_tree_min(APR& apr, const ParticleData<S>& particle_data, ParticleData<U>& tree_data) {
        fill_tree_op(apr, particle_data, tree_data,
                     [](const U& a, const U& b){ return std::min(a, b); },
                     std::numeric_limits<U>::max());
    }


    template<typename S,typename U>
    void face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                                                const std::vector<float>& filter, int dimension, int ignore_levels=0);

    template<typename S,typename U>
    void seperable_face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                                         const std::vector<float>& filter, int repeats = 1, int ignore_levels=0);

    template<typename T>
    void push_down_tree(APR& apr, ParticleData<T>& tree_data, int num_levels=2);

    template<typename T, typename U>
    void push_to_leaves(APR& apr, const ParticleData<T>& tree_data, ParticleData<U>& part_data);
}


template<typename PartDataType,typename PartDataTypeTree>
void APRTreeNumerics::fill_tree_mean(APR &apr, const PartDataType &particle_data, PartDataTypeTree &tree_data) {

    APRTimer timer;
    timer.verbose_flag = false;

    timer.start_timer("ds-init");
    tree_data.init_tree(apr);

    tree_data.set_to_zero(); //works on the assumption of zero intiializtion of the tree particles

    auto tree_it = apr.tree_iterator();
    auto parent_it = apr.tree_iterator();
    auto apr_it = apr.iterator();

    int z_d;
    int x_d;

    timer.stop_timer();

    timer.start_timer("ds-1l");

    // reduce apr particles into tree nodes
    for (int level = apr_it.level_max(); level >= apr_it.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_it, parent_it)
#endif
        for (z_d = 0; z_d < parent_it.z_num(level - 1); z_d++) {
            for (int z = 2*z_d; z <= std::min(2*z_d+1, (int)apr_it.z_num(level) - 1); ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                for (x_d = 0; x_d < parent_it.x_num(level - 1); ++x_d) {
                    for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_it.x_num(level) - 1); ++x) {

                        parent_it.begin(level - 1, z / 2, x / 2);

                        //dealing with boundary conditions
                        float scale_factor_xz =
                                (((2 * parent_it.x_num(level - 1) != apr_it.x_num(level)) &&
                                  ((x / 2) == (parent_it.x_num(level - 1) - 1))) +
                                 ((2 * parent_it.z_num(level - 1) != apr_it.z_num(level)) &&
                                  (z / 2) == (parent_it.z_num(level - 1) - 1))) * 2;

                        if (scale_factor_xz == 0) {
                            scale_factor_xz = 1;
                        }

                        float scale_factor_yxz = scale_factor_xz;

                        if ((2 * parent_it.y_num(level - 1) != apr_it.y_num(level))) {
                            scale_factor_yxz = scale_factor_xz * 2;
                        }


                        for (apr_it.begin(level, z, x);
                             apr_it <
                             apr_it.end(); apr_it++) {

                            while ((parent_it.y() != (apr_it.y() / 2)) && (parent_it < parent_it.end())) {
                                parent_it++;
                            }


                            auto part_val =  particle_data[apr_it];


                            if (parent_it.y() == (parent_it.y_num(level - 1) - 1)) {
                                tree_data[parent_it] =
                                        scale_factor_yxz * part_val / 8.0f +
                                        tree_data[parent_it];
                            } else {

                                tree_data[parent_it] =
                                        scale_factor_xz * part_val / 8.0f +
                                        tree_data[parent_it];
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
    for (int level = tree_it.level_max(); level > tree_it.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d,z_d) firstprivate(tree_it, parent_it)
#endif
        for (z_d = 0; z_d < tree_it.z_num(level - 1); z_d++) {
            for (int z = 2*z_d; z <= std::min(2*z_d+1, (int)tree_it.z_num(level) - 1); ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                for (x_d = 0; x_d < tree_it.x_num(level - 1); ++x_d) {
                    for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_it.x_num(level) - 1); ++x) {

                        parent_it.begin(level - 1, z / 2, x / 2);

                        float scale_factor_xz =
                                (((2 * parent_it.x_num(level - 1) != parent_it.x_num(level)) && // is this ever false?
                                  ((x / 2) == (parent_it.x_num(level - 1) - 1))) +
                                 ((2 * parent_it.z_num(level - 1) != parent_it.z_num(level)) &&
                                  ((z / 2) == (parent_it.z_num(level - 1) - 1)))) * 2;

                        if (scale_factor_xz == 0) {
                            scale_factor_xz = 1;
                        }

                        float scale_factor_yxz = scale_factor_xz;

                        if ((2 * parent_it.y_num(level - 1) != parent_it.y_num(level))) {
                            scale_factor_yxz = scale_factor_xz * 2;
                        }

                        for (tree_it.begin(level, z, x);
                             tree_it <
                             tree_it.end(); tree_it++) {

                            while ((parent_it.y() != tree_it.y() / 2) && (parent_it < parent_it.end())) {
                                parent_it++;
                            }

                            if (parent_it.y() == (parent_it.y_num(level - 1) - 1)) {
                                tree_data[parent_it] = scale_factor_yxz * tree_data[tree_it] / 8.0f +
                                                       tree_data[parent_it];
                            } else {
                                tree_data[parent_it] = scale_factor_xz * tree_data[tree_it] / 8.0f +
                                                       tree_data[parent_it];
                            }

                        }
                    }
                }
            }
        }
    }
    timer.stop_timer();
}


template< typename PartDataType,typename TreeDataType, typename BinaryOp>
void APRTreeNumerics::fill_tree_op(APR &apr,
                                   const ParticleData<PartDataType> &particle_data,
                                   ParticleData<TreeDataType> &tree_data,
                                   BinaryOp op,
                                   TreeDataType init_val) {

    tree_data.init(apr.total_number_tree_particles());
    tree_data.fill(init_val);

    APRTimer timer;
    timer.verbose_flag = false;

    timer.start_timer("ds-init");

    auto tree_it = apr.tree_iterator();
    auto parent_it = apr.tree_iterator();
    auto apr_it = apr.iterator();

    tree_data.init(tree_it.total_number_particles());
    tree_data.fill(init_val);

    timer.stop_timer();
    timer.start_timer("ds-1l");

    // reduce apr particles into tree nodes
    for (int level = apr_it.level_max(); level >= apr_it.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, parent_it)
#endif
        for (int z_d = 0; z_d < apr_it.z_num(level - 1); z_d++) {
            for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_it.z_num(level) - 1); ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                for (int x_d = 0; x_d < apr_it.x_num(level - 1); ++x_d) {
                    for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_it.x_num(level) - 1); ++x) {

                        parent_it.begin(level - 1, z / 2, x / 2);

                        for (apr_it.begin(level, z, x); apr_it < apr_it.end(); apr_it++) {

                            while ((parent_it.y() != (apr_it.y() / 2)) && (parent_it < parent_it.end())) {
                                parent_it++;
                            }
                            tree_data[parent_it] = op((TreeDataType)particle_data[apr_it], tree_data[parent_it]);
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();
    timer.start_timer("ds-2l");

    // then do the rest of the tree where order matters
    for (int level = tree_it.level_max(); level > tree_it.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(tree_it, parent_it)
#endif
        for (int z_d = 0; z_d < tree_it.z_num(level - 1); z_d++) {
            for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) tree_it.z_num(level) - 1); ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                for (int x_d = 0; x_d < tree_it.x_num(level - 1); ++x_d) {
                    for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_it.x_num(level) - 1); ++x) {

                        parent_it.begin(level - 1, z / 2, x / 2);

                        for (tree_it.begin(level, z, x); tree_it < tree_it.end(); tree_it++) {

                            while ((parent_it.y() != tree_it.y() / 2) && (parent_it != parent_it.end())) {
                                parent_it++;
                            }
                            tree_data[parent_it] =  op(tree_data[tree_it], tree_data[parent_it]);
                        }
                    }
                }
            }
        }
    }
    timer.stop_timer();
}


template<typename S,typename U>
void APRTreeNumerics::face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                                            const std::vector<float>& filter, const int dimension, const int ignore_levels) {

    output_data.init(apr.total_number_tree_particles());

    if(apr.org_dims(dimension) <= 1) {
        output_data.copy(input_data);
        return;
    }

    int faces[2] = {2*dimension, 2*dimension+1};

    auto tree_iterator = apr.random_tree_iterator();
    auto neighbour_iterator = apr.random_tree_iterator();

    const int level_max = tree_iterator.level_max() - ignore_levels;
    const std::vector<float> filter_t = {filter[2], filter[0]};

    for (int level = tree_iterator.level_min(); level <= level_max; ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(tree_iterator, neighbour_iterator)
#endif
        for (int z = 0; z < tree_iterator.z_num(level); z++) {
            for (int x = 0; x < tree_iterator.x_num(level); ++x) {
                for (tree_iterator.begin(level, z, x); tree_iterator < tree_iterator.end(); tree_iterator++) {

                    float current_intensity = input_data[tree_iterator];
                    output_data[tree_iterator] = current_intensity * filter[1];

                    for (int i = 0; i < 2; ++i) {

                        const int direction = faces[i];
                        tree_iterator.find_neighbours_same_level(direction);

                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        if (neighbour_iterator.set_neighbour_iterator(tree_iterator, direction, 0)) {
                            output_data[tree_iterator] += filter_t[i] * input_data[neighbour_iterator];
                        } else {
                            output_data[tree_iterator] += filter_t[i] * current_intensity;
                        }
                    }
                }
            }
        }
    }
}


template<typename S,typename U>
void APRTreeNumerics::seperable_face_neighbour_filter(APR &apr, const ParticleData<S>& input_data, ParticleData<U>& output_data,
                                                      const std::vector<float>& filter, const int repeats, const int ignore_levels) {

    output_data.init(apr.total_number_tree_particles());

    ParticleData<U> tmp(apr.total_number_tree_particles());
    tmp.data.copy(input_data.data);

    for(int i = 0; i < repeats; ++i) {
        face_neighbour_filter(apr, tmp, output_data, filter, 0, ignore_levels);
        face_neighbour_filter(apr, output_data, tmp, filter, 1, ignore_levels);
        face_neighbour_filter(apr, tmp, output_data, filter, 2, ignore_levels);
        output_data.swap(tmp);
    }
    output_data.swap(tmp);
}


template<typename T>
void APRTreeNumerics::push_down_tree(APR &apr, ParticleData<T> &tree_data, const int num_levels) {
    auto it = apr.tree_iterator();
    auto parent_it = apr.tree_iterator();

    if(num_levels <= 0) {
        return;
    }

    int level_start = std::max(it.level_max()-num_levels+1, it.level_min()+1);

    for(int level = level_start; level <= it.level_max(); ++level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it, parent_it)
#endif
        for (int z = 0; z < it.z_num(level); ++z) {
            for (int x = 0; x < it.x_num(level); ++x) {

                parent_it.begin(level - 1, z / 2, x / 2);

                for (it.begin(level, z, x); it < it.end(); it++) {

                    while (parent_it.y() != it.y() / 2) {
                        parent_it++;
                    }
                    tree_data[it] = tree_data[parent_it];
                }
            }
        }
    }
}


template<typename T, typename U>
void APRTreeNumerics::push_to_leaves(APR& apr, const ParticleData<T>& tree_data, ParticleData<U>& part_data) {

    part_data.init(apr.total_number_particles());

    auto it = apr.iterator();
    auto parent_it = apr.tree_iterator();

    for(int level = it.level_max(); level >= it.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it, parent_it)
#endif
        for (int z = 0; z < it.z_num(level); ++z) {
            for (int x = 0; x < it.x_num(level); ++x) {

                parent_it.begin(level - 1, z / 2, x / 2);

                for (it.begin(level, z, x); it < it.end(); ++it) {

                    while(parent_it.y() < it.y()/2) {
                        parent_it++;
                    }
                    part_data[it] = tree_data[parent_it];
                }
            }
        }
    }
}


//template<typename T, typename S>
//static void compute_adaptive_max(APR& apr, ParticleData<T>& parts, ParticleData<S>& adaptive_max, const int level_offset) {
//
//    auto apr_it = apr.random_iterator();
//    auto tree_it = apr.random_tree_iterator();
//    auto neigh_it = apr.random_iterator();
//    auto neigh_it_tree = apr.random_tree_iterator();
//    auto parent_it = apr.random_tree_iterator();
//
//    ParticleData<float> tree_max;
//    ParticleData<float> max_spread;
//
//    fill_tree_max(apr, parts, tree_max);
//
//    max_spread.init(tree_max.size());
//    max_spread.set_to_zero();
//
//    std::cout << " copy max level tree " << std::endl;
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(static)
//#endif
//    for(size_t i = tree_it.particles_level_begin(tree_it.level_max()); i < tree_it.particles_level_end(tree_it.level_max()); ++i) {
//        max_spread[i] = tree_max[i];
//    }
//
//    std::cout << " first loop " << std::endl;
//    for (int level = tree_it.level_max(); level > tree_it.level_min(); --level) {
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(tree_it, neigh_it_tree, parent_it)
//#endif
//        for (int z_d = 0; z_d < tree_it.z_num(level - 1); z_d++) {
//            for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) tree_it.z_num(level) - 1); ++z) {
//                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
//                for (int x_d = 0; x_d < tree_it.x_num(level - 1); ++x_d) {
//                    for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_it.x_num(level) - 1); ++x) {
//
//                        parent_it.begin(level - 1, z / 2, x / 2);
//
//                        for (tree_it.begin(level, z, x); tree_it < tree_it.end(); tree_it++) {
//
//                            while (parent_it.y() < tree_it.y() / 2 && parent_it < parent_it.end()) {
//                                parent_it++;
//                            }
//
//                            float loc_sum = max_spread[tree_it];
//                            int counter = 1;
//
//                            if(loc_sum > 0) {
//                                //loop over all the neighbours and set the neighbour iterator to it
//                                for (int direction = 0; direction < 6; ++direction) {
//                                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                                    if (tree_it.find_neighbours_same_level(direction)) {
//                                        if (neigh_it_tree.set_neighbour_iterator(tree_it, direction, 0)) {
//                                            loc_sum += max_spread[neigh_it];
//                                            counter++;
//                                        }
//                                    }
//                                }
//                            }
//                            max_spread[parent_it] = std::max(max_spread[parent_it], loc_sum / counter);
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    if(level_offset > 0) {
//        std::cout << " second loop " << std::endl;
//        for(int level = tree_it.level_max()-level_offset; level <= tree_it.level_max(); ++level) {
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(tree_it, parent_it)
//#endif
//            for (int z_d = 0; z_d < tree_it.z_num(level - 1); z_d++) { // push values down one level
//                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) tree_it.z_num(level) - 1); ++z) {
//                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
//                    for (int x_d = 0; x_d < tree_it.x_num(level - 1); ++x_d) {
//                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_it.x_num(level) - 1); ++x) {
//
//                            parent_it.begin(level - 1, z / 2, x / 2);
//
//                            for (tree_it.begin(level, z, x); tree_it < tree_it.end(); tree_it++) {
//
//                                while (parent_it.y() < tree_it.y() / 2 && parent_it < parent_it.end()) {
//                                    parent_it++;
//                                }
//                                max_spread[tree_it] = max_spread[parent_it];
//                            }
//                        }
//                    }
//                }
//            }
//            // smooth tree up to level
////                smooth_tree(apr, max_spread, tree_it.level_max()-level, tree_it.level_max()-level);
//        }
//    }
//
//    adaptive_max.init(apr.total_number_particles());
//    adaptive_max.set_to_zero();
//
//    std::cout << " third loop " << std::endl;
//    // push level_max tree particles to level_max apr particles
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(tree_it, parent_it)
//#endif
//    for (int z_d = 0; z_d < apr_it.z_num(apr_it.level_max() - 1); z_d++) { // push values down one level
//        for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_it.z_num(apr_it.level_max()) - 1); ++z) {
//            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
//            for (int x_d = 0; x_d < apr_it.x_num(apr_it.level_max() - 1); ++x_d) {
//                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_it.x_num(apr_it.level_max()) - 1); ++x) {
//
//                    parent_it.begin(apr_it.level_max() - 1, z / 2, x / 2);
//
//                    for (apr_it.begin(apr_it.level_max(), z, x); apr_it < apr_it.end(); apr_it++) {
//
//                        while (parent_it.y() < apr_it.y() / 2 && parent_it < parent_it.end()) {
//                            parent_it++;
//                        }
//                        adaptive_max[apr_it] = max_spread[parent_it];
//                    }
//                }
//            }
//        }
//    }
//
//    ParticleData<uint16_t> boundary_type(apr.total_number_particles());
//    boundary_type.set_to_zero();
//
//    std::cout << " fourth loop " << std::endl;
//    // push level_max-1 tree particles to apr particles
//    int level = apr_it.level_max()-1;
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, parent_it, neigh_it_tree)
//#endif
//    for (int z_d = 0; z_d < apr_it.z_num(level - 1); z_d++) { // push values down one level
//        for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_it.z_num(level) - 1); ++z) {
//            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
//            for (int x_d = 0; x_d < apr_it.x_num(level - 1); ++x_d) {
//                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr_it.x_num(level) - 1); ++x) {
//
//                    parent_it.begin(level - 1, z / 2, x / 2);
//
//                    for (apr_it.begin(level, z, x); apr_it < apr_it.end(); apr_it++) {
//
//                        while (parent_it.y() < apr_it.y() / 2 && parent_it < parent_it.end()) {
//                            parent_it++;
//                        }
//
//                        float val = 0;
//                        int counter = 0;
//
//                        //loop over all the neighbours and set the neighbour iterator to it
//                        for (int direction = 0; direction < 6; ++direction) {
//                            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                            if (parent_it.find_neighbours_same_level(direction)) {
//                                if (neigh_it_tree.set_neighbour_iterator(parent_it, direction, 0)) {
//                                    val += max_spread[neigh_it];
//                                    counter++;
//                                }
//                            }
//                        }
//
//                        if(counter > 0) {
//                            adaptive_max[apr_it] = val / counter;
//                            boundary_type[apr_it] = level+1;
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    int maximum_iteration = 20;
//
//    for (level = (apr_it.level_max()-1); level >= apr_it.level_min() ; --level) {
//        uint64_t empty_counter = 0;
//        bool still_empty = true;
//
//        std::cout << " while loop " << std::endl;
//        while(still_empty && (empty_counter < maximum_iteration)) {
//            empty_counter++;
//            still_empty = false;
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it) reduction(||:still_empty)
//#endif
//            for(int z = 0; z < apr_it.z_num(level); ++z) {
//                for(int x = 0; x < apr_it.x_num(level); ++x) {
//                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
//
//                        if(boundary_type[apr_it] == 0) {
//
//                            int counter = 0;
//                            float temp = 0;
//
//                            //loop over all the neighbours and set the neighbour iterator to it
//                            for (int direction = 0; direction < 6; ++direction) {
//                                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//                                if (apr_it.find_neighbours_in_direction(direction)) {
//
//                                    if (neigh_it.set_neighbour_iterator(apr_it, direction, 0)) {
//
//                                        if (boundary_type[neigh_it] >= (level + 1)) {
//                                            counter++;
//                                            temp += adaptive_max[neigh_it];
//                                        }
//                                    }
//                                }
//                            }
//
//                            if (counter > 0) {
//                                adaptive_max[apr_it] = temp / counter;
//                                boundary_type[apr_it] = level;
//                            } else {
//                                still_empty = true;
//                            }
//                        } else {
//                            boundary_type[apr_it] = level+1;
//                        }
//                    }
//                }
//            }
//
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it) reduction(||:still_empty)
//#endif
//            for(int z = 0; z < apr_it.z_num(level); ++z) {
//                for(int x = 0; x < apr_it.x_num(level); ++x) {
//                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
//
//                        if(boundary_type[apr_it] == level) {
//
//                            int counter = 1;
//                            float temp = adaptive_max[apr_it];
//
//                            //loop over all the neighbours and set the neighbour iterator to it
//                            for (int direction = 0; direction < 6; ++direction) {
//                                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//                                if (apr_it.find_neighbours_in_direction(direction)) {
//
//                                    if (neigh_it.set_neighbour_iterator(apr_it, direction, 0)) {
//
//                                        if (boundary_type[neigh_it] >= (level + 1)) {
//                                            counter++;
//                                            temp += adaptive_max[neigh_it];
//                                        }
//                                    }
//                                }
//                            }
//                            adaptive_max[apr_it] = temp / counter;
//                        }
//                    }
//                }
//            }
//        } // end
//        std::cout << "level " << level << " empty counter = " << empty_counter << std::endl;
//
//#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(apr_it)
//#endif
//        for(int z = 0; z < apr_it.z_num(level); ++z) {
//            for (int x = 0; x < apr_it.x_num(level); ++x) {
//                for (apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
//
//                    if(boundary_type[apr_it] == 0) {
//                        adaptive_max[apr_it] = 1337; //parts[apr_it];
//                    }
//                }
//            }
//        }
//    }//end for level
//
//}

#endif