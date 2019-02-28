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
    APRTree(APR<ImageType> &apr) { initialize_apr_tree_sparse(apr); APROwn = &apr; }


    void init(APR<ImageType> &apr) { initialize_apr_tree_sparse(apr); APROwn = &apr;}

    inline uint64_t total_number_parent_cells() const { return tree_access.total_number_particles; }

    ExtraParticleData<ImageType> particles_ds_tree; //down-sampled tree intensities

    operator uint64_t() { return total_number_parent_cells(); }

    void copyTree(APRTree<ImageType>& copyTree){
        tree_access = copyTree.tree_access;
        APROwn = copyTree.APROwn;
    }

    APRAccess tree_access;

    APRTreeIterator tree_iterator() {
        return APRTreeIterator(APROwn->apr_access,tree_access);
    }

    template<typename S>
    void fill_tree_mean_downsample(ExtraParticleData<S>& input_particles){
        this->fill_tree_mean(*APROwn,*this,input_particles,particles_ds_tree); //down-sampled tree intensities
    }



    template<typename S>
    uint8_t& get_val(SparseParticleCellMap& map,S key){
        //if it doesn't exist its initialized to 0

        auto it = map.mesh.find(key);
        bool empty = false;
        if (it == map.mesh.end()){
            empty = true;
        }

        uint8_t& val = map.mesh[key];

        if(empty){
            val = 0;
        }

        return val;

    }


    void initialize_apr_tree_sparse(APR<ImageType>& apr) {

        APROwn = &apr;

        APRTimer timer(false);

        auto apr_iterator = apr.iterator();

        // --------------------------------------------------------------------
        // Init APR tree memory
        // --------------------------------------------------------------------

        // extend one extra level
        uint64_t l_max = apr.level_max() - 1;
        uint64_t l_min = apr.level_min() - 1;

        tree_access.l_min = l_min;
        tree_access.l_max = l_max;

        std::vector<std::vector<SparseParticleCellMap>> particle_cell_tree;


        std::vector<uint64_t> y_num;
        std::vector<uint64_t> x_num;
        std::vector<uint64_t> z_num;

        z_num.resize(l_max + 1);
        x_num.resize(l_max + 1);
        y_num.resize(l_max + 1);
        particle_cell_tree.resize(l_max + 1);

        timer.start_timer("tree - init sparse structure");
        for (unsigned int l = l_min; l < (l_max) ;l ++){

            x_num[l] = (uint64_t) ceil((1.0 * apr.apr_access.org_dims[1]) / pow(2.0, 1.0 * l_max - l + 1));
            z_num[l] = (uint64_t) ceil((1.0 * apr.apr_access.org_dims[2]) / pow(2.0, 1.0 * l_max - l + 1));
            particle_cell_tree[l].resize(z_num[l]*x_num[l]);
            y_num[l] = (uint64_t) ceil((1.0 * apr.apr_access.org_dims[0]) / pow(2.0, 1.0 * l_max - l + 1));

        }
        timer.stop_timer();


        // --------------------------------------------------------------------
        // Insert values to the APR tree
        // --------------------------------------------------------------------
        timer.start_timer("tree - insert vals sparse");

        //note the use of the dynamic OpenMP schedule.

        for (unsigned int level = (apr_iterator.level_max()); level >= apr_iterator.level_min(); --level) {
            int z_d = 0;
            int x_d = 0;
            int z =0;
            int x =0;
            if (level < (apr.level_max())) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
                for (z_d = 0; z_d < z_num[level-1]; z_d++) {
                    for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
                        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                        for (x_d = 0; x_d < x_num[level-1]; ++x_d) {
                            for (int x = 2 * x_d;
                                 x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level) - 1); ++x) {

                                size_t x_p = x / 2;
                                size_t z_p = z / 2;
                                int current_level = level - 1;

                                for (apr_iterator.set_new_lzx(level, z, x);
                                     apr_iterator.global_index() < apr_iterator.end_index;
                                     apr_iterator.set_iterator_to_particle_next_particle()) {
                                    auto &pct = particle_cell_tree[current_level][z_p * x_num[current_level] + x_p];

                                    size_t y_p = apr_iterator.y() / 2;

                                    pct.mesh[y_p] = 1;

                                }
                            }
                        }
                    }
                }

                if (level != (apr.level_max()-1)) {
                    //second loop here pushing them further up the tree, the max level is special, as always exists on next level so not required.
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
                    for (z_d = 0; z_d < z_num[level-1]; z_d++) {
                        for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level) - 1); ++z) {
                            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                            for (x_d = 0; x_d < x_num[level - 1]; ++x_d) {
                                for (int x = 2 * x_d;
                                     x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level) - 1); ++x) {

                                    auto &pct = particle_cell_tree[level][z * x_num[level] + x];

                                    size_t x_p = x / 2;
                                    size_t z_p = z / 2;
                                    int parent_level = level - 1;

                                    auto &pct_p = particle_cell_tree[parent_level][z_p * x_num[parent_level] + x_p];

                                    for (auto it = pct.mesh.begin(); it != pct.mesh.end(); ++it) {
                                        size_t y_p = it->first / 2;

                                        pct_p.mesh[y_p] = 1;

                                    }
                                }
                            }
                        }
                    }
                }

            }
            else {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
                for (z_d = 0; z_d < apr_iterator.spatial_index_z_max(level - 2); z_d++) {
                    for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr.spatial_index_z_max(level-1) - 1); ++z) {
                        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                        for (x_d = 0; x_d < apr_iterator.spatial_index_x_max(level-2); ++x_d) {
                            for (int x = 2 * x_d;
                                 x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level-1) - 1); ++x) {

                                int x_p = x_d;
                                int z_p = z_d;
                                int current_level = level - 2;

                                for (apr_iterator.set_new_lzx(level, 2 * z, 2 * x);
                                     apr_iterator.global_index() < apr_iterator.end_index;
                                     apr_iterator.set_iterator_to_particle_next_particle()) {

                                    auto &pct = particle_cell_tree[current_level][z_p * x_num[current_level] + x_p];

                                    if (apr_iterator.y() % 2 == 0) {
                                        size_t y_p = apr_iterator.y() / 4;

                                        pct.mesh[y_p] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        timer.stop_timer();






        tree_access.l_max = l_max;
        tree_access.l_min = l_min;

        tree_access.x_num.resize(l_max+1);
        tree_access.y_num.resize(l_max+1);
        tree_access.z_num.resize(l_max+1);

        for(int i = l_min;i <l_max;i++){
            tree_access.x_num[i] = x_num[i];
            tree_access.y_num[i] = y_num[i];
            tree_access.z_num[i] = z_num[i];
        }

        tree_access.x_num[l_max] = apr.apr_access.x_num[l_max];
        tree_access.y_num[l_max] = apr.apr_access.y_num[l_max];
        tree_access.z_num[l_max] = apr.apr_access.z_num[l_max];


        timer.start_timer("tree - init tree");
        tree_access.initialize_tree_access_sparse(apr.apr_access, particle_cell_tree);
        timer.stop_timer();

    }

protected:

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
            particle_cell_parent_tree[l].initWithValue(ceil(apr.orginal_dimensions(0) / cellSize),
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
        timer.verbose_flag = false;

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
                                            scale_factor_yxz * particle_data[apr_iterator] / 8.0f +
                                            tree_data[parentIterator];
                                } else {

                                    tree_data[parentIterator] =
                                            scale_factor_xz * particle_data[apr_iterator] / 8.0f +
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
