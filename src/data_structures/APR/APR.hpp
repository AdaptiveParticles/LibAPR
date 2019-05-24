//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "APRAccess.hpp"
#include "APRIterator.hpp"
#include "APRTree.hpp"
#include "APRTreeIterator.hpp"
#include "LinearIterator.hpp"

class APR {
    friend class APRFile;

protected:

    bool linear_or_random = false;
    bool linear_or_random_tree = false;

    APRAccess tree_access;
    bool tree_initialized = false;

    //APR Tree function
    void initialize_apr_tree_sparse();
    void initialize_apr_tree();
    void initialize_linear_access(APRAccess& aprAccess,GenIterator& it);

public:

    APRParameters parameters; // #TODO move to protected. Introduce a get and set method
    APRAccess apr_access; // #TODO move to protected.

    std::string name;
    //APRParameters parameters;

    uint64_t level_max() const { return apr_access.l_max; }
    uint64_t level_min() const { return apr_access.l_min; }
    inline uint64_t spatial_index_x_max(const unsigned int level) const { return apr_access.x_num[level]; }
    inline uint64_t spatial_index_y_max(const unsigned int level) const { return apr_access.y_num[level]; }
    inline uint64_t spatial_index_z_max(const unsigned int level) const { return apr_access.z_num[level]; }
    inline uint64_t total_number_particles() const { return apr_access.total_number_particles; }
    unsigned int orginal_dimensions(int dim) const { return apr_access.org_dims[dim]; }

    inline uint64_t total_number_tree_particles() const { return tree_access.total_number_particles; } // #TODO remove one of these
    inline uint64_t total_number_parent_cells() const { return tree_access.total_number_particles; }


    APRIterator iterator() {
        return APRIterator(apr_access);
    }

    LinearIterator linear_iterator() {
        return LinearIterator(apr_access);
    }


    APRTreeIterator tree_iterator() {
        return APRTreeIterator(apr_access,tree_access);
    }

    bool init_tree(){
        if(!tree_initialized){
            initialize_apr_tree_sparse();
            tree_initialized = true;
        }
        return tree_initialized;
    }

    void init_linear(){
        if(!linear_or_random){
            auto it = iterator();
            initialize_linear_access(apr_access,it);
        }
    }

    void init_tree_linear(){
        init_tree();
        if(!linear_or_random_tree){
            auto it = tree_iterator();
            initialize_linear_access(tree_access,it);
        }
    }


    APR(){
        //default
    }


    void copy_from_APR(APR& copyAPR){
        apr_access = copyAPR.apr_access;
        tree_access = copyAPR.tree_access;
        name = copyAPR.name;
    }


};

#define INTERIOR_PARENT 9


/**
   * Initializes linear access apr structures, that require more memory, but are faster. However, the do not allow the same neighbour access as the random iterators
   */
void APR::initialize_linear_access(APRAccess& aprAccess,GenIterator& it){

    auto& lin_a = aprAccess.linearAccess;

    uint64_t counter = 0;
    uint64_t counter_xz = 1;

    lin_a.level_end_vec.resize(it.level_max() + 1);
    lin_a.level_xz_vec.resize(it.level_max() + 1);

    lin_a.xz_end_vec.push_back(counter); // adding padding by one to allow the -1 syntax without checking.

    for (unsigned int level = 0; level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                for (it.begin(level, z, x); it < it.end();
                     it++) {
                    lin_a.y_vec.push_back(it.y());
                    counter++;
                }


                lin_a.xz_end_vec.push_back(counter);
                counter_xz++;
            }
        }

        lin_a.level_end_vec[level] = counter;
        lin_a.level_xz_vec[level] = counter_xz;
    }


}


/**
   * Initializes the APR tree datastructures using a dense structure for memory, these are all particle cells that are parents of particles in the APR
   * , alternatively can be thought of as the interior nodes of an APR represented as a binary tree.
   */
void APR::initialize_apr_tree() {

    APRTimer timer(true);

    auto apr_iterator = iterator();

    // --------------------------------------------------------------------
    // Init APR tree memory
    // --------------------------------------------------------------------

    // extend one extra level
    uint64_t l_max = apr_iterator.level_max() - 1;
    uint64_t l_min = apr_iterator.level_min() - 1;

    tree_access.l_min = l_min;
    tree_access.l_max = l_max;

    std::vector<PixelData<uint8_t>> particle_cell_parent_tree(l_max);


    timer.start_timer("tree - init structure");
    for (uint64_t l = l_min; l < l_max; ++l) {
        double cellSize = pow(2.0, l_max - l + 1);
        particle_cell_parent_tree[l].initWithValue(ceil(apr_iterator.orginal_dimensions(0) / cellSize),
                                                   ceil(apr_iterator.orginal_dimensions(1) / cellSize),
                                                   ceil(apr_iterator.orginal_dimensions(2) / cellSize),
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
        if (level < (apr_iterator.level_max())) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for ( z = 0; z < apr_iterator.z_num(level); z++) {
                for ( x = 0; x < apr_iterator.x_num(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator < apr_iterator.end();
                         apr_iterator++) {

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
            for ( z = 0; z < apr_iterator.z_num(level-1); z++) {
                for ( x = 0; x < apr_iterator.x_num(level-1); ++x) {
                    for (apr_iterator.set_new_lzx(level, 2*z, 2*x);
                         apr_iterator < apr_iterator.end();
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
    tree_access.initialize_tree_access(apr_access,particle_cell_parent_tree);
    timer.stop_timer();


}
/**
   * Initializes the APR tree datastructures using a sparse structure for reduced memory, these are all particle cells that are parents of particles in the APR
   * , alternatively can be thought of as the interior nodes of an APR represented as a binary tree.
   */
void APR::initialize_apr_tree_sparse() {

    APRTimer timer(false);

    auto apr_iterator = iterator();

    // --------------------------------------------------------------------
    // Init APR tree memory
    // --------------------------------------------------------------------

    // extend one extra level
    uint64_t l_max = apr_iterator.level_max() - 1;
    uint64_t l_min = apr_iterator.level_min() - 1;

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

        x_num[l] = (uint64_t) ceil((1.0 * apr_iterator.orginal_dimensions(1)) / pow(2.0, 1.0 * l_max - l + 1));
        z_num[l] = (uint64_t) ceil((1.0 * apr_iterator.orginal_dimensions(2)) / pow(2.0, 1.0 * l_max - l + 1));
        particle_cell_tree[l].resize(z_num[l]*x_num[l]);
        y_num[l] = (uint64_t) ceil((1.0 * apr_iterator.orginal_dimensions(0)) / pow(2.0, 1.0 * l_max - l + 1));

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

        if (level < (apr_iterator.level_max())) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
            for (z_d = 0; z_d < z_num[level-1]; z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr_iterator.z_num(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < x_num[level-1]; ++x_d) {
                        for (int x = 2 * x_d;
                             x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level) - 1); ++x) {

                            size_t x_p = x / 2;
                            size_t z_p = z / 2;
                            int current_level = level - 1;

                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator < apr_iterator.end();
                                 apr_iterator++) {
                                auto &pct = particle_cell_tree[current_level][z_p * x_num[current_level] + x_p];

                                size_t y_p = apr_iterator.y() / 2;

                                pct.mesh[y_p] = 1;

                            }
                        }
                    }
                }
            }

            if (level != (apr_iterator.level_max()-1)) {
                //second loop here pushing them further up the tree, the max level is special, as always exists on next level so not required.
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
                for (z_d = 0; z_d < z_num[level-1]; z_d++) {
                    for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_iterator.z_num(level) - 1); ++z) {
                        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                        for (x_d = 0; x_d < x_num[level - 1]; ++x_d) {
                            for (int x = 2 * x_d;
                                 x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level) - 1); ++x) {

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
            for (z_d = 0; z_d < apr_iterator.z_num(level - 2); z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_iterator.z_num(level-1) - 1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < apr_iterator.x_num(level-2); ++x_d) {
                        for (int x = 2 * x_d;
                             x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level-1) - 1); ++x) {

                            int x_p = x_d;
                            int z_p = z_d;
                            int current_level = level - 2;

                            for (apr_iterator.set_new_lzx(level, 2 * z, 2 * x);
                                 apr_iterator < apr_iterator.end();
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

    tree_access.x_num[l_max] = apr_access.x_num[l_max];
    tree_access.y_num[l_max] = apr_access.y_num[l_max];
    tree_access.z_num[l_max] = apr_access.z_num[l_max];


    timer.start_timer("tree - init tree");
    tree_access.initialize_tree_access_sparse(apr_access, particle_cell_tree);
    timer.stop_timer();

}



#endif //PARTPLAY_APR_HPP
