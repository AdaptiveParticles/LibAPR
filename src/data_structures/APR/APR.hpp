//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "RandomAccess.hpp"
#include "APRIterator.hpp"
#include "APRTreeIterator.hpp"
#include "LinearIterator.hpp"
#include "GenInfo.hpp"

class APR {
    friend class APRFile;
    template<typename T>
    friend class APRConverter;

    friend class BenchmarkAPR;

protected:

    bool apr_initialized = false;
    bool apr_initialized_random = false;

    RandomAccess tree_access;
    bool tree_initialized = false;
    bool tree_initialized_random = false;

    //APR Tree function
    void initialize_apr_tree_sparse();
    void initialize_apr_tree_sparse_linear();
    void initialize_apr_tree();
    void initialize_linear_access(LinearAccess& aprAccess,GenIterator& it);

    LinearAccess linearAccess;
    LinearAccess linearAccessTree;

    RandomAccess apr_access;

    GenInfo aprInfo;
    GenInfo treeInfo;

public:

    APRParameters parameters; // #TODO move to protected. Introduce a get and set method

    std::string name;
    //APRParameters parameters;

    uint64_t level_max() const { return aprInfo.l_max; }
    uint64_t level_min() const { return aprInfo.l_min; }
    inline uint64_t spatial_index_x_max(const unsigned int level) const { return aprInfo.x_num[level]; }
    inline uint64_t spatial_index_y_max(const unsigned int level) const { return aprInfo.y_num[level]; }
    inline uint64_t spatial_index_z_max(const unsigned int level) const { return aprInfo.z_num[level]; }
    inline uint64_t total_number_particles() const { return aprInfo.total_number_particles; }
    unsigned int org_dims(int dim) const { return aprInfo.org_dims[dim]; }

    inline uint64_t total_number_tree_particles() const { return treeInfo.total_number_particles; } // #TODO remove one of these
    inline uint64_t total_number_parent_cells() const { return treeInfo.total_number_particles; }

    float computational_ratio(){
        return (org_dims(0)*org_dims(1)*org_dims(2))/(1.0f*aprInfo.total_number_particles);
    }


    APRIterator random_iterator() {

        if(!apr_initialized_random){
           initialize_random_access();
            apr_initialized_random = true;
        }

        return APRIterator(apr_access,aprInfo);
    }

    LinearIterator iterator() {
        // Just checking if its initialized
        if(!apr_initialized){
            init_linear();
            apr_initialized = true;
        }

        return LinearIterator(linearAccess,aprInfo);
    }

    APRTreeIterator random_tree_iterator() {

        if(!tree_initialized_random){
            init_tree_random();
        }

        return APRTreeIterator(apr_access,tree_access,treeInfo);
    }

    LinearIterator tree_iterator() {
        // Checking if initialized.
        if(!tree_initialized){
            init_tree_linear();
        }

        return LinearIterator(linearAccessTree,treeInfo);
    }



    APR(){
        //default
    }

    void copy_from_APR(APR& copyAPR){
        apr_access = copyAPR.apr_access;
        tree_access = copyAPR.tree_access;
        name = copyAPR.name;
    }

protected:

    bool init_tree_random(){
        if(!tree_initialized_random){

            initialize_apr_tree_sparse();
            tree_initialized_random = true;

        }
        return tree_initialized_random;
    }

    void init_linear(){
        if(!apr_initialized){
            auto it = random_iterator();
            initialize_linear_access(linearAccess,it);
        }
    }

    void init_tree_linear(){
        if(!tree_initialized){
            initialize_apr_tree_sparse_linear();
            tree_initialized = true;
        }
    }

    void initialize_random_access();


};

#define INTERIOR_PARENT 9


/**
   * Initializes linear access apr structures, that require more memory, but are faster. However, the do not allow the same neighbour access as the random iterators
   */
void APR::initialize_linear_access(LinearAccess& aprAccess,GenIterator& it){

    // TODO: Should be renamed.. random-> linear access. also need the reverse function

    auto& lin_a = aprAccess;

    uint64_t counter = 0;
    uint64_t counter_xz = 1;

    lin_a.level_xz_vec.resize(it.level_max() + 2,0);

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

        lin_a.level_xz_vec[level+1] = counter_xz;
    }


}

/**
   * Initializes linear access apr structures, that require more memory, but are faster. However, the do not allow the same neighbour access as the random iterators
   */
void APR::initialize_random_access(){
    //
    //  TODO: Note this is not really performance orientataed, and should be depreciated in the future. (the whole random access iterations should be removed.)
    //


    auto it = iterator();

    apr_access.genInfo = &aprInfo;


    SparseGaps<SparseParticleCellMap> particle_cell_tree;

    particle_cell_tree.data.resize(aprInfo.l_max);

    for (int l = aprInfo.l_min; l < (aprInfo.l_max) ;l ++){
        particle_cell_tree.data[l].resize(aprInfo.z_num[l]*aprInfo.x_num[l]);

        for (int i = 0; i < particle_cell_tree.data[l].size(); ++i) {
            particle_cell_tree.data[l][i].resize(1);
        }
    }

    for (int level = it.level_min(); level <= it.level_max(); level++) {
        int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it)
#endif
        for (z = 0; z < it.z_num(level); z++) {
            for (int x = 0; x < it.x_num(level); ++x) {

                if (level == it.level_max()) {
                    if ((x % 2 == 0) && (z % 2 == 0)) {
                        const size_t offset_pc =
                                aprInfo.x_num[level - 1] * ((z ) / 2) +
                                ((x ) / 2);

                        auto &pc = particle_cell_tree.data[level-1][offset_pc][0].mesh;

                        for (it.begin(level, z, x); it < it.end();
                             it++) {
                            //insert
                            int y = (it.y()) / 2;
                            pc[y] = 1;
                        }
                    }
                } else {

                    const size_t offset_pc =
                            aprInfo.x_num[level] * (z) + (x);

                    auto &pc = particle_cell_tree.data[level][offset_pc][0].mesh;

                    for (it.begin(level, z, x); it < it.end();
                         it++) {
                        //insert
                        auto y = it.y();
                        pc[y] = 4;
                    }
                }
            }
        }
    }

    apr_access.initialize_structure_from_particle_cell_tree_sparse(parameters,particle_cell_tree);

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

    treeInfo.init_tree(org_dims(0),org_dims(1),org_dims(2));
    tree_access.genInfo = &treeInfo;

    std::vector<PixelData<uint8_t>> particle_cell_parent_tree(treeInfo.l_max);


    timer.start_timer("tree - init structure");
    for (uint64_t l = treeInfo.l_min; l < treeInfo.l_max; ++l) {

        particle_cell_parent_tree[l].initWithValue(treeInfo.y_num[l],
                                                   treeInfo.x_num[l],
                                                   treeInfo.z_num[l],
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
                    for (apr_iterator.begin(level, z, x);
                         apr_iterator < apr_iterator.end();
                         apr_iterator++) {

                        size_t y_p = apr_iterator.y() / 2;
                        size_t x_p = x / 2;
                        size_t z_p = z / 2;
                        int current_level = level - 1;

                        if (particle_cell_parent_tree[current_level](y_p, x_p, z_p) == INTERIOR_PARENT) {
                            particle_cell_parent_tree[current_level](y_p, x_p, z_p) = 1;
                        } else {
                            particle_cell_parent_tree[current_level](y_p, x_p, z_p)++;
                        }

                        while (current_level > treeInfo.l_min) {
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
                    for (apr_iterator.begin(level, 2*z, 2*x);
                         apr_iterator < apr_iterator.end();
                         apr_iterator++) {

                        if (apr_iterator.y()%2 == 0) {
                            size_t y_p = apr_iterator.y() / 2;
                            size_t x_p = x / 2;
                            size_t z_p = z/ 2;
                            int current_level = level - 1;

                            while (current_level > treeInfo.l_min) {
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

void APR::initialize_apr_tree_sparse_linear() {

    APRTimer timer(false);

    auto apr_iterator = iterator();

    // --------------------------------------------------------------------
    // Init APR tree memory
    // --------------------------------------------------------------------

    treeInfo.init_tree(org_dims(0),org_dims(1),org_dims(2));
    linearAccessTree.genInfo = &treeInfo;

    std::vector<std::vector<SparseParticleCellMap>> particle_cell_tree;

    particle_cell_tree.resize(treeInfo.l_max + 1);

    timer.start_timer("tree - init sparse structure");
    for (int l = treeInfo.l_min; l <= (treeInfo.l_max) ;l ++){

        particle_cell_tree[l].resize(treeInfo.z_num[l]*treeInfo.x_num[l]);

    }
    timer.stop_timer();


    // --------------------------------------------------------------------
    // Insert values to the APR tree
    // --------------------------------------------------------------------
    timer.start_timer("tree - insert vals sparse");

    //note the use of the dynamic OpenMP schedule.

    for (int level = (apr_iterator.level_max()); level >= apr_iterator.level_min(); --level) {
        int z_d = 0;
        int x_d = 0;

        int z_max_ds = treeInfo.z_num[level - 1];
        int x_max_ds = treeInfo.x_num[level - 1];

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
        for (z_d = 0; z_d <z_max_ds; z_d++) {
            int z_max = std::min(2 * z_d + 1, ((int) apr_iterator.z_num(level)) - 1);
            for (int z = 2 * z_d; z <= z_max; ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                for (x_d = 0; x_d < x_max_ds; ++x_d) {
                    int x_max = std::min(2 * x_d + 1, ((int) apr_iterator.x_num(level)) - 1);
                    for (int x = 2 * x_d; x <= x_max; ++x) {

                        int x_p = x / 2;
                        int z_p = z / 2;
                        int current_level = level - 1;


                        for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); apr_iterator++) {

                            auto &pct = particle_cell_tree[current_level][z_p * treeInfo.x_num[current_level] +
                                                                          x_p];
                            int y_p = apr_iterator.y() / 2;

                            pct.mesh[y_p] = 1;

                        }
                    }
                }
            }
        }

        if (level < (apr_iterator.level_max())) {
            //second loop here pushing them further up the tree, the max level is special, as always exists on next level so not required.
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
            for (z_d = 0; z_d < treeInfo.z_num[level-1]; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) treeInfo.z_num[level] - 1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < treeInfo.x_num[level - 1]; ++x_d) {
                        for (int x = 2 * x_d;
                             x <= std::min(2 * x_d + 1, (int) treeInfo.x_num[level] - 1); ++x) {

                            auto &pct = particle_cell_tree[level][z * treeInfo.x_num[level] + x];

                            auto x_p = x / 2;
                            auto z_p = z / 2;
                            int parent_level = level - 1;

                            auto &pct_p = particle_cell_tree[parent_level][z_p * treeInfo.x_num[parent_level] + x_p];

                            for (auto it = pct.mesh.begin(); it != pct.mesh.end(); ++it) {
                                int y_p = (it->first) / 2;

                                pct_p.mesh[y_p] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();


    timer.start_timer("tree - init tree");
    linearAccessTree.initialize_tree_access_sparse(particle_cell_tree);
    timer.stop_timer();


};

/**
   * Initializes the APR tree datastructures using a sparse structure for reduced memory, these are all particle cells that are parents of particles in the APR
   * , alternatively can be thought of as the interior nodes of an APR represented as a binary tree.
   */
void APR::initialize_apr_tree_sparse() {

    APRTimer timer(false);

    auto apr_iterator = iterator();

    //need to create a local copy

    // --------------------------------------------------------------------
    // Init APR tree memory
    // --------------------------------------------------------------------

    treeInfo.init_tree(org_dims(0),org_dims(1),org_dims(2));
    tree_access.genInfo = &treeInfo;

    std::vector<std::vector<SparseParticleCellMap>> particle_cell_tree;

    particle_cell_tree.resize(treeInfo.l_max + 1);

    timer.start_timer("tree - init sparse structure");
    for (unsigned int l = treeInfo.l_min; l < (treeInfo.l_max) ;l ++){

        particle_cell_tree[l].resize(treeInfo.z_num[l]*treeInfo.x_num[l]);

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
            for (z_d = 0; z_d < treeInfo.z_num[level-1]; z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr_iterator.z_num(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < treeInfo.x_num[level-1]; ++x_d) {
                        for (int x = 2 * x_d;
                             x <= std::min(2 * x_d + 1, (int) treeInfo.x_num[level] - 1); ++x) {

                            size_t x_p = x / 2;
                            size_t z_p = z / 2;
                            int current_level = level - 1;

                            for (apr_iterator.begin(level, z, x);
                                 apr_iterator < apr_iterator.end();
                                 apr_iterator++) {
                                auto &pct = particle_cell_tree[current_level][z_p * treeInfo.x_num[current_level] + x_p];

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
                for (z_d = 0; z_d < treeInfo.z_num[level-1]; z_d++) {
                    for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) apr_iterator.z_num(level) - 1); ++z) {
                        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                        for (x_d = 0; x_d < treeInfo.x_num[level - 1]; ++x_d) {
                            for (int x = 2 * x_d;
                                 x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level) - 1); ++x) {

                                auto &pct = particle_cell_tree[level][z * treeInfo.x_num[level] + x];

                                size_t x_p = x / 2;
                                size_t z_p = z / 2;
                                int parent_level = level - 1;

                                auto &pct_p = particle_cell_tree[parent_level][z_p * treeInfo.x_num[parent_level] + x_p];

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

                            for (apr_iterator.begin(level, 2 * z, 2 * x);
                                 apr_iterator < apr_iterator.end();
                                 apr_iterator++) {

                                auto &pct = particle_cell_tree[current_level][z_p * treeInfo.x_num[current_level] + x_p];

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

    timer.start_timer("tree - init tree");
    tree_access.initialize_tree_access_sparse(apr_access, particle_cell_tree);
    timer.stop_timer();

}



#endif //PARTPLAY_APR_HPP
