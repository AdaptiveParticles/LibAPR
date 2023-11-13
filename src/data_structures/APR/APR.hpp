//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "access/RandomAccess.hpp"
#include "iterators/APRIterator.hpp"
#include "iterators/APRTreeIterator.hpp"
#include "iterators/LinearIterator.hpp"
#include "GenInfo.hpp"

#ifdef APR_USE_CUDA
#include "access/GPUAccess.hpp"
#endif

class APR {
    friend class APRFile;
    template<typename T>
    friend class APRConverter;
    template<typename T>
    friend class APRConverterBatch;
    friend class APRBenchHelper;

public:

    // initialize tree RandomAccess
    void initialize_tree_random_sparse();
    void initialize_tree_random_dense();    // appears to be broken #TODO: remove or fix

    // initialize tree LinearAccess
    void initialize_tree_sparse();
    void initialize_tree_dense();

    void initialize_linear_access_from_random(LinearAccess& aprAccess, APRIterator& it);
    void initialize_random_access_from_linear();

    //New Access
    LinearAccess linearAccess;
    LinearAccess linearAccessTree;

    //Old Access
    RandomAccess apr_access;
    RandomAccess tree_access;

    bool apr_initialized = false;
    bool apr_initialized_random = false;

    bool tree_initialized = false;
    bool tree_initialized_random = false;

#ifdef APR_USE_CUDA
    GPUAccess gpuAccess;
    GPUAccess gpuTreeAccess;
#endif

    GenInfo aprInfo;
    GenInfo treeInfo;

    APRParameters parameters; // this is here to keep a record of what parameters were used, to then be written if needed.

#ifdef APR_USE_CUDA


    GPUAccessHelper gpuAPRHelper(){
        if(!apr_initialized){
            initialize_linear();
        }

        return GPUAccessHelper(gpuAccess,linearAccess);
    }

    GPUAccessHelper gpuTreeHelper(){
        if(!tree_initialized){
            initialize_tree_linear();
        }
        return GPUAccessHelper(gpuTreeAccess,linearAccessTree);
    }


    /**
     * Initialize GPU access and copy data to device
     * @param with_tree     include the tree access
     */
    void init_cuda(bool with_tree=true) {
        auto apr_helper = gpuAPRHelper();
        if(with_tree) {
            auto tree_helper = gpuTreeHelper();
            tree_helper.init_gpu();
            apr_helper.init_gpu(tree_helper);
        } else {
            apr_helper.init_gpu();
        }
    }
#endif


    APRParameters get_apr_parameters(){
        return parameters;
    }

    bool is_initialized(){
        if(!apr_initialized_random && !apr_initialized){
            return false;
        }
        return true;
    }

    std::string name;

    int level_max() const { return aprInfo.l_max; }
    int level_min() const { return aprInfo.l_min; }
    int level_size(int level) const { return aprInfo.level_size[level]; }
    inline int x_num(const int level) const { return aprInfo.x_num[level]; }
    inline int y_num(const int level) const { return aprInfo.y_num[level]; }
    inline int z_num(const int level) const { return aprInfo.z_num[level]; }
    inline uint64_t total_number_particles() const { return aprInfo.total_number_particles; }
    uint64_t org_dims(int dim) const { return aprInfo.org_dims[dim]; }

    inline uint64_t total_number_tree_particles() {
        if(!tree_initialized && !tree_initialized_random) {
            initialize_tree_linear();
        }
        return treeInfo.total_number_particles;
    }

    inline int number_dimensions() const {
        return aprInfo.number_dimensions;
    }

    double computational_ratio(){
        return (org_dims(0)*org_dims(1)*org_dims(2))/(1.0*aprInfo.total_number_particles);
    }


    APRIterator random_iterator() {

        if(!apr_initialized_random){
            initialize_random();
        }

        return APRIterator(apr_access,aprInfo);
    }

    LinearIterator iterator() {

        if(!apr_initialized){
            initialize_linear();
        }

        return LinearIterator(linearAccess,aprInfo);
    }

    APRTreeIterator random_tree_iterator() {

        if(!tree_initialized_random){
            initialize_tree_random();
        }

        return APRTreeIterator(apr_access,tree_access,treeInfo);
    }

    LinearIterator tree_iterator() {

        if(!tree_initialized){
            initialize_tree_linear();
        }

        return LinearIterator(linearAccessTree,treeInfo);
    }



    APR(){
        //default
    }

    //copy constructor
    APR(const APR& apr2copy){
        //Note: GPU datastructures are not copied.

        //default
        linearAccess.copy(apr2copy.linearAccess);
        linearAccessTree.copy(apr2copy.linearAccessTree);
        aprInfo = apr2copy.aprInfo;
        treeInfo = apr2copy.treeInfo;
        tree_initialized = apr2copy.tree_initialized;
        apr_initialized = apr2copy.apr_initialized;
        name = apr2copy.name;
        parameters = apr2copy.parameters;

        //old data structures
        apr_access = apr2copy.apr_access;
        tree_access = apr2copy.tree_access;
        apr_initialized_random = apr2copy.apr_initialized_random;
        tree_initialized_random = apr2copy.tree_initialized_random;

        linearAccess.genInfo = &aprInfo;
        linearAccessTree.genInfo = &treeInfo;
        apr_access.genInfo = &aprInfo;
        tree_access.genInfo = &treeInfo;

#ifdef APR_USE_CUDA
        gpuAccess.genInfo = &aprInfo;
        gpuTreeAccess.genInfo = &treeInfo;
#endif

    }

    void initialize_linear(){
        if(!apr_initialized){
            if(!apr_initialized_random) {
                std::cerr << "APR::initialize_linear - No APR initialized" << std::endl;
            } else {
                auto it = random_iterator();
                initialize_linear_access_from_random(linearAccess, it);
                apr_initialized = true;
            }
        }
    }

    void initialize_random() {
        if(!apr_initialized_random) {
            if(!apr_initialized) {
                std::cerr << "APR::initialize_random - No APR initialized" << std::endl;
            } else {
                initialize_random_access_from_linear();
                apr_initialized_random = true;
            }
        }
    }

    void initialize_tree_linear(bool sparse=false){

        if(!tree_initialized){
            if(sparse) {
                initialize_tree_sparse();
            } else {
                initialize_tree_dense();
            }
            tree_initialized = true;
        }
    }

    void initialize_tree_random(){

        if(!tree_initialized_random){
            initialize_tree_random_sparse();
            tree_initialized_random = true;
        }
    }

};

#define INTERIOR_PARENT 9


/**
 * Initializes linear access apr structures from the random access data. The linear structure is faster in most
 * cases, but require more memory and do not allow the same neighbor access as the random access iterators.
 */
void APR::initialize_linear_access_from_random(LinearAccess& lin_a, APRIterator& it){

    uint64_t counter = 0;
    uint64_t counter_xz = 1;

    lin_a.genInfo = &aprInfo;
    lin_a.initialize_xz_linear();

    lin_a.y_vec.resize(it.total_number_particles());

    for (int level = 0; level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                for (it.begin(level, z, x); it < it.end();
                     it++) {
                    lin_a.y_vec[counter] = it.y();
                    counter++;
                }

                lin_a.xz_end_vec[counter_xz] = counter;
                counter_xz++;
            }
        }
    }
}

/**
   * Initializes linear access apr structures, that require more memory, but are faster. However, the do not allow the same neighbour access as the random iterators
   */
void APR::initialize_random_access_from_linear(){
    //
    //  TODO: Note this is not performance orientataed, and should be depreciated in the future. (the whole random access iterations should be removed.)
    //


    auto it = iterator();

    apr_access.genInfo = &aprInfo;


    SparseGaps<SparseParticleCellMap> particle_cell_tree;

    particle_cell_tree.data.resize(aprInfo.l_max);

    for (int l = aprInfo.l_min; l < (aprInfo.l_max) ;l ++){
        particle_cell_tree.data[l].resize(aprInfo.z_num[l]*aprInfo.x_num[l]);

        for(size_t i = 0; i < particle_cell_tree.data[l].size(); ++i) {
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
void APR::initialize_tree_random_dense() {

    APRTimer timer(false);

    auto apr_iterator = iterator();

    // --------------------------------------------------------------------
    // Init APR tree memory
    // --------------------------------------------------------------------

    // extend one extra level

    treeInfo.init_tree(org_dims(0),org_dims(1),org_dims(2));
    tree_access.genInfo = &treeInfo;

    std::vector<PixelData<uint8_t>> particle_cell_parent_tree(treeInfo.l_max);


    timer.start_timer("tree - init structure");
    for (int l = treeInfo.l_min; l < treeInfo.l_max; ++l) {

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

    for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
        int z = 0;
        int x = 0;
        if (level < apr_iterator.level_max()) {
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

void APR::initialize_tree_sparse() {

    APRTimer timer(false);

    timer.start_timer("init");
    auto apr_iterator = iterator();

    // --------------------------------------------------------------------
    // Init APR tree memory
    // --------------------------------------------------------------------

    treeInfo.init_tree(org_dims(0),org_dims(1),org_dims(2));
    linearAccessTree.genInfo = &treeInfo;

    timer.stop_timer();

    timer.start_timer("tree - init sparse structure");

    std::vector<std::vector<SparseParticleCellMap>> particle_cell_tree;
    particle_cell_tree.resize(treeInfo.l_max + 1);

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

        if (level < (apr_iterator.level_max())) {

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

                            //if ((x % 2 == 0) && (z % 2 == 0)) {

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


            //second loop here pushing them further up the tree, the max level is special, as always exists on next level so not required.
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d) firstprivate(apr_iterator)
#endif
            for (z_d = 0; z_d < treeInfo.z_num[level-1]; z_d++) {
                for (int z = 2 * z_d; z <= std::min(2 * z_d + 1, (int) treeInfo.z_num[level] - 1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (int x_d = 0; x_d < treeInfo.x_num[level - 1]; ++x_d) {
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
        } else {
            int current_level = level - 1;
            int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.z_num(level); ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                for (int x = 0; x < apr_iterator.x_num(level); ++x) {

                    if ((x % 2 == 0) && (z % 2 == 0)) {
                        int x_p = x / 2;
                        int z_p = z / 2;

                        for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); apr_iterator++) {

                            auto &pct = particle_cell_tree[current_level][z_p * treeInfo.x_num[current_level] +x_p];
                            int y_p = apr_iterator.y() / 2;
                            pct.mesh[y_p] = 1;

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


}


void APR::initialize_tree_dense() {

    APRTimer timer(false);
    auto apr_iterator = iterator();

    treeInfo.init_tree(org_dims(0),org_dims(1),org_dims(2));
    linearAccessTree.genInfo = &treeInfo;

    std::vector<PixelData<uint8_t>> particle_cell_parent_tree(treeInfo.l_max+1);

    timer.start_timer("init tree - allocate dense tree structure");
    for (int l = treeInfo.l_min; l <= treeInfo.l_max; ++l) {

        particle_cell_parent_tree[l].initWithValue(treeInfo.y_num[l],
                                                   treeInfo.x_num[l],
                                                   treeInfo.z_num[l],
                                                   0);
    }
    timer.stop_timer();

    timer.start_timer("init tree - fill particle parents");
    // fill in parents of APR particles
    for(int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 2) firstprivate(apr_iterator)
#endif
        for(int z = 0; z < apr_iterator.z_num(level); ++z) {
            for(int x = 0; x < apr_iterator.x_num(level); ++x) {
                for(apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end(); ++apr_iterator) {
                    particle_cell_parent_tree[level-1].at(apr_iterator.y() / 2, x / 2, z / 2) = 1;
                }
            }
        }
    }
    timer.stop_timer();

    timer.start_timer("init tree - fill tree recursive");
    // fill rest of tree, level by level
    for(int level = treeInfo.l_max-1; level >= treeInfo.l_min; --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int z = 0; z < treeInfo.z_num[level]; ++z) {
            for (int x = 0; x < treeInfo.x_num[level]; ++x) {
                for(int y = 0; y < treeInfo.y_num[level]; ++y) {
                    // suffices to check one child
                    particle_cell_parent_tree[level].at(y, x, z) = particle_cell_parent_tree[level].at(y, x, z) | particle_cell_parent_tree[level+1].at(2*y, 2*x, 2*z);
                }
            }
        }
    }
    timer.stop_timer();

    timer.start_timer("create sparse data structure");
    linearAccessTree.initialize_tree_access_dense(particle_cell_parent_tree);
    timer.stop_timer();
}



/**
   * Initializes the APR tree datastructures using a sparse structure for reduced memory, these are all particle cells that are parents of particles in the APR
   * , alternatively can be thought of as the interior nodes of an APR represented as a binary tree.
   */
void APR::initialize_tree_random_sparse() {

    APRTimer timer(false);

    auto apr_iterator = random_iterator();

    //need to create a local copy

    // --------------------------------------------------------------------
    // Init APR tree memory
    // --------------------------------------------------------------------

    treeInfo.init_tree(org_dims(0),org_dims(1),org_dims(2));
    tree_access.genInfo = &treeInfo;

    std::vector<std::vector<SparseParticleCellMap>> particle_cell_tree;

    particle_cell_tree.resize(treeInfo.l_max + 1);

    timer.start_timer("tree - init sparse structure");
    for (int l = treeInfo.l_min; l < (treeInfo.l_max) ;l ++){

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

        if (level < (apr_iterator.level_max())) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z_d, x_d) firstprivate(apr_iterator)
#endif
            for (z_d = 0; z_d < treeInfo.z_num[level-1]; z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr_iterator.z_num(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < treeInfo.x_num[level-1]; ++x_d) {
                        for (int x = 2 * x_d;
                             x <= std::min(2 * x_d + 1, (int) apr_iterator.x_num(level) - 1); ++x) {

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
