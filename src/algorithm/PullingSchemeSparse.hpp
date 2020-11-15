//
// Created by cheesema on 05.07.18.
//

#ifndef LIBAPR_PULLINGSCHEMESPARSE_HPP
#define LIBAPR_PULLINGSCHEMESPARSE_HPP

#include <cassert>
#include "data_structures/Mesh/PixelData.hpp"
#include "../data_structures/APR/APR.hpp"
#include "../data_structures/APR/access/APRAccessStructures.hpp"

#ifdef HAVE_OPENMP
#include "omp.h"
#endif

#define EMPTY 0
#define SEED_TYPE 1
#define BOUNDARY_TYPE 2
#define FILLER_TYPE 3
#define ASCENDANT 8
#define PROPOGATE 15
#define ASCENDANTNEIGHBOUR 16

#define NEIGHBOURLOOP(jn,in,kn, boundaries) \
for(jn = boundaries[0][0]; jn < boundaries[0][1]; jn++) \
    for(in = boundaries[1][0]; in < boundaries[1][1]; in++) \
        for(kn = boundaries[2][0]; kn < boundaries[2][1]; kn++)


#define CHILDRENLOOP(jn,in,kn, children_boundaries) \
for(jn = j * 2; jn < j * 2 + children_boundaries[0]; jn++) \
    for(in = i * 2; in < i * 2 + children_boundaries[1]; in++) \
        for(kn = k * 2; kn < k * 2 + children_boundaries[2]; kn++)

// don't try to optimize check boundaries - every check is needed due to parallelism
#define CHECKBOUNDARIES(axis,var,limit,boundaries) \
    if (var == 0) { \
        boundaries[axis][0] = 0;\
    } else {\
        boundaries[axis][0] = -1;\
    }\
    if (var == limit) {\
        boundaries[axis][1] = 1;\
    } else {\
        boundaries[axis][1] = 2;\
    }


struct imagePatch {

//    uint64_t x_begin;
//    uint64_t x_end;
    uint64_t x_offset;

//    uint64_t y_begin;
//    uint64_t y_end;
    uint64_t y_offset;

//    uint64_t z_begin;
//    uint64_t z_end;
    uint64_t z_offset;

//    uint64_t z_begin_ghost;
//    uint64_t z_end_ghost;
//
//    uint64_t x_begin_ghost;
//    uint64_t x_end_ghost;
//
//    uint64_t y_begin_ghost;
//    uint64_t y_end_ghost;
//
//    uint64_t z_begin_global;
//    uint64_t z_end_global;
//
//    uint64_t x_begin_global;
//    uint64_t x_end_global;
//
//    uint64_t y_begin_global;
//    uint64_t y_end_global;

};


class PullingSchemeSparse {

    double powr(uint64_t num,uint64_t pow2){
        //return (uint64_t) std::round(std::pow(num,pow2));
        return std::round(pow(num,pow2));
    }

public:

    SparseGaps<SparseParticleCellMap> particle_cell_tree;

    template<typename T>
    void fill(float level, const PixelData<T> &input,imagePatch& patch);
    template<typename T>
    void fill(float level, const PixelData<T> &input);

    void pulling_scheme_main();

    void initialize_particle_cell_tree(const GenInfo& aprInfo);

    int pct_level_max(){
        return l_max;
    };
    int pct_level_min(){
        return l_min;
    };

private:

    int l_min;
    int l_max;

    std::vector<size_t> y_num_l;

    void set_ascendant_neighbours(int level);
    void set_filler(int level);
    void fill_neighbours(int level);
    void fill_parent(size_t j, size_t i, size_t k, size_t x_num, size_t y_num, int new_level);
};


inline void PullingSchemeSparse::initialize_particle_cell_tree(const GenInfo& aprInfo) {
    //  Initializes the particle cell tree structure
    //
    //  Contains pc up to l_max - 1,
    //

    l_max = aprInfo.l_max - 1;
    l_min = aprInfo.l_min;
    //make so you can reference the array as l
    particle_cell_tree.data.resize(l_max + 1);

    particle_cell_tree.z_num.resize(l_max + 1);
    particle_cell_tree.x_num.resize(l_max + 1);
    y_num_l.resize(l_max + 1);
    particle_cell_tree.data.resize(l_max + 1);

    for (int l = l_min; l < (l_max + 1) ;l ++){

        particle_cell_tree.x_num[l] = (uint64_t) ceil((1.0 * aprInfo.org_dims[1]) / powr(2, 1 * l_max - l + 1));
        particle_cell_tree.z_num[l] = (uint64_t) ceil((1.0 * aprInfo.org_dims[2]) / powr(2, 1 * l_max - l + 1));
        particle_cell_tree.data[l].resize(particle_cell_tree.z_num[l]*particle_cell_tree.x_num[l]);
        y_num_l[l] = (uint64_t) ceil((1.0 * aprInfo.org_dims[0]) / powr(2, 1 * l_max - l + 1));

        for (size_t i = 0; i < particle_cell_tree.z_num[l] ; ++i) {
            for (size_t j = 0; j < particle_cell_tree.x_num[l]; ++j) {
                particle_cell_tree.data[l][i*particle_cell_tree.x_num[l]+j].resize(1);
            }
        }

    }




}

inline void PullingSchemeSparse::pulling_scheme_main() {
    //
    //  Bevan Cheeseman 2016
    //
    //  The Pulling Scheme for forming the Optimal Valid Particle Cell set from the Local Particle Cell set L
    //
    //  Implimented as discussed in Cheeseman et al. 2018 for full description.
    //
    //  Generates the implied resolution function that is used to sample the image in the APR.
    //

    APRTimer timer;
    timer.verbose_flag = false;


    //loop over all levels from l_max to l_min
    for (int level = l_max; level >= (int)l_min; --level) {

        timer.start_timer("set");
        if (level != (int)l_max) {
            set_ascendant_neighbours(level); //step 1 and step 2.
            set_filler(level); // step 3.
        }
        timer.stop_timer();

        timer.start_timer("fill");
        fill_neighbours(level); // step 4.
        timer.stop_timer();
    }
}

template<typename T>
inline void PullingSchemeSparse::fill(const float level, const PixelData<T> &input){
    imagePatch patch;
    patch.x_offset = 0;
    patch.y_offset = 0;
    patch.z_offset = 0;

    fill(level, input,patch);

}


template<typename T>
inline void PullingSchemeSparse::fill(const float level, const PixelData<T> &input,imagePatch& patch) {
    //  Bevan Cheeseman 2016
    //
    //  Updates the hash table from the down sampled images

    //auto &mesh = particle_cell_tree[k].mesh;

    const size_t x_num = particle_cell_tree.x_num[level];
   // const size_t y_num = y_num_l[level];

    const size_t offset_x = patch.x_offset/((int)powr(2,(int)l_max + 1 - level));
    const size_t offset_y = patch.y_offset/((int)powr(2,(int)l_max + 1 - level));
    const size_t offset_z = patch.z_offset/((int)powr(2,(int)l_max + 1 - level));

    //
    // Need offset and original x,y,z nums
    //

    if (level == l_max){
        // k_max loop, has to include
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for (size_t z = 0; z < input.z_num; ++z) {
            for (size_t x = 0; x < input.x_num; ++x) {
                const size_t offset_part_map = x * input.y_num + z * input.y_num * input.x_num;
                const size_t offset_pc =  x_num * (z+offset_z) + (x+offset_x);
                auto& mesh = particle_cell_tree.data[level][offset_pc][0].mesh;
                for (size_t y = 0; y < input.y_num; ++y) {

                    if (input.mesh[offset_part_map + y ] >= level) {
                        mesh[y + offset_y] = SEED_TYPE;
                    }
                }
            }

        }


    }
    else if (level == l_min) {
        // k_min loop, has to include
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for (size_t z = 0; z < input.z_num; ++z) {
            for (size_t x = 0; x < input.x_num; ++x) {
                const size_t offset_part_map = x * input.y_num + z * input.y_num * input.x_num;
                const size_t offset_pc =  x_num * (z+offset_z) + (x+offset_x);
                auto& mesh = particle_cell_tree.data[level][offset_pc][0].mesh;
                for (size_t y = 0; y < input.y_num; ++y) {

                    if (input.mesh[offset_part_map + y] <= level) mesh[y + offset_y] = SEED_TYPE;
                }
            }

        }
    }
    else {
        // other k's
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for (size_t z = 0; z < input.z_num; ++z) {
            for (size_t x = 0; x < input.x_num; ++x) {
                const size_t offset_part_map = x * input.y_num + z * input.y_num * input.x_num;
                const size_t offset_pc =  x_num * (z+offset_z) + (x+offset_x);
                auto& mesh = particle_cell_tree.data[level][offset_pc][0].mesh;
                for (size_t y = 0; y < input.y_num; ++y) {

                    if (input.mesh[offset_part_map + y] == level) mesh[y + offset_y] = SEED_TYPE;
                }
            }

        }

    }
}

inline void PullingSchemeSparse::set_ascendant_neighbours(int level) {
    const size_t x_num = particle_cell_tree.x_num[level];
    const size_t y_num = y_num_l[level];
    const size_t z_num = particle_cell_tree.z_num[level];

    short boundaries[3][2] = {{0,2},{0,2},{0,2}};

    // loop unrolling in order to avoid concurrent write
    for (size_t out = 0; out < std::min((size_t)3, z_num); ++out) {
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) schedule(dynamic) firstprivate(boundaries) if(z_num * x_num * y_num > 100000)
#endif
        for (size_t j = out; j < z_num; j += 3) {
            CHECKBOUNDARIES(0, j, z_num - 1, boundaries);
            for (size_t i = 0; i < x_num; i++) {
                CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                //size_t index = j * x_num * y_num + i * y_num;

                const size_t offset_pc =  x_num * j + i;
                auto& mesh = particle_cell_tree.data[level][offset_pc][0].mesh;

                //SPARSE iteration
                for (auto it=mesh.begin(); it!=mesh.end(); ++it){
                    size_t k = it->first;
                    CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                    uint8_t status = it->second;
                    if (status == ASCENDANT) {
                        int64_t jn, in, kn;
                        NEIGHBOURLOOP(jn, in, kn, boundaries) {
                                    size_t neighbour_index = kn + k;

                                    const size_t offset_pc_n = offset_pc + x_num * jn + in;
                                    auto& mesh_n = particle_cell_tree.data[level][offset_pc_n][0].mesh;

                                    if (mesh_n[neighbour_index] == EMPTY) {
                                        // type is EMPTY
                                        mesh_n[neighbour_index]  = ASCENDANTNEIGHBOUR;
                                    }

                                    if (mesh_n[neighbour_index] == SEED_TYPE) {
                                        // type is SEED
                                        mesh_n[neighbour_index] = PROPOGATE;
                                    }
                                }
                    }
                }
            }
        }
    }
}

inline void PullingSchemeSparse::set_filler(int level) {
    short children_boundaries[3] = {2,2,2};

    const int64_t x_num = particle_cell_tree.x_num[level];
    const int64_t y_num = y_num_l[level];
    const int64_t z_num = particle_cell_tree.z_num[level];

    int64_t prev_x_num = particle_cell_tree.x_num[level+1];
    int64_t prev_y_num = y_num_l[level+1];
    int64_t prev_z_num = particle_cell_tree.z_num[level+1];


#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) schedule(dynamic) if (z_num * x_num * y_num > 10000) firstprivate(level, children_boundaries)
#endif
    for (int64_t j = 0; j < z_num; ++j) {
        if ( (j == z_num - 1) && prev_z_num % 2 ) {
            children_boundaries[0] = 1;
        }

        for (int64_t i = 0; i < x_num; ++i) {

            if ( (i == x_num - 1) && prev_x_num % 2 ) {
                children_boundaries[1] = 1;
            }
            else if ( i == 0 ) {
                children_boundaries[1] = 2;
            }

            children_boundaries[2] = 2;

            const size_t offset_pc = (size_t) x_num * j + i;
            auto& mesh = particle_cell_tree.data[level][offset_pc][0].mesh;

            //SPARSE iteration
            for(auto it=mesh.begin(); it!=mesh.end(); ++it){
                int64_t k = it->first;
                if ( (k == y_num - 1) && prev_y_num % 2 ) {
                    children_boundaries[2] = 1;
                }
                else if ( k == 0 ) {
                    children_boundaries[2] = 2;
                }

                const uint8_t status = it->second;
                if (status == ASCENDANTNEIGHBOUR || status == PROPOGATE) {
                    // go down, and set empty children to FILLER
                    int64_t jn, in, kn;
                    CHILDRENLOOP(jn, in, kn, children_boundaries) {
//                                size_t children_index = jn * prev_x_num * prev_y_num + in * prev_y_num + kn;
//                                uint8_t children_status = particle_cell_tree[level + 1].mesh[children_index];
//
                                size_t children_index = kn;

                                size_t offset_pc_c =  prev_x_num * jn + in;


                                uint8_t status_c = particle_cell_tree.data[level+1][offset_pc_c][0].mesh[children_index];

                                if (status_c == EMPTY) {
                                    particle_cell_tree.data[level+1][offset_pc_c][0].mesh[children_index] = FILLER_TYPE;
                                }
                            }
                }
            }
        }
    }
}

inline void PullingSchemeSparse::fill_neighbours(int level) {
    const size_t x_num = particle_cell_tree.x_num[level];
    const size_t y_num = y_num_l[level];
    const size_t z_num = particle_cell_tree.z_num[level];

    short boundaries[3][2] = {{0,2},{0,2},{0,2}};
    // loop unrolling in order to avoid concurrent write
    for (size_t out = 0; out < std::min((size_t)3,z_num); ++out) {
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) firstprivate(boundaries) schedule(dynamic) if (z_num * x_num * y_num > 100000)
#endif
        for (size_t j = out; j < z_num; j += 3) {
            CHECKBOUNDARIES(0, j, z_num - 1, boundaries);
            for (size_t i = 0; i < x_num; ++i) {
                CHECKBOUNDARIES(1, i, x_num - 1, boundaries);

                const size_t offset_pc = (size_t) x_num * j + i;
                auto& mesh = particle_cell_tree.data[level][offset_pc][0].mesh;

                //SPARSE iteration
                for (auto it=mesh.begin(); it!=mesh.end(); ++it){
                    size_t k = it->first;
                    CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                    uint8_t status = it->second;
                    if (status == SEED_TYPE || status == PROPOGATE) {
                        int64_t jn, in, kn;
                        NEIGHBOURLOOP(jn, in, kn, boundaries) {

                                    size_t neighbour_index = kn + k;

                                    const size_t offset_pc_n = offset_pc + x_num * jn + in;
                                    auto& mesh_n = particle_cell_tree.data[level][offset_pc_n][0].mesh;

                                    if (mesh_n[neighbour_index] == EMPTY) {
                                        mesh_n[neighbour_index] = BOUNDARY_TYPE;
                                    }
                                }
                        fill_parent(j, i, k, x_num, y_num, level - 1);
                    }
                    else if (status == ASCENDANT) {
                        fill_parent(j, i, k, x_num, y_num, level - 1);
                    }
                }
            }
        }
    }
}

inline void PullingSchemeSparse::fill_parent(size_t j, size_t i, size_t k, size_t x_num, size_t y_num, int new_level) {
    if(new_level >= l_min) {
        size_t new_x_num = ((x_num + 1) / 2);
        //size_t new_y_num = ((y_num + 1) / 2);
        //size_t new_index = (j / 2) * new_x_num * new_y_num + (i / 2) * new_y_num + (k / 2);

        const size_t offset_pc = (size_t) new_x_num * (j/2) + (i/2);
        auto& mesh = particle_cell_tree.data[new_level][offset_pc][0].mesh;
        size_t new_index = (k/2);
        uint8_t status = mesh[new_index];

        if (status != SEED_TYPE) {
            mesh[new_index] = ASCENDANT;
        }
    }
}




#endif //LIBAPR_PULLINGSCHEMESPARSE_HPP
