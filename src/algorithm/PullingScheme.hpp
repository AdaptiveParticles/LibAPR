//
//  Bevan Cheeseman 2018
//
//  The Pulling Scheme using the full Particle Cell Pyramid as a tree structure.
//
//
//  This code was originally optimized for the OpenMP version by Matteusz Susik in (2016)
//
//

#ifndef PARTPLAY_PULLING_SCHEME_HPP
#define PARTPLAY_PULLING_SCHEME_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "../data_structures/APR/APRAccess.hpp"

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


class PullingScheme {

public:
    template<typename T>
    void fill(float k, const PixelData<T> &input);
    void pulling_scheme_main();
    void initialize_particle_cell_tree(const APRAccess &apr_access);
    std::vector<PixelData<uint8_t>>& getParticleCellTree() { return particle_cell_tree; }

private:
    void set_ascendant_neighbours(int level);
    void set_filler(int level);
    void fill_neighbours(int level);
    void fill_parent(size_t j, size_t i, size_t k, size_t x_num, size_t y_num, size_t new_level);

    std::vector<PixelData<uint8_t>> particle_cell_tree;
    unsigned int l_min;
    unsigned int l_max;
};

inline void PullingScheme::initialize_particle_cell_tree(const APRAccess &apr_access) {
    //  Initializes the particle cell tree structure
    //
    //  Contains pc up to l_max - 1,
    //

    l_max = apr_access.level_max - 1;
    l_min = apr_access.level_min;
    //make so you can reference the array as l
    particle_cell_tree.resize(l_max + 1);

    for (unsigned int l = l_min; l < (l_max + 1) ;l ++){
        particle_cell_tree[l].init(ceil((1.0 * apr_access.org_dims[0]) / pow(2.0, 1.0 * l_max - l + 1)),
                                   ceil((1.0 * apr_access.org_dims[1]) / pow(2.0, 1.0 * l_max - l + 1)),
                                   ceil((1.0 * apr_access.org_dims[2]) / pow(2.0, 1.0 * l_max - l + 1)), EMPTY);
    }
}

inline void PullingScheme::pulling_scheme_main() {
    //
    //  Bevan Cheeseman 2016
    //
    //  The Pulling Scheme for forming the Optimal Valid Particle Cell set from the Local Particle Cell set L
    //
    //  Implimented as discussed in Cheeseman et al. 2017 for full description.
    //
    //  Generates the implied resolution function that is used to sample the image in the APR.
    //


    //loop over all levels from l_max to l_min
    for (int level = l_max; level >= (int)l_min; --level) {
        if (level != (int)l_max) {
            set_ascendant_neighbours(level); //step 1 and step 2.
            set_filler(level); // step 3.
        }
        fill_neighbours(level); // step 4.
    }
}

template<typename T>
inline void PullingScheme::fill(const float k, const PixelData<T> &input) {
    //  Bevan Cheeseman 2016
    //
    //  Updates the hash table from the down sampled images

    auto &mesh = particle_cell_tree[k].mesh;

    if (k == l_max){
        // k_max loop, has to include
        #ifdef HAVE_OPENMP
	    #pragma omp parallel for default(shared)
        #endif
        for (size_t i = 0; i < input.mesh.size(); ++i) {
            if (input.mesh[i] >= k) mesh[i] = SEED_TYPE;
        }
    }
    else if (k == l_min) {
        // k_min loop, has to include
        #ifdef HAVE_OPENMP
	    #pragma omp parallel for default(shared)
        #endif
        for (size_t i = 0; i < input.mesh.size(); ++i) {
            if (input.mesh[i] <= k) mesh[i] = SEED_TYPE;
        }
    }
    else {
        // other k's
        #ifdef HAVE_OPENMP
	    #pragma omp parallel for default(shared)
        #endif
        for (size_t i = 0; i < input.mesh.size(); ++i) {
            if (input.mesh[i] == k) mesh[i] = SEED_TYPE;
        }
    }
}

inline void PullingScheme::set_ascendant_neighbours(int level) {
    const size_t x_num = particle_cell_tree[level].x_num;
    const size_t y_num = particle_cell_tree[level].y_num;
    const size_t z_num = particle_cell_tree[level].z_num;

    short boundaries[3][2] = {{0,2},{0,2},{0,2}};

    // loop unrolling in order to avoid concurrent write
    for (size_t out = 0; out < std::min((size_t)3, z_num); ++out) {
        #ifdef HAVE_OPENMP
        #pragma omp parallel for default(shared) firstprivate(boundaries) if(z_num * x_num * y_num > 100000) schedule(static)
        #endif
        for (size_t j = out; j < z_num; j += 3) {
            CHECKBOUNDARIES(0, j, z_num - 1, boundaries);
            for (size_t i = 0; i < x_num; i++) {
                CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                size_t index = j * x_num * y_num + i * y_num;
                for (size_t k = 0; k < y_num; k++) {
                    CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                    uint8_t status = particle_cell_tree[level].mesh[index + k];
                    if (status == ASCENDANT) {
                        int64_t jn, in, kn;
                        NEIGHBOURLOOP(jn, in, kn, boundaries) {
                            size_t neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                            if (particle_cell_tree[level].mesh[neighbour_index] == EMPTY) {
                                // type is EMPTY
                                particle_cell_tree[level].mesh[neighbour_index] = ASCENDANTNEIGHBOUR;
                            }

                            if (particle_cell_tree[level].mesh[neighbour_index] == SEED_TYPE) {
                                // type is SEED
                                particle_cell_tree[level].mesh[neighbour_index] = PROPOGATE;
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void PullingScheme::set_filler(int level) {
    short children_boundaries[3] = {2,2,2};
    const int64_t x_num = particle_cell_tree[level].x_num;
    const int64_t y_num = particle_cell_tree[level].y_num;
    const int64_t z_num = particle_cell_tree[level].z_num;

    int64_t prev_x_num = particle_cell_tree[level + 1].x_num;
    int64_t prev_y_num = particle_cell_tree[level + 1].y_num;
    int64_t prev_z_num = particle_cell_tree[level + 1].z_num;

    #ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) if (z_num * x_num * y_num > 10000) firstprivate(level, children_boundaries)
    #endif
    for (int64_t j = 0; j < z_num; ++j) {
        if ( j == z_num - 1 && prev_z_num % 2 ) {
            children_boundaries[0] = 1;
        }

        for (int64_t i = 0; i < x_num; ++i) {

            if ( i == x_num - 1 && prev_x_num % 2 ) {
                children_boundaries[1] = 1;
            }
            else if ( i == 0 ) {
                children_boundaries[1] = 2;
            }

            size_t index = j*x_num*y_num + i*y_num;

            for (int64_t k = 0; k < y_num; ++k) {
                if ( k == y_num - 1 && prev_y_num % 2 ) {
                    children_boundaries[2] = 1;
                }
                else if ( k == 0 ) {
                    children_boundaries[2] = 2;
                }

                uint8_t status = particle_cell_tree[level].mesh[index + k];
                if (status == ASCENDANTNEIGHBOUR || status == PROPOGATE) {
                    // go down, and set empty children to FILLER
                    int64_t jn, in, kn;
                    CHILDRENLOOP(jn, in, kn, children_boundaries) {
                        size_t children_index = jn * prev_x_num * prev_y_num + in * prev_y_num + kn;
                        uint8_t children_status = particle_cell_tree[level + 1].mesh[children_index];
                        if (children_status == EMPTY) {
                            particle_cell_tree[level + 1].mesh[children_index] = FILLER_TYPE;
                        }
                    }
                }
            }
        }
    }
}

inline void PullingScheme::fill_neighbours(int level) {
    const size_t x_num = particle_cell_tree[level].x_num;
    const size_t y_num = particle_cell_tree[level].y_num;
    const size_t z_num = particle_cell_tree[level].z_num;

    short boundaries[3][2] = {{0,2},{0,2},{0,2}};
    // loop unrolling in order to avoid concurrent write
    for (size_t out = 0; out < std::min((size_t)3,z_num); ++out) {
        #ifdef HAVE_OPENMP
        #pragma omp parallel for default(shared) firstprivate(boundaries) if (z_num * x_num * y_num > 100000)
        #endif
        for (size_t j = out; j < z_num; j += 3) {
            CHECKBOUNDARIES(0, j, z_num - 1, boundaries);
            for (size_t i = 0; i < x_num; ++i) {
                CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                size_t index = j*x_num*y_num + i*y_num;
                for (size_t k = 0; k < y_num; ++k) {
                    CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                    uint8_t status = particle_cell_tree[level].mesh[index + k];
                    if (status == SEED_TYPE || status == PROPOGATE) {
                        int64_t jn, in, kn;
                        NEIGHBOURLOOP(jn, in, kn, boundaries) {
                            size_t neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;
                            if (particle_cell_tree[level].mesh[neighbour_index] == EMPTY) {
                                particle_cell_tree[level].mesh[neighbour_index] = BOUNDARY_TYPE;
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

inline void PullingScheme::fill_parent(size_t j, size_t i, size_t k, size_t x_num, size_t y_num, size_t new_level) {
    if(new_level >= l_min) {
        size_t new_x_num = ((x_num + 1) / 2);
        size_t new_y_num = ((y_num + 1) / 2);
        size_t new_index = (j / 2) * new_x_num * new_y_num + (i / 2) * new_y_num + (k / 2);

        if (particle_cell_tree[new_level].mesh[new_index] != SEED_TYPE) {
            particle_cell_tree[new_level].mesh[new_index] = ASCENDANT;
        }
    }
}

#endif //PARTPLAY_PULLING_SCHEME_HPP
