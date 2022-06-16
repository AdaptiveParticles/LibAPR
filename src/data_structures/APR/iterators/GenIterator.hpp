//
// Created by cheesema on 29.06.18.
//

#ifndef LIBAPR_GENITERATOR_HPP
#define LIBAPR_GENITERATOR_HPP

#include "../access/RandomAccess.hpp"

class GenIterator {

protected:

    //Pointer to the actual access information used by the iterator

    uint64_t end_index = 0;

    GenInfo *genInfo;

public:

    inline uint64_t total_number_particles() { return genInfo->total_number_particles; }

    inline int x_nearest_pixel(const int level, const int x) { return (int) floor((x+0.5)*level_size(level)); }
    inline int y_nearest_pixel(const int level, const int y) { return (int) floor((y+0.5)*level_size(level)); }
    inline int z_nearest_pixel(const int level, const int z) { return (int) floor((z+0.5)*level_size(level)); }
    inline float x_global(const int level, const int x) { return (x+0.5f)*level_size(level); }
    inline float y_global(const int level, const int y) { return (y+0.5f)*level_size(level); }
    inline float z_global(const int level, const int z) { return (z+0.5f)*level_size(level); }

    inline int level_size(const int level) { return genInfo->level_size[level]; }

    inline int org_dims(const int dim) const { return genInfo->org_dims[dim]; }

    inline int number_dimensions() { return genInfo->number_dimensions; }

    inline int level_min() { return genInfo->l_min; }
    inline int level_max() { return genInfo->l_max; }

    inline int x_num(const unsigned int level) { return genInfo->x_num[level]; }
    inline int y_num(const unsigned int level) { return genInfo->y_num[level]; }
    inline int z_num(const unsigned int level) { return genInfo->z_num[level]; }
};


#endif //LIBAPR_GENITERATOR_HPP
