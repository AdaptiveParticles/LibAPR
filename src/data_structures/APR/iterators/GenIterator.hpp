//
// Created by cheesema on 29.06.18.
//

#ifndef LIBAPR_GENITERATOR_HPP
#define LIBAPR_GENITERATOR_HPP

#include "../access/RandomAccess.hpp"

class GenIterator {

protected:

    //Pointer to the actuall access information used by the iterator

    uint64_t end_index = 0;

    GenInfo *genInfo;

public:

    uint64_t total_number_particles();


//    uint16_t y() const {return 0;};
//
//    operator uint64_t() {return 0;};

    //helpers
    inline int x_nearest_pixel(int level, int x);
    inline float x_global(int level, int x);
    inline int y_nearest_pixel(int level, int y);
    inline float y_global(int level, int y);
    inline int z_nearest_pixel(int level, int z);
    inline float z_global(int level, int z);

    int level_size(int level);

    int org_dims(int dim) const { return genInfo->org_dims[dim]; }

    int number_dimensions(){return genInfo->number_dimensions;};

    int level_min();
    int level_max();

    inline int x_num(const unsigned int level){
        return genInfo->x_num[level];
    }

    inline int y_num(const unsigned int level){
        return genInfo->y_num[level];
    }

    inline int z_num(const unsigned int level){
        return genInfo->z_num[level];
    }



};


inline uint64_t GenIterator::total_number_particles() {return genInfo->total_number_particles;}


inline int GenIterator::level_size(const int level){
    return genInfo->level_size[level];
}

inline int GenIterator::x_nearest_pixel(const int level,const int x ){
    return (int) floor((x+0.5)*level_size(level));
}

inline float GenIterator::x_global(const int level,const int x) {
    return  (x+0.5f)*level_size(level);
}

inline int GenIterator::y_nearest_pixel(const int level,const int y ){
    return (int) floor((y+0.5)*level_size(level));
}

inline float GenIterator::y_global(const int level,const int y) {
    return  (y+0.5f)*level_size(level);
}

inline int GenIterator::z_nearest_pixel(const int level,const int z ){
    return (int) floor((z+0.5)*level_size(level));
}

inline float GenIterator::z_global(const int level,const int z) {
    return  (z+0.5f)*level_size(level);
}

inline int GenIterator::level_min() {return genInfo->l_min;}
inline int GenIterator::level_max() {return genInfo->l_max;}


#endif //LIBAPR_GENITERATOR_HPP
