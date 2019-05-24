//
// Created by cheesema on 29.06.18.
//

#ifndef LIBAPR_GENITERATOR_HPP
#define LIBAPR_GENITERATOR_HPP

#include "APRAccess.hpp"

class GenIterator {

protected:

    //Pointer to the actuall access information used by the iterator
    APRAccess* apr_access;
    uint64_t end_index = 0;

public:

    uint64_t total_number_particles();

    uint64_t particles_level_begin(const uint16_t& level_);
    uint64_t particles_level_end(const uint16_t& level_);

    virtual uint16_t y() const {return 0;};

    virtual operator uint64_t() {return 0;};

    //helpers
    inline unsigned int x_nearest_pixel( unsigned int level, int x);
    inline float x_global( unsigned int level, int x);
    inline unsigned int y_nearest_pixel( unsigned int level, int y);
    inline float y_global( unsigned int level, int y);
    inline unsigned int z_nearest_pixel( unsigned int level, int z);
    inline float z_global( unsigned int level, int z);

    unsigned int level_size( unsigned int level);

    uint16_t level_min();
    uint16_t level_max();

    inline uint64_t x_num(const unsigned int level){
        return apr_access->x_num[level];
    }

    inline uint64_t y_num(const unsigned int level){
        return apr_access->y_num[level];
    }

    inline uint64_t z_num(const unsigned int level){
        return apr_access->z_num[level];
    }

    //defining the iterator interface
    virtual inline void operator++ (int){
    }

    virtual inline void operator++ (){
    }

    virtual inline uint64_t end(){
        return 0;
    }

    virtual inline uint64_t begin(const uint16_t level,const uint16_t z,const uint16_t x){
        return 0;
    }


};


inline uint64_t GenIterator::total_number_particles() {return apr_access->total_number_particles;}

inline uint64_t GenIterator::particles_level_begin(const uint16_t& level_) {return apr_access->global_index_by_level_and_zx_end[level_-1].back()+1;}
inline uint64_t GenIterator::particles_level_end(const uint16_t& level_) {return apr_access->global_index_by_level_and_zx_end[level_].back();}

inline unsigned int GenIterator::level_size(const unsigned int level){
    return apr_access->level_size[level];
}

inline unsigned int GenIterator::x_nearest_pixel(const unsigned int level,const int x ){
    return (unsigned int) floor((x+0.5)*level_size(level));
}

inline float GenIterator::x_global(const unsigned int level,const int x) {
    return  (x+0.5f)*level_size(level);
}

inline unsigned int GenIterator::y_nearest_pixel(const unsigned int level,const int y ){
    return (unsigned int) floor((y+0.5)*level_size(level));
}

inline float GenIterator::y_global(const unsigned int level,const int y) {
    return  (y+0.5f)*level_size(level);
}

inline unsigned int GenIterator::z_nearest_pixel(const unsigned int level,const int z ){
    return (unsigned int) floor((z+0.5)*level_size(level));
}

inline float GenIterator::z_global(const unsigned int level,const int z) {
    return  (z+0.5f)*level_size(level);
}


inline uint16_t GenIterator::level_min() {return apr_access->l_min;}
inline uint16_t GenIterator::level_max() {return apr_access->l_max;}


#endif //LIBAPR_GENITERATOR_HPP
