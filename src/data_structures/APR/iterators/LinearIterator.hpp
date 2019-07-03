//
// Created by cheesema on 2019-05-23.
//

#ifndef LIBAPR_LINEARITERATOR_HPP
#define LIBAPR_LINEARITERATOR_HPP

#include "../access/APRAccessStructures.hpp"
#include "GenIterator.hpp"
#include "../access/LinearAccess.hpp"

class LinearIterator: public GenIterator {

    template<typename T>
    friend class PartCellData;

    template<typename T>
    friend class LazyData;

    uint64_t current_index;

    // TODO: need to add the level,x,z,y into here..
    // Also need to add the datageneration. (should make it lazy as possible)
    LinearAccess* linearAccess;

    uint64_t level_start = 0;
    uint64_t xz_start = 0;
    uint64_t level = 0;
    uint64_t offset = 0;
    uint64_t begin_index = 0;

public:

    explicit LinearIterator(LinearAccess& apr_access_,GenInfo& genInfo_) {
        linearAccess = &apr_access_;
        genInfo = &genInfo_;
        current_index = 0;
    }

    operator uint64_t()  { return current_index; }

    inline uint64_t global_index() const {
        return current_index;
    }

    uint16_t y() const  {
        return linearAccess->y_vec[current_index];
    }

    //Moves the iterator forward
    inline void operator++ (int) {
        current_index++;
    }

    //Moves the iterator forward (Note these are identical and only both including for convenience)
    inline void operator++ () {
        current_index++;
    }

    inline uint64_t end() {
        return end_index;
    }

    inline uint64_t particles_level_begin(const uint16_t& level_)  {
        //gives the begining index of a level
        auto level_start_ = linearAccess->level_xz_vec[level_];

        return linearAccess->xz_end_vec[level_start_-1];

    }

    inline uint64_t particles_level_end(const uint16_t& level_)  {
        // gives end index at that level, + 1;
        uint64_t index = linearAccess->level_xz_vec[level_] + x_num(level_) - 1 + (z_num(level_)-1)*x_num(level_);
        return linearAccess->xz_end_vec[index];
    }

    inline uint64_t total_number_particles(unsigned int level_=0){

        if(level_ == 0){
            level_ = level_max();
        }

        uint64_t index = linearAccess->level_xz_vec[level_] + x_num(level_) - 1 + (z_num(level_)-1)*x_num(level_);
        return linearAccess->xz_end_vec[index];
    }

    inline uint64_t begin(const uint16_t level_,const uint16_t z_,const uint16_t x_)  {
        //
        //  This initializes the iterator for the new row, by finding the starting index for both the particles and y data
        //

        level_start = linearAccess->level_xz_vec[level_];
        offset = x_ + z_*x_num(level_);
        xz_start = level_start + offset;
        level = level_;

        //intialize
        begin_index = linearAccess->xz_end_vec[xz_start-1];
        end_index = linearAccess->xz_end_vec[xz_start];
        current_index = begin_index;

        return current_index;
    }


};


#endif //LIBAPR_LINEARITERATOR_HPP
