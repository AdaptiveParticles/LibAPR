//
// Created by cheesema on 2019-05-23.
//

#ifndef LIBAPR_LINEARITERATOR_HPP
#define LIBAPR_LINEARITERATOR_HPP

#include "APRAccessStructures.hpp"
#include "GenIterator.hpp"
#include "LinearAccess.hpp"

class LinearIterator: public GenIterator {

    template<typename T>
    friend class PartCellData;

    uint64_t current_index;
    //uint64_t begin_index;

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

    operator uint64_t() override { return current_index; }

    uint16_t y() const override {
        return linearAccess->y_vec[current_index];
    }

    //defining the iterator interface
    inline void operator++ (int) override{
        current_index++;
    }

    inline void operator++ () override{
        current_index++;
    }

    inline uint64_t end() override{
        return end_index;
    }

    inline uint64_t begin(const uint16_t level_,const uint16_t z_,const uint16_t x_) override {

        level_start = linearAccess->level_xz_vec[level_]; //do i make these variables in the class
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
