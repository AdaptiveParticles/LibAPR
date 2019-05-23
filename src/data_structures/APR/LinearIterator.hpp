//
// Created by cheesema on 2019-05-23.
//

#ifndef LIBAPR_LINEARITERATOR_HPP
#define LIBAPR_LINEARITERATOR_HPP

#include "APRAccessStructures.hpp"
#include "GenIterator.hpp"

class LinearIterator: public GenIterator {

    uint64_t current_index;

public:

    explicit LinearIterator(APRAccess& apr_access_) {
        this->apr_access = &apr_access_;
    }

    operator uint64_t() { return current_index; }

    //defining the iterator interface
    inline void operator++ (int){
        current_index++;
    }

    inline void operator++ (){
        current_index++;
    }

    inline uint64_t end(){
        return end_index;
    }

    inline uint64_t begin(const uint16_t level,const uint16_t z,const uint16_t x){
        const auto level_start = apr_access->linearAccess.level_xz_vec[level-1];
        const auto xz_start = level_start + x + z*x_num(level);

        //intialize
        current_index = apr_access->linearAccess.xz_end_vec[xz_start-1];
        end_index = apr_access->linearAccess.xz_end_vec[xz_start];

        return 0;
    }


};


#endif //LIBAPR_LINEARITERATOR_HPP
