//
// Created by cheesema on 28.02.18.
//

#ifndef APR_TIME_APRSPARSEROW_HPP
#define APR_TIME_APRSPARSEROW_HPP

#include <algorithm>
#include <vector>


class APRSparseRow {

public:
    std::vector<uint16_t> y_begin;
    std::vector<uint16_t> y_end;
    std::vector<uint64_t> global_index;

    uint16_t y_0;
    uint16_t y_f;

    APRSparseRow():y_0(0),y_f(0){
    }

};

#endif //APR_TIME_APRSPARSEROW_HPP
