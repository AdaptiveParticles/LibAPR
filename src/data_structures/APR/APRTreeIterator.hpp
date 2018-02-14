//
// Created by cheesema on 14.02.18.
//

#ifndef LIBAPR_APRTREEITERATOR_HPP
#define LIBAPR_APRTREEITERATOR_HPP

#include "APRIterator.hpp"
#include "APRTree.hpp"

// APRIteration class, with extra methods designed for the use with APRTree

template<typename ImageType>
class APRTreeIterator : public APRIterator<ImageType> {
public:
    APRTreeIterator(APRTree<ImageType>& apr_tree){
        this->apr_access = &apr_tree.tree_access;
        this->current_particle_cell.global_index = UINT64_MAX;
        this->highest_resolution_type = 5;
    }

    void get_child();

    void get_partent();


};


#endif //LIBAPR_APRTREEITERATOR_HPP
