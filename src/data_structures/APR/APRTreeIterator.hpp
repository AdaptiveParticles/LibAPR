//
// Created by cheesema on 14.02.18.
//

#ifndef LIBAPR_APRTREEITERATOR_HPP
#define LIBAPR_APRTREEITERATOR_HPP

#include "APRIterator.hpp"
#include "APRTree.hpp"

// APRIteration class, with extra methods designed for the use with APRTree

class APRTreeIterator : public APRIterator {
public:
    APRTreeIterator(APRTree &apr_tree) : APRIterator(apr_tree.tree_access, /*highest_resolution_type*/ 8) {}

    bool set_iterator_to_parent(APRIterator &current_iterator){
        //takes an input iterator and sets it THIS iterator to the parent of the particle cell that the current_iterator is pointing to.
        this->current_particle_cell.y = (uint16_t)floor(current_iterator.y()/2.0);
        this->current_particle_cell.x = (uint16_t)floor(current_iterator.x()/2.0);
        this->current_particle_cell.z = (uint16_t)floor(current_iterator.z()/2.0);
        this->current_particle_cell.level = (uint16_t)(current_iterator.level() - 1);

        if(this->current_particle_cell.level >= this->level_min()) {
            return this->set_iterator_by_particle_cell(this->current_particle_cell);
        } else {
            //is at a level lower then the set minimum level.
            return false;
        }
    }

    void set_particle_cell_no_search(APRIterator &current_iterator){
        this->current_particle_cell.y = (uint16_t)(current_iterator.y());
        this->current_particle_cell.x = (uint16_t)(current_iterator.x());
        this->current_particle_cell.z = (uint16_t)(current_iterator.z());
        this->current_particle_cell.level = (uint16_t)(current_iterator.level());
        this->set_neighbour_flag();

    }
};


#endif //LIBAPR_APRTREEITERATOR_HPP
