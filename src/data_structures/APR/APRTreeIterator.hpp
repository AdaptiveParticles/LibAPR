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
        this->highest_resolution_type = 8;
    }

    void set_iterator_to_child(APRIterator<ImageType>& current_iterator,uint8_t child_index){
        //empty
    };

    bool set_iterator_to_parent(APRIterator<ImageType>& current_iterator){
        //takes an input iterator and sets it THIS iterator to the parent of the particle cell that the current_iterator is pointing to.
        this->current_particle_cell.y = (uint16_t)(current_iterator.y()/2);
        this->current_particle_cell.x = (uint16_t)(current_iterator.x()/2);
        this->current_particle_cell.z = (uint16_t)(current_iterator.z()/2);
        this->current_particle_cell.level = (uint16_t)(current_iterator.level() - 1);

        if(this->current_particle_cell.level >= this->level_min()) {
            return this->set_iterator_by_particle_cell(this->current_particle_cell);
        } else {
            //is at a level lower then the set minimum level.
            return false;
        }

    }

    void set_particle_cell_no_search(APRIterator<ImageType>& current_iterator){
        this->current_particle_cell.y = (uint16_t)(current_iterator.y());
        this->current_particle_cell.x = (uint16_t)(current_iterator.x());
        this->current_particle_cell.z = (uint16_t)(current_iterator.z());
        this->current_particle_cell.level = (uint16_t)(current_iterator.level());
        this->set_neighbour_flag();

    }



};


#endif //LIBAPR_APRTREEITERATOR_HPP
