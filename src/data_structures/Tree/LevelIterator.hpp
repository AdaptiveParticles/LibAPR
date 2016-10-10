//
// Created by msusik on 28.09.16.
//

#ifndef TREE_LEVELITERATOR_HPP
#define TREE_LEVELITERATOR_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "Tree.hpp"

template <typename T>
class LevelIterator {

    //complexity of iterating through a single layer - O(number of nodes in tree)

public:

    uint16_t level_multiplier;
    uint8_t  child_index;
    // change types to pair
    typedef uint64_t value_type;
    typedef ptrdiff_t difference_type;
    typedef uint64_t* pointer;
    typedef uint64_t& reference;
    typedef std::forward_iterator_tag iterator_category;

    LevelIterator(Tree<T> &tree): current(ITERATOREND), tree(tree) {}

    LevelIterator(Tree<T> &tree, int level) : tree(tree), level(level)
    {
        level_multiplier = pow(2, tree.depth);

        current_coords.x = current_coords.y = current_coords.z = level_multiplier * 2;
        current = 0;

        uint8_t current_level = 1;
        child_index = 0;

        if (level == 1) {
            // the parent is the only node at this level
            return;
        }


        uint8_t children = tree.tree[0] >> 8;

        for (uint8_t i = 0; i < 8; i++) {

            coords3d new_coords = tree.shift_coords(current_coords, i, level_multiplier);

            if (tree.child_exists(children, i) &&
                found_children(current + tree.tree[current + 2 + i], level - 1, level_multiplier / 2, new_coords)) {
                // the coordinates are set inside this function
                break;
            }

            if (i == 7) {
                // start = end?
                current = ITERATOREND;
            }
        }

    }

    LevelIterator(const LevelIterator& iterator)
            : tree(iterator.tree), level(iterator.level),
              current_coords(iterator.current_coords),
              current(iterator.current),
              level_multiplier(iterator.level_multiplier),
              child_index(iterator.child_index)
    {}

    LevelIterator &operator=(LevelIterator&& iterator)
    {
        this -> ~LevelIterator();
        new (this) LevelIterator (std::move(iterator));

        //std::cout << this->current << std::endl;
        return *this;
    }

    bool operator==(const LevelIterator &iterator)
    {
        return current == iterator.current;
    }

    bool operator!=(const LevelIterator &iterator)
    {
        return current != iterator.current;
    }


    uint64_t &operator*() {
        return current;
    }

    uint64_t *operator->() {
        return &current;
    }

    LevelIterator &operator++()
    {
        return (*this)++;
    }


    LevelIterator operator++(int)
    {

        // firstly check if sibling is available

        if(current == NOPARENT) {
            current = ITERATOREND;
            return *this;
        }

        uint64_t parent = current - tree.tree[current+1];

        uint8_t next_child = child_index + 1;

        coords3d parent_coords = tree.get_parent_coords(current_coords, level_multiplier, child_index);

        uint8_t children = tree.tree[parent] >> 8;

        while(next_child != 8)
        {
            if(tree.child_exists(children, next_child))
            {
                // great, sibling exits, return it!
                current = parent + tree.tree[parent + 2 + next_child];
                child_index = next_child;
                current_coords = tree.shift_coords(parent_coords, next_child, level_multiplier);
                return *this;
            }

            ++next_child;
        }

        // no sibling, going up the tree
        child_index = 0;
        return next_in_parent(parent, parent_coords, 2*level_multiplier, 2);
    }


    LevelIterator end(){
        LevelIterator result(tree);
        return result;
    }


    coords3d get_current_coords()
    {
        return current_coords;
    }

private:

    bool found_children(uint64_t child, uint8_t new_level, uint16_t multiplier, coords3d current_coords)
    {
        if(new_level == 1)
        {
            // it's here
            current = child;
            level_multiplier = multiplier ? 2*multiplier : 1;
            this->current_coords = current_coords;
            return true;
        }

        // otherwise check children
        uint8_t status = tree.get_status(child);

        switch(status){
            case PARENTSTATUS:
            {
                // we can go down!

                uint8_t children = tree.tree[child] >> 8;

                for(uint8_t i = 0; i < 8; i++)
                {
                    coords3d new_coords = tree.shift_coords(current_coords, i, multiplier);

                    if(tree.child_exists(children, i) &&
                       found_children(child + tree.tree[child + 2 + i], new_level - 1, multiplier / 2, new_coords))
                    {
                        // the coordinates are set inside this function
                        return true;
                    }
                }
            }
            default:
            {
                return false;
            }
        }
    }



    LevelIterator& next_in_parent(uint64_t parent, coords3d parent_coords,
                                  uint16_t multiplier, int local_depth)
    {

        if(parent == NOPARENT)
        {
            // no parent, thus the iteration has finished
            current = ITERATOREND;
            return *this;
        }

        uint64_t grandparent = parent - tree.tree[parent+1];

        uint8_t parent_child_index = tree.get_child_index(parent, grandparent);

        uint8_t next_child = parent_child_index + 1;
        coords3d grandparent_coords = tree.get_parent_coords(parent_coords, multiplier, parent_child_index);

        uint8_t children = tree.tree[grandparent] >> 8;

        while(next_child != 8)
        {
            if(tree.child_exists(children, next_child))
            {

                coords3d new_coords = tree.shift_coords(grandparent_coords, next_child, multiplier);

                // great, sibling exits, go down the tree
                if(found_children(grandparent + tree.tree[grandparent+ 2 + next_child],
                                  local_depth, multiplier / 2, new_coords)) {
                    return *this;
                }
            }

            ++next_child;
        }

        return next_in_parent(grandparent, grandparent_coords, multiplier * 2, local_depth + 1);
    }

    uint64_t current;
    uint8_t level;
    Tree<T> &tree;
    coords3d current_coords;
};

#endif //TREE_LEVELITERATOR_HPP
