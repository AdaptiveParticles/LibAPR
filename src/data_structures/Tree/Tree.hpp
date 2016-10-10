

//
// Created by msusik on 26.09.16.
//

#ifndef PARTPLAY_TREE_HPP
#define PARTPLAY_TREE_HPP

#include "Content.hpp"
#include "../particle_map.hpp"
#include "../meshclass.h"

#include <bitset>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

#define PARENTSIZE 10
#define LEAFSIZE 3
#define NUMBEROFSONS 8

#define PARENTSTATUS EMPTY

#define ALLKIDS 0xFF00
#define NOPARENT 0
#define NOCHILD 0

#define ITERATOREND 1

template <typename T>
class LevelIterator;

template <typename T>
class Tree {
    friend class LevelIterator<T>;
public:

    std::vector<uint8_t> y_shift_half = {0,1,0,1,0,1,0,1};
    std::vector<uint8_t> x_shift_half = {0,0,1,1,0,0,1,1};
    std::vector<uint8_t> z_shift_half = {0,0,0,0,1,1,1,1};

    coords3d dims, dims_coords;
    uint8_t depth;

    uint64_t * get_raw_tree()
    {
        return tree.data();
    }

    Content* get_raw_content()
    {
        return contents.data();
    }

    Tree(const Particle_map<T> &particle_map, std::vector<uint64_t> &tree_mem, std::vector<Content> &contents_mem)
    {
        tree = std::move(tree_mem);
        contents = std::move(contents_mem);

        depth = particle_map.layers.size() - 1; // only tree, no content included
        dims.x = particle_map.layers.back().x_num * 2;
        dims.y = particle_map.layers.back().y_num * 2;
        dims.z = particle_map.layers.back().z_num * 2;
        dims_coords = dims * 2;

        tree[0] = NOCHILD; // and a parent as well!! - not true!!
        tree[1] = NOPARENT; // no parent

        size_t tofill = 2;
        size_t current = PARENTSIZE;

        current_tree_index = 10;

        uint16_t cell_elements = pow(2, depth);

        size_t kids = 0;
        int current_kid = 1;

        // here we can parallelize
        for(unsigned int i = 0; i < NUMBEROFSONS; i++)
        {

            coords3d new_coords = {(i & 2) != 0, i % 2, i > 3};

            tree[tofill + i] = current;


            if(child_in_image(new_coords, cell_elements))
            {
                current += fill_child(particle_map, 1, i, 0, new_coords, cell_elements / 2);
                // in order to parallelize the endindices have to be computed
                kids += current_kid;
            }
            else
            {
                tree[2 + i] = NOCHILD;
            }

            current_kid <<= 1;

        }

        tree[0] = kids * 256;
        tree.resize(current_tree_index);
        contents.resize(current_context_index);

    }

    coords3d get_coordinates(unsigned long current)
    {

        coords3d result = {1,1,1};
        uint16_t depth_multiplier = 2;
        uint8_t depth_reached = 1;
        unsigned long parent, child_distance;
        uint8_t child_index;

        do
        {
            child_distance = tree[current + 1];
            child_index = 0;
            parent = current - child_distance;

            do
            {
                if(tree[parent + 2 + child_index] == child_distance)
                {
                    break;
                }
            } while(++child_index);

            result.x += depth_multiplier * x_shift_half[child_index];
            result.y += depth_multiplier * y_shift_half[child_index];
            result.z += depth_multiplier * z_shift_half[child_index];
            depth_multiplier *= 2;
            depth_reached += 1;
            current = parent;

        } while(current != 0); // reached parent

        depth_multiplier = pow(2, depth - depth_reached);
        result.x *= depth_multiplier;
        result.y *= depth_multiplier;
        result.z *= depth_multiplier;

        return result;
    }
    // complexity - log8 (depth)

    uint8_t get_status(uint64_t node) const
    {
        return (uint8_t)(tree[node] & 0b1111);
    }

    uint64_t get_parent(uint64_t node) const
    {
        return node - tree[node + 1];
    }


    bool child_exists(uint8_t children, uint8_t current) const
    {
        return (children & mask[current]) != 0;
    }


    inline uint8_t get_children(uint64_t node)
    {

        uint8_t children = tree[node] >> 8;

        return children;
    }

    Content& get_content(uint64_t index)
    {
        // assumes index points to a leaf!
        return contents[tree[index + 2]];
    }


    void get_face_neighbours(uint64_t index, uint8_t face, coords3d coords, uint16_t multiplier,
                             uint8_t child_index, std::vector<uint64_t> &result)
    {
        bool faces[6] = {0,0,0,0,0,0};
        faces[face] = true;
        get_neighbours_internal(index, coords, multiplier, child_index, result, faces);
    }

    void get_neighbours(uint64_t index, coords3d coords, uint16_t multiplier,
                        uint8_t child_index, std::vector<uint64_t> &result)
    {
        bool faces[6] = {1,1,1,1,1,1};
        get_neighbours_internal(index, coords, multiplier, child_index, result, faces);
    }

    uint8_t get_child_index(uint64_t parent, uint64_t grandparent)
    {
        uint64_t difference = parent - grandparent;

        for(uint8_t i = 0; i < 8; i++)
        {
            if(tree[grandparent + 2 + i] == difference)
            {
                return i;
            }
        }

        throw std::runtime_error("get_child_index: parent is not a child of grandparent");
    }


    coords3d get_neighbour_coords(coords3d old, uint8_t index, uint16_t multiplier)
    {
        old.y += von_neumann_y[index] * multiplier;
        old.x += von_neumann_x[index] * multiplier;
        old.z += von_neumann_z[index] * multiplier;
        return old;
    }

    coords3d shift_coords(coords3d old, uint8_t index, uint16_t multiplier)
    {

        old.y += y_shift[index] * multiplier;
        old.x += x_shift[index] * multiplier;
        old.z += z_shift[index] * multiplier;
        return old;
    }

    coords3d get_parent_coords(coords3d old, uint16_t multiplier, uint8_t child_index)
    {
        old.y -=  y_shift[child_index] * multiplier;
        old.x -=  x_shift[child_index] * multiplier;
        old.z -=  z_shift[child_index] * multiplier;
        return old;
    }


private:
    std::vector<uint64_t> tree;
    /*
     * Structure of a node:
     *
     * 32 bits empty (to use later)
     * 8 bits showing which kids are filled (if a parent node)
     * 4 bits empty
     * 4 bits status
     * 64 bits difference to parent in 64bits - local to the current node
     * One of:
     *  8 * 64 bits of indices to child nodes (if parent node) - local to the current node
     *  8 * 64 bits of indices to contents array (if taken node)
     *  64 bit index to contents array (otherwise)
     */


    std::vector<Content> contents;

    size_t current_tree_index = 0;
    size_t current_context_index = 0;

    uint8_t mask[NUMBEROFSONS] = {1, 2, 4, 8, 16, 32, 64, 128};

    int8_t y_shift[8] = {-1,1,-1,1,-1,1,-1,1};
    int8_t x_shift[8] = {-1,-1,1,1,-1,-1,1,1};
    int8_t z_shift[8] = {-1,-1,-1,-1,1,1,1,1};

    int8_t von_neumann_y[6] = {0, 0, -1, 1, 0, 0};
    int8_t von_neumann_x[6] = {0, -1, 0, 0, 1, 0};
    int8_t von_neumann_z[6] = {-1, 0, 0, 0, 0, 1};

    int8_t von_neumann_children[6][4] = {
            {4,5,6,7}, {2,3,6,7}, {1,3,5,7}, {0,2,4,6}, {0,1,4,5}, {0,1,2,3}
    };


    size_t fill_child(const Particle_map<T> &particle_map,
                      uint8_t level, size_t position_in_level, size_t parent_index,
                      coords3d coords, uint16_t cell_elements)
    {

        // if no entries for level, it should be skipped

        size_t status;

        if(particle_map.layers[level].mesh.size())
        {
            status = particle_map.layers[level].mesh[position_in_level];

            if(status > NEIGHBOURSTATUS)
            {
                // ASCENDANT or ASCENDANTNEIGHBOUR
                status = PARENTSTATUS;
            }
        }
        else
        {
            status = PARENTSTATUS;
        }

        size_t index = current_tree_index;
        uint64_t current_index = 0;

        tree[current_tree_index] = status;
        tree[current_tree_index + 1] = index - parent_index;
        current_tree_index += 2;

        switch(status){
            case PARENTSTATUS:
            {
                current_tree_index += 8;
                size_t kids = 0;
                int current = 1;
                current_index += PARENTSIZE;
                for (uint8_t i = 0; i < NUMBEROFSONS; i++) {
                    // boundaries?
                    coords3d new_coords = {2 * coords.x + ((i & 2) != 0), 2 * coords.y + i % 2, 2 * coords.z + (i > 3)};

                    if (child_in_image(new_coords, cell_elements)) {
                        kids += current;
                        tree[index + 2 + i] = current_index;

                        current_index += fill_child(particle_map, level + 1,
                                                    particle_map.layers[level + 1].index(new_coords),
                                                    index, new_coords, cell_elements / 2);

                    } else {
                        tree[index + 2 + i] = NOCHILD;
                    }

                    current <<= 1;
                }
                tree[index] = tree[index] | (kids * 256);
                break;
            }
            case TAKENSTATUS:
            {
                tree[current_tree_index++] = current_context_index;
                for (unsigned int i = 0; i < NUMBEROFSONS; i++) {

                    coords3d new_coords = {2 * coords.x + ((i & 2) != 0),
                                           2 * coords.y + i % 2,
                                           2 * coords.z + (i > 3)};

                    if(child_in_image(new_coords, cell_elements)) {
                        contents[current_context_index++] = {
                                particle_map.downsampled[level + 1].mesh[
                                        particle_map.downsampled[level + 1].index(new_coords)
                                ]
                        };
                    } else {
                        contents[current_context_index + 1] = contents[current_context_index];
                        current_context_index++;
                    }
                }
                current_index += LEAFSIZE;
                break;
            }
            default:
            {

                tree[current_tree_index++] = current_context_index;
                contents[current_context_index++] = {
                        particle_map.downsampled[level].mesh[
                                particle_map.downsampled[level].index(coords)
                        ]
                };
                current_index += LEAFSIZE;
                break;
            }
        }

        return current_index;
    }

    bool child_in_image(coords3d new_coords, uint16_t cell_elements) const
    {
        new_coords = new_coords * cell_elements;
        return new_coords < dims;
    }

    bool coords_in_image(coords3d new_coords) const
    {
        return new_coords <= dims_coords;
    }



    void get_neighbours_internal(uint64_t index, coords3d coords, uint16_t multiplier,
                                 uint8_t child_index, std::vector<uint64_t> &result,
                                 bool faces[])
    {

        uint8_t elems = 0;

        for(int i = 0; i < 6; i++)
        {
            if(faces[i])
            {

                coords3d neighbour = get_neighbour_coords(coords, i, 2 * multiplier);
                coords3d parent_coords = get_parent_coords(coords, multiplier, child_index);

                if (coords_in_image(neighbour) && neighbour.x > 0 && neighbour.y > 0 && neighbour.z > 0) {
                    uint64_t parent = index - tree[index + 1];

                    if (parent_coords.contains(neighbour, multiplier)) {
                        // the neighbour has the same parent, find it and return (not true!!)

                        uint8_t neighbour_index = 4 * (neighbour.z > parent_coords.z) +
                                                  2 * (neighbour.x > parent_coords.x) +
                                                  (neighbour.y > parent_coords.y);
                        uint64_t sibling = parent + tree[parent + 2 + neighbour_index];
                        uint8_t status = get_status(sibling);
                        if (status != PARENTSTATUS) {
                            result[elems++] = sibling;
                        } else {
                            coords3d sibling_coords = {
                                    parent_coords.x + multiplier * (-1 + 2 * von_neumann_x[i]),
                                    parent_coords.y + multiplier * (-1 + 2 * von_neumann_y[i]),
                                    parent_coords.z + multiplier * (-1 + 2 * von_neumann_z[i])
                            };

                            add_neighbour_down(sibling, sibling_coords, neighbour, i,
                                               multiplier, 0, &elems, result);
                        }
                    } else {
                        add_neighbour(i, neighbour, parent, parent_coords, &elems, result, 1, multiplier * 2);
                    }

                }
            }
        }

        result.resize(elems);

    }

    void add_neighbour(uint8_t i, coords3d neighbour, uint64_t parent, coords3d parent_coords,
                       uint8_t *elems, std::vector<uint64_t> &result, uint8_t level_diff, uint16_t multiplier)
    {
        // compute grandpapa, eventually go to the children, etc.
        if(parent == NOPARENT)
        {
            return;
        }

        uint64_t grandparent = parent - tree[parent+1];

        uint8_t parent_child_index = get_child_index(parent, grandparent);
        coords3d grandparent_coords = get_parent_coords(parent_coords, multiplier, parent_child_index);
        if(grandparent_coords.contains(neighbour, multiplier*2))
        {
            uint8_t neighbour_index = 4 * (neighbour.z > grandparent_coords.z) +
                                      2 * (neighbour.x > grandparent_coords.x) +
                                      (neighbour.y > grandparent_coords.y);
            coords3d child_index = shift_coords(grandparent_coords, neighbour_index, multiplier);

            add_neighbour_down(grandparent + tree[grandparent + 2 + neighbour_index], child_index, neighbour, i,
                               multiplier, level_diff, elems, result);
        }
        else
        {
            add_neighbour(i, neighbour, grandparent, grandparent_coords,
                          elems, result, level_diff + 1, multiplier * 2);
        }

        // another function for going down
    }

    void add_neighbour_down(uint64_t parent, coords3d parent_coords, coords3d neighbour, uint8_t i,
                            uint16_t multiplier, uint8_t level_diff, uint8_t *elems, std::vector<uint64_t> &result)
    {
        uint8_t status = get_status(parent);

        switch(status)
        {
            case PARENTSTATUS:
            {
                if(level_diff == 0)
                {
                    uint8_t children = get_children(parent);

                    // return four children
                    // i: -z -x -y y x z
                    for(int j = 0; j < 4; j++) {
                        if(child_exists(children, von_neumann_children[i][j])) {
                            result[*elems] = parent + tree[parent + 2 + von_neumann_children[i][j]];
                            (*elems)++;
                        }
                    }

                }
                else
                {

                    uint8_t neighbour_index = 4 * (neighbour.z > parent_coords.z) +
                                              2 * (neighbour.x > parent_coords.x) +
                                              (neighbour.y > parent_coords.y);

                    coords3d child_index = shift_coords(parent_coords, neighbour_index, multiplier/2);

                    add_neighbour_down(parent + tree[parent + 2 + neighbour_index], child_index, neighbour, i,
                                       multiplier / 2, level_diff-1, elems, result);
                }

                break;
            }
            default:
            {
                // does not have children, so it has to be returned
                result[*elems] = parent;
                (*elems)++;
                break;
            }
        }
    }

};



#endif //PARTPLAY_TREE_HPP

