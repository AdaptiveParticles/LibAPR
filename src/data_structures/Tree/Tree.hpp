

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
#include <algorithm>

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

    /*
     * Size of the model in number of particles that fit and number of possible indices
     */
    coords3d dims, dims_coords;

    /*
     * Number of layers without the root and the contents.
     */
    uint8_t depth;

    uint64_t * get_raw_tree()
    {
        return tree.data();
    }

    Content* get_raw_content()
    {
        return contents.data();
    }
    
    uint64_t get_tree_size(){
        return tree.size();
    }
    
    uint64_t get_content_size(){
        return contents.size();
    }
    

    Tree(const Particle_map<T> &particle_map, std::vector<uint64_t> &tree_mem, std::vector<Content> &contents_mem)
    {
        /** Constructs the tree and fills the content. Resizes the memory containers, so they fit the tree.
         *  The memory is moved with C++ moved semantics from the input variables to object fields.
         *
         * @param particle_map structure with local particles and downsample imput images
         * @param tree_mem     memory placeholder for the tree. Has to be able to contain whole tree without resizing.
         * @param contents_mem memory placeholder for the contents. Same as above.
         */
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

            coords3d new_coords = {(i & 2) != 0, (int)i % 2, i > 3};

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

    coords3d get_coordinates(uint64_t current)
    {
        /** Returns coordinates of a node pointed by the current index
         *  Complexity - O(log_{8}n) where n is number of nodes in the tree.
         *
         *  @param current index to the node
         */

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

    inline uint8_t get_status(uint64_t node) const
    {
        /** Returns status of the node - can be one of:
         *   PARENTSTATUS, TAKENSTATUS, NEIGHBOURSTATUS, SLOPESTATUS
         *
         *  @param node index to the node
         */
        
        return (uint8_t)(tree[node] & 0b1111);
    }

    inline uint64_t get_parent(uint64_t node) const
    {
        /** Returns index to the parent of the current node. In case there is no parent, returns the current node.
         *
         *  @param node index to the node
         */
        return node - tree[node + 1];
    }


    inline bool child_exists(uint8_t children, uint8_t current) const
    {
        /** Check if nth child of a node exists
         *
         *  @param children byte containing info about children
         *  @param current  index of a child
         */
        return (children & mask[current]) != 0;
    }


    inline uint8_t get_children(uint64_t node)
    {
        /** Get byte containing information about children
         *
         *  @param node index of a node
         */
        uint8_t children = tree[node] >> 8;

        return children;
    }

    Content& get_content(uint64_t index)
    {
        /** Get contents of a node. Undefined behaviour for nodes that are not leaves.
         *
         *  @param index index of a node
         */
        return contents[tree[index + 2]];
    }
    
    Content& get_content_part(uint64_t index,uint64_t part_index)
    {
        /** Get contents of a node. Undefined behaviour for nodes that are not leaves.
         *
         *  @param index index of a node
         */
        return contents[tree[index + 2] + part_index];
    }


    void get_face_neighbours(uint64_t index, uint8_t face, coords3d coords, uint16_t multiplier,
                             uint8_t child_index, std::vector<uint64_t> &result)
    {
        /** Get neighbours of a cell in one of the directions.
         *
         *  @param index       index of a node
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [-z,-x,-y,y,x,z]
         *  @param coords      coordinates of the node
         *  @param multiplier  half of the radius of the current cell ( * 2^(k_max - level + 2) diameter of the current cell in the final co_ordinate system (used for computing the coordinates) (Double the effective pixel grid))
         *  @param child_index what is the index of current node in parent's children
         *  @param result      placeholder for the result. Should have at least 4 elements.
         *                     Will contain result, and will be resized to correct size.
         */
        bool faces[6] = {0,0,0,0,0,0};
        faces[face] = true;
        get_neighbours_internal(index, coords, multiplier, child_index, result, faces);
    }

    void get_neighbours(uint64_t index, coords3d coords, uint16_t multiplier,
                        uint8_t child_index, std::vector<uint64_t> &result)
    {
        /** Get all neighbours of a cell.
         *
         *  @param index       index of a node
         *  @param coords      coordinates of the node
         *  @param multiplier  half of the radius of the current cell (  2^(k_max - level + 2) diameter of the current cell in the final co_ordinate system (used for computing the coordinates) (Double the effective pixel grid))
         *  @param child_index what is the index of current node in parent's children
         *  @param result      placeholder for the result. Should have 15 elements (4*3 + 3). Will contain result,
         *                     and will be resized to correct size.
         */
        bool faces[6] = {1,1,1,1,1,1};
        get_neighbours_internal(index, coords, multiplier, child_index, result, faces);
    }

    uint8_t get_child_index(uint64_t child, uint64_t parent)
    {
        uint64_t difference = child - parent;

        for(uint8_t i = 0; i < 8; i++)
        {
            if(tree[parent + 2 + i] == difference)
            {
                return i;
            }
        }

        throw std::runtime_error("get_child_index: parent is not a child of grandparent");
    }


    coords3d get_neighbour_coords(coords3d old, uint8_t index, uint16_t multiplier)
    {
        /** Shifts coords to a neighbour using the index of a face
         *
         *  @param old        coordinates before shifting
         *  @param index      the index of the face
         *  @param multiplier radius of the cell  ( 2^(k_max - level + 2) diameter of the current cell in the final co_ordinate system (used for computing the coordinates) (Double the effective pixel grid))
         *
         */

        old.y += von_neumann_y[index] * multiplier;
        old.x += von_neumann_x[index] * multiplier;
        old.z += von_neumann_z[index] * multiplier;
        return old;
    }

    coords3d shift_coords(coords3d old, uint8_t index, uint16_t multiplier)
    {
        /** Shifts coords to a child using the index of a child
         *
         *  @param old        coordinates before shifting
         *  @param index      the index of the child
         *  @param multiplier half of the radius of the parent's cell ( * 2^(k_max - level + 2) diameter of the current cell in the final co_ordinate system (used for computing the coordinates) (Double the effective pixel grid))
         *
         */

        old.y += y_shift[index] * multiplier;
        old.x += x_shift[index] * multiplier;
        old.z += z_shift[index] * multiplier;
        return old;
    }

    coords3d get_parent_coords(coords3d old, uint16_t multiplier, uint8_t child_index)
    {
        /** Get coordinates of the parent
         *
         * @param old         index of the current cell
         * @param multiplier  half of the radius of the current cell (  2^(k_max - level + 2) diameter of the current cell in the final co_ordinate system (used for computing the coordinates) (Double the effective pixel grid))
         * @param child_index what is the index of current node in parent's children
         */
        old.y -=  y_shift[child_index] * multiplier;
        old.x -=  x_shift[child_index] * multiplier;
        old.z -=  z_shift[child_index] * multiplier;
        return old;
    }
    
    void get_particle_coords(std::vector<coords3d>& part_coords, coords3d cell_coords,uint16_t level_multiplier, uint8_t status){
    
        /** Get coordinates of the particle in a cell
         *
         * @out   part_coords co_ordinates of the individual particles
         * @param cell_coords coordinates of the current cell
         * @param level_multiplier  half of the radius of the current cell (  2^(k_max - level + 2) diameter of the current cell in the final co_ordinate system (used for computing the coordinates) (Double the effective pixel grid))
         * @param status status of the current cell
         */
        
        
        switch(status){
            case PARENTSTATUS:
            {
                part_coords.resize(0);
                
                break;
            }
            case TAKENSTATUS:
            {
                part_coords.resize(8);
                level_multiplier /= 4;
                
                for (unsigned int i = 0; i < NUMBEROFSONS; i++){
                    part_coords[i] = { cell_coords.x + level_multiplier*(2*((i & 2) != 0)-1),
                        cell_coords.y + level_multiplier*(2*(((int)i % 2)-1)),cell_coords.z + level_multiplier*(2*(i > 3)-1)
                    };
                }

                break;
            }
            default:
            {
                part_coords.resize(1);
                part_coords[0] = cell_coords;
                break;
            }
        }
        
    }


private:
    std::vector<uint64_t> tree;
    /**
     * Structure of a node:
     *
     * 48 bits empty (to use later)
     * 8 bits showing which kids are filled (if a parent node)
     * 4 bits empty
     * 4 bits status
     * 64 bits difference to parent in 64bits - local to the current node
     * One of:
     *  8 * 64 bits of indices to child nodes (if parent node) - local to the current node
     *  64 bit index to contents array (otherwise)
     *
     * Explanation: local pointer is a difference between current node index and parent/child node index
     *              ex. current node index - parent local pointer = parent node index
     *                  current node index + child local pointer = children node index
     */

    std::vector<Content> contents;

    // Variables used when the tree is built
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

        /**
         *
         *  @param particle_map      particle map structure with statuses and downsampled
         *  @param level             the level of the parent
         *  @param position_in_level the position of the current node in the 3d array (using x,y,z)
         *  @param parent_index      index of the parent in the tree
         *  @param coords            coordinates of the current node
         *  @param cell_elements     diameter of the current cell
         */

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
                                           2 * coords.y + (int)i % 2,
                                           2 * coords.z + (i > 3)};

                    if(child_in_image(new_coords, cell_elements)) {
                        contents[current_context_index++] = {
                                particle_map.downsampled[level + 1].mesh[
                                        particle_map.downsampled[level + 1].index(new_coords)
                                ]
                        };
                    } else {
                        // out of boundary check, needs to take an intensity of the boundary of the image
                        new_coords = {std::min((uint16_t) new_coords.x,(uint16_t)(floor(dims.x/((float)cell_elements)))),std::min((uint16_t)new_coords.y,(uint16_t)(floor(dims.y/((float)cell_elements)))),std::min((uint16_t)new_coords.z,(uint16_t)(floor(dims.z/((float)cell_elements))))};
                        
                        contents[current_context_index++] = {
                            particle_map.downsampled[level + 1].mesh[
                                        particle_map.downsampled[level + 1].index(new_coords)]};
                        
                        
                        //contents[current_context_index + 1] = contents[current_context_index];
                        //current_context_index++;
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

