///////////////////
//
//  Bevan Cheeseman 2016
//
//  Part cell base class, specifies interface
//
///////////////

#ifndef PARTPLAY_PARTCELLSTRUCTURE_HPP
#define PARTPLAY_PARTCELLSTRUCTURE_HPP

#include "PartCellBase.hpp"
#include "../particle_map.hpp"
#include "../meshclass.h"

#include <vector>
#include <algorithm>

// Bit manipulation defitinitions
//masks for storing the neighbour indices (for 13 bit indices and 64 node)
#define TYPE_MASK ((1 << 0) - 1) << 1
#define TYPE_SHIFT 1
#define STATUS_MASK ((1 << 3) - 1) << 2
#define STATUS_SHIFT 2

//xp is x + 1 neigh
#define XP_DEPTH_MASK ((1 << 5) - 1) << 4
#define XP_DEPTH_SHIFT 4
#define XP_INDEX_MASK ((1 << 18) - 1) << 6
#define XP_INDEX_SHIFT 6
//xm is x - 1 neigh
#define XM_DEPTH_MASK ((1 << 20) - 1) << 19
#define XM_DEPTH_SHIFT 19
#define XM_INDEX_MASK ((1 << 33) - 1) << 21
#define XM_INDEX_SHIFT 21

#define ZP_DEPTH_MASK ((1 << 35) - 1) << 34
#define ZP_DEPTH_SHIFT 34
#define ZP_INDEX_MASK ((1 << 48) - 1) << 36
#define ZP_INDEX_SHIFT 36

#define ZM_DEPTH_MASK ((1 << 50) - 1) << 49
#define ZM_DEPTH_SHIFT 49
#define ZM_INDEX_MASK ((1 << 63) - 1) << 51
#define ZM_INDEX_SHIFT 51
//gap node defs

#define YP_DEPTH_MASK ((1 << 3) - 1) << 2
#define YP_DEPTH_SHIFT 2
#define YP_INDEX_MASK ((1 << 16) - 1) << 4
#define YP_INDEX_SHIFT 4

#define YM_DEPTH_MASK ((1 << 18) - 1) << 17
#define YM_DEPTH_SHIFT 17
#define YM_INDEX_MASK ((1 << 31) - 1) << 19
#define YM_INDEX_SHIFT 19

#define NEXT_COORD_MASK ((1 << 44) - 1) << 32
#define NEXT_COORD_SHIFT 32
#define PREV_COORD_MASK ((1 << 57) - 1) << 45
#define PREV_COORD_SHIFT 45

#define FREE_MEM_MASK ((1 << 63) - 1) << 58
#define FREE_MEM_SHIFT 58

//parent node defitions

template <typename T,typename S> // type T is the image type, type S is the data structure base type
class PartCellStructure: public PartCellBase<T,S> {
    
    
    
private:
    
    void get_neigh_in_dir(std::vector<Mesh_data<uint16_t>>& layer_index,coords3d coords,uint64_t* index,uint64_t* indicator,unsigned int level,const std::vector<int>& dir){
        //
        //  Bevan Cheeseman 2016
        //
        //  Get the index of the neighbour, and whether it is up or down
        //
        
        coords = {coords.x + dir[0],coords.y + dir[1],coords.z + dir[2]};
        
        uint16_t neigh = layer_index[level](coords.y,coords.x,coords.z);
        
        if(neigh > 0){
            *index = neigh;
            *indicator = 0;
            
        } else {
            //check parent (do first, smaller array, so faster)
            
            neigh = layer_index[level+1](ceil(coords.y/2),ceil(coords.x/2),ceil(coords.z/2.0));
            
            if(neigh > 0){
                //parent is your neighbour
                *index = neigh;
                *indicator = 1;
                
            } else {
                //add the first of the children, you can then raster from that point to your neighbour using the pointers given.
                *index = layer_index[level-1](2*coords.y-((dir[0] ==0) | (dir[0] == 1)),2*coords.x -((dir[1] ==0) | (dir[1] == 1)),2*coords.z -((dir[2] ==0) | (dir[2] == 1)));
                *indicator = 2;
            }
            
        }
        
        
    }
    
    void compute_map_key(std::vector<Mesh_data<std::vector<uint64_t>>>& new_map,std::vector<Mesh_data<uint16_t>>& layers_large,coords3d& coords,unsigned int depth){
        //
        //
        //  Bevan Cheeseman 2016
        //
        //
        //  Computes the neighbour indices, and up down key for the access tree.
        //
        //   (type 2 bit) (status 2 bit) (+-= y forward (2bits)) (+-=, x up coord 15bits), (+-=, x down coord 15bits) (+-=, z up coord 15bits), (+-=, z down coord 15bits) = 64
        //
        //   Allows for cell indices up to 13 bits = 8192, therefore, maximum single dim of 16384. (17.6 TB full image size)
        //
        //   status is stored in an 8bit or less array
        //
        
        uint64_t node_val = 0;
        
        uint64_t index = 0;
        uint64_t indicator = 0;
        
        std::vector<int> dir = {0,0,0};
        
        //set type
        node_val = 1;
        
        //xp
        dir = {1,0,0};
        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
        
        //shift bits
        index = index << XP_INDEX_SHIFT;
        node_val = node_val | index;
        
        indicator = indicator << XP_DEPTH_SHIFT;
        node_val = node_val | indicator;
        
        //xm
        dir = {-1,0,0};
        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
        //shift bits
        index = index << XM_INDEX_SHIFT;
        node_val = node_val | index;
        
        indicator = indicator << XM_DEPTH_SHIFT;
        node_val = node_val | indicator;
        
        //zp
        dir = {0,0,1};
        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
        //shift bits
        index = index << ZP_INDEX_SHIFT;
        node_val = node_val | index;
        
        indicator = indicator << ZP_DEPTH_SHIFT;
        node_val = node_val | indicator;
        
        //zm
        dir = {0,0,-1};
        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
        //shift bits
        index = index << ZM_INDEX_SHIFT;
        node_val = node_val | index;
        
        indicator = indicator << ZM_DEPTH_SHIFT;
        node_val = node_val | indicator;
        
        //status is still unset
        
        //add value to structure
        new_map[depth](coords.z,coords.x,1).push_back(node_val);
        
        
        ////////////////////////////////
        //Y direction back case check
        dir = {0,-1,0};
        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
        
        if(indicator!=0){
            //then get the previous node value (which should be a gap node)
            node_val = new_map[depth](coords.z,coords.x,1)[new_map[depth](coords.z,coords.x,1).size()-2];
            
            index = index << YM_INDEX_SHIFT;
            node_val = node_val | index;
            
            indicator = indicator << YM_DEPTH_SHIFT;
            node_val = node_val | indicator;
            
            //previous coord
            index = coords.y;
            node_val = node_val | (index << NEXT_COORD_SHIFT);
            
            new_map[depth](coords.z,coords.x,1)[new_map[depth](coords.z,coords.x,1).size()-2] = node_val;
            
        }
        
        ////////////////////////////////
        //Y direction forward case check
        
        //yp
        dir = {0,1,0};
        get_neigh_in_dir(layers_large,coords,&index,&indicator,depth,dir);
        
        if(indicator!=0){
            node_val = 0;
            index = index << YP_INDEX_SHIFT;
            node_val = node_val | index;
            
            indicator = indicator << YP_DEPTH_SHIFT;
            node_val = node_val | indicator;
            
            //previous coord
            index = coords.y;
            node_val = node_val | (index << PREV_COORD_SHIFT);
            
            //add the gap node
            new_map[depth](coords.z,coords.x,1).push_back(node_val);
        }
    }
    
    
    
    void create_sparse_graph_format(Particle_map<T>& part_map){
        //
        // Playing with a new data-structure (access paradigm)
        //
        //
        
        
        //need larger layers structure
        std::vector<Mesh_data<uint16_t>> layers_large;
        layers_large.resize(part_map.layers.size());
        
        for(int i = 0;i < part_map.layers.size();i++){
            layers_large[i].initialize(part_map.layers[i].y_num,part_map.layers[i].x_num,part_map.layers[i].z_num,0);
        }
        
        
        //then loop over the structure and create the indices
        
        uint16_t curr_index;
        coords3d curr_coords;
        
        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
            
            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
                for(int x_ = 0;x_ < part_map.layers[i].x_num;x_++){
                    
                    curr_index = 0;
                    
                    for(int y_ = 0;y_ < part_map.layers[i].y_num;y_++){
                        if(part_map.layers[i](y_,x_,z_) > 0){
                            curr_index++;
                            layers_large[i](y_,x_,z_)=curr_index;
                        }
                    }
                }
            }
        }
        
        
        
        //then create the new structure
        
        std::vector<Mesh_data<std::vector<uint64_t>>> new_map;
        
        new_map.resize(part_map.k_max+1);
        
        int count = 0;
        
        for(int i = part_map.k_min;i < part_map.k_max +1 ;i++){
            new_map[i].set_size(part_map.layers[i].z_num,part_map.layers[i].x_num,1);
            new_map[i].mesh.resize(part_map.layers[i].z_num*part_map.layers[i].x_num);
            
            for(int z_ = 0;z_ < part_map.layers[i].z_num;z_++){
                for(int x_ = 0;x_ < part_map.layers[i].x_num;x_++){
                    for(int y_ = 0;y_ < part_map.layers[i].y_num;y_++){
                        if(part_map.layers[i](y_,x_,z_) > 0){
                            count++;
                            
                            curr_coords = {x_,y_,z_};
                            
                            compute_map_key(new_map,layers_large,curr_coords,i);
                            
                            
                        }
                    }
                }
            }
        }
        
        
        
        
        
        std::cout << "New Structure Size Estmate: " << (count*8.0/1000000.0) << " MB" << std::endl;
        
    }

public:
    
    //decleration
    void initialize_structure(Particle_map<T>& particle_map){
        
        
        create_sparse_graph_format(particle_map);
        
        
        uint64_t test_int = 0;
        
        //test bits
        uint64_t x_coord_p = 153;
        uint64_t x_coord_p_ind = 1;
        
        //take the number and shift it
        
        
        uint64_t mask1;
        uint64_t mask2;
        uint64_t startBit = 2;
        uint64_t X = 15;
        
        mask1 = ((1 << X) - 1) << startBit;
        
        x_coord_p = x_coord_p << startBit;
        
        std::cout << x_coord_p << std::endl;
        
        test_int = x_coord_p | x_coord_p_ind;
        
        std::cout << test_int << std::endl;
        
        uint64_t out_int_1 = (test_int & mask1) >> startBit;
        
        std::cout << out_int_1 << std::endl;
        
        mask2 = ((1 << 1) - 1) << 0;
        
        uint64_t out_int_2 = (test_int & mask2);
        
        std::cout << out_int_2 << std::endl;
        
        
    }
    
    
    
    PartCellStructure(Particle_map<T> &particle_map){
        //
        //  initialization of the tree structure
        //
        
        initialize_structure(particle_map);
    }

    
    
};





#endif //PARTPLAY_PARTCELLSTRUCTURE_HPP