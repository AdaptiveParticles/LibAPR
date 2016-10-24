///////////////////
//
//  Bevan Cheeseman 2016
//
//  PartCellData class, the data container for CRS sparse format for APR
//
///////////////

#ifndef PARTPLAY_PARTCELLDATA_HPP
#define PARTPLAY_PARTCELLDATA_HPP

#include <stdint.h>
// Bit manipulation defitinitions
//masks for storing the neighbour indices (for 13 bit indices and 64 node)
#define TYPE_MASK ((((uint64_t)1) << 2) - 1)
#define TYPE_SHIFT 1
#define STATUS_MASK ((((uint64_t)1) << 2) - 1) << 2
#define STATUS_SHIFT 2

//xp is x + 1 neigh
#define XP_DEPTH_MASK ((((uint64_t)1) << 2) - 1) << 4
#define XP_DEPTH_SHIFT 4
#define XP_INDEX_MASK ((((uint64_t)1) << 13) - 1) << 6
#define XP_INDEX_SHIFT 6
//xm is x - 1 neigh
#define XM_DEPTH_MASK ((((uint64_t)1) << 2) - 1) << 19
#define XM_DEPTH_SHIFT 19
#define XM_INDEX_MASK ((((uint64_t)1) << 13) - 1) << 21
#define XM_INDEX_SHIFT 21

#define ZP_DEPTH_MASK ((((uint64_t)1) << 2) - 1) << 34
#define ZP_DEPTH_SHIFT 34
#define ZP_INDEX_MASK ((((uint64_t)1) << 13) - 1) << 36
#define ZP_INDEX_SHIFT 36

#define ZM_DEPTH_MASK  ((((uint64_t)1) << 2) - 1) << 49
#define ZM_DEPTH_SHIFT 49
#define ZM_INDEX_MASK ((((uint64_t)1) << 13) - 1) << 51
#define ZM_INDEX_SHIFT 51
//gap node defs

#define YP_DEPTH_MASK ((((uint64_t)1) << 2) - 1) << 2
#define YP_DEPTH_SHIFT 2
#define YP_INDEX_MASK ((((uint64_t)1) << 13) - 1) << 4
#define YP_INDEX_SHIFT 4

#define YM_DEPTH_MASK ((((uint64_t)1) << 2) - 1) << 17
#define YM_DEPTH_SHIFT 17
#define YM_INDEX_MASK ((((uint64_t)1) << 13) - 1) << 19
#define YM_INDEX_SHIFT 19

#define NEXT_COORD_MASK ((((uint64_t)1) << 13) - 1) << 32
#define NEXT_COORD_SHIFT 32
#define PREV_COORD_MASK ((((uint64_t)1) << 13) - 1) << 45
#define PREV_COORD_SHIFT 45

#define FREE_MEM_MASK ((((uint64_t)1) << 6) - 1) << 58
#define FREE_MEM_SHIFT 58

//Neighbour definitions
#define NO_NEIGHBOUR 4
#define LEVEL_SAME 1
#define LEVEL_DOWN 0
#define LEVEL_UP 2

//Define Status definitions
#define SEED 1
#define BOUNDARY 2
#define FILLER 3

#define SEED_SHIFTED (uint64_t)1 << 2
#define BOUNDARY_SHIFTED (uint64_t)2 << 2
#define FILLER_SHIFTED (uint64_t)3 << 2

#include "PartCellKey.hpp"
#include "../particle_map.hpp"

template <typename T> // type T data structure base type
class PartCellData {
    
public:
    
    /*
     * Number of layers without the root and the contents.
     */
    uint8_t depth_max;
    uint8_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    
    std::vector<std::vector<std::vector<T>>> data;
    
    PartCellData(){};
    
    T& operator ()(int depth, int x_,int z_,int j_){
        // data access
        return data[depth][x_num[depth]*z_ + x_][j_];
    }
    
    
    void push_back(int depth, int x_,int z_,T val){
        data[depth][x_num[depth]*z_ + x_].push_back(val);
    }
    
    
    T& operator ()(const PartCellKey& key){
        // data access
        return data[key.depth][x_num[key.depth]*key.z + key.x][key.j];
    }
    
    template<typename S>
    void initialize_base_structure(Particle_map<S>& part_map){
        //initializes the partcell data structure based on part_map size
        
        //first add the layers
        depth_max = part_map.k_max;
        depth_min = part_map.k_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        
        data.resize(depth_max+1);
        
        for(int i = depth_min;i <= depth_max;i++){
            z_num[i] = part_map.layers[i].z_num;
            x_num[i] = part_map.layers[i].x_num;
            data[i].resize(z_num[i]*x_num[i]);
        }
        
        
    }
    
    template<typename S>
    void initialize_from_partcelldata(PartCellData<S>& part_cell_data){
        //initializes the partcell data structure based on part_map size
        
        //first add the layers
        depth_max = part_cell_data.depth_max;
        depth_min = part_cell_data.depth_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        
        data.resize(depth_max+1);
        
        for(int i = depth_min;i <= depth_max;i++){
            z_num[i] = part_cell_data.z_num[i];
            x_num[i] = part_cell_data.x_num[i];
            data[i].resize(z_num[i]*x_num[i]);
        }

    }

    void debug_node(T node_val){
        //
        // Gets all properties of the node
        //
        
        if (node_val&1){
            T type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
            T yp_index = (node_val & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
            T yp_depth = (node_val & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;

            T ym_index = (node_val & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
            T ym_depth = (node_val & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;

            T next_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
            
            T prev_coord = (node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT;
            
            if(prev_coord>next_coord){
                int stop =1;
            }
            
            int stop = 1;
        } else {
        
            T status = (node_val & STATUS_MASK) >> STATUS_SHIFT;
            T type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
            T xp_index = (node_val & XP_INDEX_MASK) >> XP_INDEX_SHIFT;
            T xp_depth = (node_val & XP_DEPTH_MASK) >> XP_DEPTH_SHIFT;
            T zp_index = (node_val & ZP_INDEX_MASK) >> ZP_INDEX_SHIFT;
            T zp_depth = (node_val & ZP_DEPTH_MASK) >> ZP_DEPTH_SHIFT;
            T xm_index = (node_val & XM_INDEX_MASK) >> XM_INDEX_SHIFT;
            T xm_depth = (node_val & XM_DEPTH_MASK) >> XM_DEPTH_SHIFT;
            T zm_index = (node_val & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
            T zm_depth = (node_val & ZM_DEPTH_MASK) >> ZM_DEPTH_SHIFT;
            
            int stop = 1;
        }
        
        
        
    }
    
private:
    
    
                           
    
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP