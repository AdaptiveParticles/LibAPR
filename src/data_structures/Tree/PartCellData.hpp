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
    
    
    PartCellKey get_neighbour_same_level(PartCellKey& curr_key,uint8_t face)
    {
        /** Get neighbours of a cell in one of the direction that are guranteed to be on the same level
         *
         *  @param curr_key    input: current key, output: neighbour key        
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [+y,-y,+x,-x,+z,-z] //change this ordering.. (y+ y-) are different,
         */
        
        //inits
        uint64_t node_val;
        
        PartCellKey neigh_key;
        
        //this is restricted to cells on the same level
        neigh_key.depth = neigh_key.depth;
        
        //get the node_val
        if(face < 2){
            //y_values need to use next node
            neigh_key.x = curr_key.x;
            neigh_key.z = curr_key.z;
            neigh_key.j = curr_key.j + von_neumann_y_cells[face];
            
        } else {
            //otherwise
            
            //get the node value
            node_val = data[curr_key.depth][x_num[curr_key.depth]*curr_key.z + curr_key.x][curr_key.j];
            
            neigh_key.j = ((node_val & index_mask_dir[face]) >> index_shift_dir[face]);
            
            neigh_key.x = curr_key.x + von_neumann_x_cells[face];
            neigh_key.z = curr_key.z + von_neumann_z_cells[face];
            
        }
        
        return neigh_key;

    }
    
    void get_neighs_face(PartCellKey& curr_key,uint64_t node_val, uint8_t face,std::vector<PartCellKey>& neigh_keys){
        //
        //  Bevan Cheeseman (2016)
        //
        //  Get all the nieghbours in direction face
        //
        /** Get neighbours of a cell in one of the direction
         *
         *  @param curr_key    input: current key, output: neighbour key
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [+y,-y,+x,-x,+z,-z] //change this ordering.. (y+ y-) are different,
         */
        //
        

        uint64_t neigh_indicator;
        
        PartCellKey neigh_key;
        
        // +-y direction is different
        if(face < 2){
            
            node_val = data[curr_key.depth][x_num[curr_key.depth]*curr_key.z + curr_key.x][curr_key.j + von_neumann_y_cells[face]];

            if(!(node_val&1)){
                //same level
                neigh_key.x = curr_key.x;
                neigh_key.z = curr_key.z;
                neigh_key.depth = curr_key.depth;
                neigh_key.j = curr_key.j + von_neumann_y_cells[face];
                return;
            }
            
        }
        
        //dir
        neigh_indicator = (node_val & depth_mask_dir[face]) >> depth_shift_dir[face];
        
        switch(neigh_indicator){
            case(LEVEL_SAME):{
                //same level return single neighbour
                neigh_key.j = ((node_val & index_mask_dir[face]) >> index_shift_dir[face]);
                neigh_key.x = curr_key.x + von_neumann_x_cells[face];
                neigh_key.z = curr_key.z + von_neumann_z_cells[face];
                
                neigh_key.depth = curr_key.depth;
                
                neigh_keys.push_back(neigh_key);
                
                return;
            }
            case(LEVEL_UP):{
                // neighbour is higher level
                neigh_key.j = ((node_val & index_mask_dir[face]) >> index_shift_dir[face]);
                
                neigh_key.x = (curr_key.x + von_neumann_x_cells[face])/2;
                neigh_key.z = (curr_key.z + von_neumann_z_cells[face])/2;
                
                neigh_key.depth = curr_key.depth - 1;
                
                neigh_keys.push_back(neigh_key);
                
                return;
            }
            case(LEVEL_DOWN):{
                //first of four children
                neigh_key.j = ((node_val & index_mask_dir[face]) >> index_shift_dir[face]);
                
                neigh_key.x = 2*(curr_key.x + von_neumann_x_cells[face]) -(von_neumann_x_cells[face] > -1);
                neigh_key.z = 2*(curr_key.z + von_neumann_z_cells[face]) -(von_neumann_z_cells[face] > -1);
                
                neigh_key.depth = curr_key.depth + 1;
                
                neigh_keys.push_back(neigh_key);
                
                //three other neighbour children
                neigh_key = get_neighbour_same_level(neigh_key,neigh_child_dir[face][0]);
                neigh_keys.push_back(neigh_key);
                
                neigh_key = get_neighbour_same_level(neigh_key,neigh_child_dir[face][1]);
                neigh_keys.push_back(neigh_key);
                
                neigh_key = get_neighbour_same_level(neigh_key,neigh_child_dir[face][2]);
                neigh_keys.push_back(neigh_key);
            
                return;
            }
        }

        
        
    }
    
    
    void test_get_neigh_dir(){
        //
        // Test the get neighbour direction code for speed
        //
        
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        uint64_t node_val;
        
        
        Part_timer timer;
        
        timer.verbose_flag = 1;
        
        PartCellKey curr_key;
        std::vector<PartCellKey> neigh_keys;
        
        timer.start_timer("get neighbour cells ");
        
        for(int i = depth_min;i <= depth_max;i++){
            
            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,neigh_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val = data[i][offset_pc_data][j_];
                        
                        if (node_val&1){
                            //get the index gap node
                            curr_key.x = x_;
                            curr_key.z = z_;
                            curr_key.depth = i;
                            curr_key.j = j_;
                            
                            neigh_keys.resize(0);
                            
                            get_neighs_face(curr_key,node_val,0,neigh_keys);
                            get_neighs_face(curr_key,node_val,1,neigh_keys);
                            get_neighs_face(curr_key,node_val,2,neigh_keys);
                            get_neighs_face(curr_key,node_val,3,neigh_keys);
                            get_neighs_face(curr_key,node_val,4,neigh_keys);
                            get_neighs_face(curr_key,node_val,5,neigh_keys);
                     
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
        
    }


private:
    //  Direction definitions for Particle Cell Neighbours
    //  [+y,-y,+x,-x,+z,-z]
    //  [0,1,2,3,4,5]
    
    uint64_t depth_mask_dir[6] = {YP_DEPTH_MASK,YM_DEPTH_MASK,XP_DEPTH_MASK,XM_DEPTH_MASK,ZP_DEPTH_MASK,ZM_DEPTH_MASK};
    uint64_t depth_shift_dir[6] =  {YP_DEPTH_SHIFT,YM_DEPTH_SHIFT,XP_DEPTH_SHIFT,XM_DEPTH_SHIFT,ZP_DEPTH_SHIFT,ZM_DEPTH_SHIFT};
    
    uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
    uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
    
    int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
    int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
    int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};
    
    //the ordering of retrieval of four neighbour cells
    uint8_t neigh_child_dir[6][3] = {{4,2,5},{4,2,5},{0,4,1},{0,4,1},{0,2,1},{0,2,1}};

};

#endif //PARTPLAY_PARTCELLDATA_HPP