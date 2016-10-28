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
#define NO_NEIGHBOUR ((uint64_t)3)
#define LEVEL_SAME ((uint64_t)1)
#define LEVEL_DOWN ((uint64_t)0)
#define LEVEL_UP ((uint64_t)2)

//Define Status definitions
#define SEED ((uint64_t)1)
#define BOUNDARY ((uint64_t)2)
#define FILLER ((uint64_t)3)
#define PARENT ((uint64_t)0)

//Type definitions
#define TYPE_PC ((uint64_t) 0)
#define TYPE_GAP ((uint64_t)1)
#define TYPE_GAP_END ((uint64_t)3)

#define SEED_SHIFTED ((uint64_t)1) << 2
#define BOUNDARY_SHIFTED ((uint64_t)2) << 2
#define FILLER_SHIFTED ((uint64_t)3) << 2

//Neighbour Keys

#define PC_KEY_DEPTH_MASK ((((uint64_t)1) << 5) - 1) << 0
#define PC_KEY_DEPTH_SHIFT 0
#define PC_KEY_X_MASK ((((uint64_t)1) << 13) - 1) << 5
#define PC_KEY_X_SHIFT 5

#define PC_KEY_Z_MASK ((((uint64_t)1) << 13) - 1) << 18
#define PC_KEY_Z_SHIFT 18
#define PC_KEY_J_MASK ((((uint64_t)1) << 13) - 1) << 31
#define PC_KEY_J_SHIFT 31

#define PC_KEY_INDEX_MASK ((((uint64_t)1) << 15) - 1) << 44
#define PC_KEY_INDEX_SHIFT 44



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
    
    T&  get_val(const uint64_t pc_key){
        // data access
        
        const uint64_t depth = (pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t j_ = (pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
        uint64_t shift = x_num[depth]*z_ + x_;
        
        return data[depth][x_num[depth]*z_ + x_][j_];
        
        //return data[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT][x_num[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT]*((pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + ((pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT)][(pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT];
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
    
    
    uint64_t get_neighbour_same_level(const uint64_t curr_key,const uint8_t face)
    {
        /** Get neighbours of a cell in one of the direction that are guranteed to be on the same level
         *
         *  @param curr_key    input: current key, output: neighbour key        
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [+y,-y,+x,-x,+z,-z] //change this ordering.. (y+ y-) are different,
         */
        
        
        //inits
        uint64_t node_val;
        uint64_t neigh_key;
        
        //this is restricted to cells on the same level
        neigh_key = curr_key;
        
        //get the node_val
        if(face < 2){
            //y_values need to use next node
            
            neigh_key &= -((PC_KEY_J_MASK) + 1);
            neigh_key |= (((neigh_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT) + von_neumann_y_cells[face]) << PC_KEY_J_SHIFT;
            
        } else {
            //otherwise
            
            //get the node value
            node_val = get_val(curr_key);
            
            //set the index
            neigh_key &= -((PC_KEY_J_MASK) + 1);
            neigh_key |= (((node_val & index_mask_dir[face]) >> index_shift_dir[face])) << PC_KEY_J_SHIFT;
            
            //neigh_key.x = curr_key.x + von_neumann_x_cells[face];
            //neigh_key.z = curr_key.z + von_neumann_z_cells[face];
            neigh_key &= -((PC_KEY_X_MASK) + 1);
            neigh_key |= (((neigh_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT) + von_neumann_x_cells[face]) << PC_KEY_X_SHIFT;
            neigh_key &= -((PC_KEY_Z_MASK) + 1);
            neigh_key |= (((neigh_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + von_neumann_z_cells[face]) << PC_KEY_Z_SHIFT;
            
            
            
        }
        
        return neigh_key;

    }
    
    //void get_neighs_face(PartCellKey& curr_key,uint64_t node_val, uint8_t face,std::vector<PartCellKey>& neigh_keys){
    void get_neighs_face(const uint64_t curr_key,uint64_t node_val,const uint8_t face,std::vector<uint64_t>& neigh_keys){
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
        
        uint64_t neigh_key;
        
        // +-y direction is different
        if(face < 2){
            
            node_val = get_val(curr_key);

            if(!(node_val&1)){
                //same level
                neigh_key = curr_key;
                
                neigh_key &= -((PC_KEY_J_MASK) + 1);
                neigh_key|=  (((neigh_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT) + von_neumann_y_cells[face]) << PC_KEY_J_SHIFT;
                
                neigh_keys.push_back(neigh_key);
                
                return;
            }
            
            
        }
        
        //dir
        neigh_indicator = (node_val & depth_mask_dir[face]) >> depth_shift_dir[face];
        
        switch(neigh_indicator){
            case(LEVEL_SAME):{
                //same level return single neighbour
                neigh_key = 0;
                neigh_key |= (((node_val & index_mask_dir[face]) >> index_shift_dir[face])) << PC_KEY_J_SHIFT;
                
                neigh_key |= (((curr_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT) + von_neumann_x_cells[face]) << PC_KEY_X_SHIFT;
                neigh_key |= (((curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + von_neumann_z_cells[face]) << PC_KEY_Z_SHIFT;
                
                neigh_keys.push_back(neigh_key);
                
                return;
            }
            case(LEVEL_DOWN):{
                // Neighbour is on parent level (depth - 1)
                
                neigh_key = 0;
                //get node index
                neigh_key |= (((node_val & index_mask_dir[face]) >> index_shift_dir[face])) << PC_KEY_J_SHIFT;
                
                //x/z coord shift
                neigh_key |= (((curr_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT) + von_neumann_x_cells[face])/2 << PC_KEY_X_SHIFT;
                neigh_key |= (((curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + von_neumann_z_cells[face])/2 << PC_KEY_Z_SHIFT;
                
                //depth shift
                neigh_key |= (((curr_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT) - 1) << PC_KEY_DEPTH_SHIFT;
                
                neigh_keys.push_back(neigh_key);
                
                
                return;
            }
            case(LEVEL_UP):{
                // Neighbour is on a lower child level
                
                //first of four children
                
                neigh_key = 0;
                
                //depth shift
                neigh_key |= (((curr_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT) +1) << PC_KEY_DEPTH_SHIFT;
                //get node index
                neigh_key |= (((node_val & index_mask_dir[face]) >> index_shift_dir[face])) << PC_KEY_J_SHIFT;
                
                //x/z coord shift
                neigh_key |= ((((curr_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT) + von_neumann_x_cells[face])*2 + (von_neumann_x_cells[face] < 0)) << PC_KEY_X_SHIFT;
                uint64_t temp = ((curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT);
                uint64_t temp2 = ((((curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + von_neumann_z_cells[face])*2 + (von_neumann_z_cells[face] < 0));
                neigh_key |= ((((curr_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + von_neumann_z_cells[face])*2 + (von_neumann_z_cells[face] < 0)) << PC_KEY_Z_SHIFT;
                

                
                neigh_keys.push_back(neigh_key);
                
                uint64_t temp3 = get_val(neigh_key);
                
                //three other neighbour children
                neigh_key = get_neighbour_same_level(neigh_key,neigh_child_dir[face][0]);
                
                neigh_keys.push_back(neigh_key);
                
                temp = get_val(neigh_key);
                
                neigh_key = get_neighbour_same_level(neigh_key,neigh_child_dir[face][1]);
                neigh_keys.push_back(neigh_key);
                
                temp = get_val(neigh_key);
                
                neigh_key = get_neighbour_same_level(neigh_key,neigh_child_dir[face][2]);
                neigh_keys.push_back(neigh_key);
                
                temp = get_val(neigh_key);
                
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
        
        uint64_t curr_key;
        //std::vector<PartCellKey> neigh_keys;
        std::vector<uint64_t> neigh_keys;
        
        
        timer.start_timer("get neighbour cells ");
        
        for(int i = depth_min;i <= depth_max;i++){
            
            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,neigh_keys) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                curr_key |= ((uint64_t)i) << PC_KEY_DEPTH_SHIFT;
                curr_key |= z_ << PC_KEY_Z_SHIFT;
                
                neigh_keys.reserve(24);
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    curr_key &=  -((PC_KEY_X_MASK) + 1);
                    curr_key |= x_ << PC_KEY_X_SHIFT;
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = data[i][offset_pc_data].size();

                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val = data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //get the index gap node
                            
                            curr_key &= -((PC_KEY_J_MASK) + 1);
                            curr_key |= j_ << PC_KEY_J_SHIFT;
                            
                            uint64_t test = get_val(curr_key);
                            
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
    
    void set_neighbor_relationships(uint8_t face){
        //
        //  Neighbour function for different face.
        //
        //
        //
        
        
        //x/z variables
        const uint64_t x_start = x_start_vec[face];
        const uint64_t x_stop = x_stop_vec[face];
        const uint64_t z_start = z_start_vec[face];
        const uint64_t z_stop = z_stop_vec[face];
        const int8_t x_offset = x_offset_vec[face];
        const int8_t z_offset = z_offset_vec[face];
        const uint64_t index_shift_0 = index_shift_dir[face];
        const uint64_t depth_shift_0 = depth_shift_dir[face];
        const uint64_t index_shift_1 = index_shift_dir_sym[face];
        const uint64_t depth_shift_1 = depth_shift_dir_sym[face];
        
        const uint64_t depth_mask_0 = depth_mask_dir[face];
        const uint64_t depth_mask_1 = depth_mask_dir_sym[face];
        
        //y variables
        
        const uint64_t next_prev_mask = next_prev_mask_vec[face];
        const uint64_t next_prev_shift= next_prev_shift_vec[face];
        const uint64_t y_offset = y_offset_vec[face];
        const uint64_t y_start = y_start_vec[face];
        const uint64_t y_stop = y_stop_vec[face];
        
        Part_timer timer;
        timer.verbose_flag = true;
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        
        timer.start_timer("Get neighbours dir: " + std::to_string(face));
        
        unsigned int y_neigh;
        unsigned int y_parent;
        uint64_t j_parent;
        uint64_t j_neigh;
        
        uint64_t node_val;
        uint64_t y_coord;
        
        if (face > 1){
            
            for(int i = (depth_min);i <= depth_max;i++){
                
                const unsigned int x_num_ = x_num[i];
                const unsigned int z_num_ = z_num[i];
                
                const unsigned int x_num_parent = x_num[i-1];
                
                if (i == depth_min){
                    
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord) if(z_num_*x_num_ > 100)
                    for(z_ = z_start;z_ < (z_num_-z_stop);z_++){
                        
                        for(x_ = x_start;x_ < (x_num_-x_stop);x_++){
                            
                            const size_t z_neigh = (z_+z_offset);
                            const size_t x_neigh = (x_+x_offset);
                            
                            const size_t offset_pc_data = x_num_*z_ + x_;
                            const size_t offset_pc_data_neigh = x_num_*z_neigh + x_neigh;
                            
                            //initialization
                            y_coord = (data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_neigh = (data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            
                            j_neigh = 1;
                            
                            if (data[i][offset_pc_data_neigh].size() == 1){
                                //set to max so its not checked
                                y_neigh = 64000;
                            }
                            
                            
                            y_coord--;
                            
                            const size_t j_num = data[i][offset_pc_data].size();
                            const size_t j_num_neigh = data[i][offset_pc_data_neigh].size();
                            
                            for(j_ = 1;j_ < j_num;j_++){
                                
                                // Parent relation
                                
                                node_val = data[i][offset_pc_data][j_];
                                
                                if (node_val&1){
                                    //get the index gap node
                                    y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                    y_coord--;
                                    
                                } else {
                                    //normal node
                                    y_coord++;
                                    
                                    while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                        
                                        j_neigh++;
                                        node_val = data[i][offset_pc_data_neigh][j_neigh];
                                        
                                        if (node_val&1){
                                            //get the index gap node
                                            y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                            j_neigh++;
                                            
                                        } else {
                                            //normal node
                                            y_neigh++;
                                        }
                                        
                                    }
                                    
                                    
                                    if(y_coord == y_neigh){
                                        data[i][offset_pc_data][j_] |= (j_neigh << index_shift_0);
                                        data[i][offset_pc_data][j_]&= -((depth_mask_0)+1);;
                                        data[i][offset_pc_data][j_] |= (LEVEL_SAME << depth_shift_0);
                                        
                                    } else {
                                        //std::cout << "BUG" << std::endl;
                                    }
                                }
                            }
                        }
                    }
                    
                    
                    
                } else {
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord) if(z_num_*x_num_ > 100)
                    for(z_ = z_start;z_ < (z_num_-z_stop);z_++){
                        
                        for(x_ = x_start;x_ < (x_num_-x_stop);x_++){
                            
                            const size_t z_parent = (z_+z_offset)/2;
                            const size_t x_parent = (x_+x_offset)/2;
                            
                            const size_t z_neigh = (z_+z_offset);
                            const size_t x_neigh = (x_+x_offset);
                            
                            const size_t offset_pc_data = x_num_*z_ + x_;
                            const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                            const size_t offset_pc_data_neigh = x_num_*z_neigh + x_neigh;
                            
                            
                            //initialization
                            y_coord = (data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT ;
                            y_neigh = (data[i][offset_pc_data_neigh][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_parent = (data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            
                            j_parent = 1;
                            j_neigh = 1;
                            
                            if (data[i-1][offset_pc_data_parent].size() == 1){
                                //set to max so its not checked
                                y_parent = 64000;
                            }
                            
                            if (data[i][offset_pc_data_neigh].size() == 1){
                                //set to max so its not checked
                                y_neigh = 64000;
                            }
                            
                            
                            y_coord--;
                            
                            const size_t j_num = data[i][offset_pc_data].size();
                            const size_t j_num_parent = data[i-1][offset_pc_data_parent].size();
                            const size_t j_num_neigh = data[i][offset_pc_data_neigh].size();
                            
                            for(j_ = 1;j_ < j_num;j_++){
                                
                                // Parent relation
                                
                                node_val = data[i][offset_pc_data][j_];
                                
                                if (node_val&1){
                                    //get the index gap node
                                    y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                    y_coord--;
                                    
                                } else {
                                    //normal node
                                    y_coord++;
                                    
                                    while ((y_neigh < y_coord) & (j_neigh < (j_num_neigh-1))){
                                        
                                        j_neigh++;
                                        node_val = data[i][offset_pc_data_neigh][j_neigh];
                                        
                                        if (node_val&1){
                                            //get the index gap node
                                            y_neigh = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                            j_neigh++;
                                            
                                        } else {
                                            //normal node
                                            y_neigh++;
                                        }
                                        
                                    }
                                    
                                    while ((y_parent < y_coord/2) & (j_parent < (j_num_parent-1))){
                                        
                                        j_parent++;
                                        node_val = data[i-1][offset_pc_data_parent][j_parent];
                                        
                                        if (node_val&1){
                                            //get the index gap node
                                            y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                            j_parent++;
                                            
                                        } else {
                                            //normal node
                                            y_parent++;
                                            
                                        }
                                    }
                                    
                                    
                                    if(y_coord == y_neigh){
                                        data[i][offset_pc_data][j_] |= (j_neigh << index_shift_0);
                                        data[i][offset_pc_data][j_] &= -((depth_mask_0)+1);
                                        data[i][offset_pc_data][j_] |= (LEVEL_SAME << depth_shift_0);
                                    } else if (y_coord/2 == y_parent){
                                        data[i][offset_pc_data][j_] |= (j_parent << index_shift_0);
                                        data[i][offset_pc_data][j_] &= -((depth_mask_0)+1);
                                        data[i][offset_pc_data][j_] |= (LEVEL_DOWN << depth_shift_0);
                                        //symmetric
                                        if((y_coord == y_parent*2) & (x_ == (x_parent*2 + (x_offset < 0))) & (z_ == (z_parent*2 + (z_offset < 0)) )){
                                            //only add parent once
                                            //need to choose the correct one... formulae
                                            
                                            data[i-1][offset_pc_data_parent][j_parent] |= (j_ << index_shift_1);
                                            data[i-1][offset_pc_data_parent][j_parent] &= -((depth_mask_1)+1);
                                            data[i-1][offset_pc_data_parent][j_parent] |= (LEVEL_UP << depth_shift_1);
                                        }
                                    } else {
                                        //std::cout << "BUG" << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            timer.stop_timer();
            
        } else {
            
            /////////////////////////////////////////////////////////////
            //
            //
            // Y direction (+-) neigh loops (In memory direction)
            //
            //
            /////////////////////////////////////////////////////////////
            
            
            timer.start_timer("Get neighbours dir: " + std::to_string(face));
            
            for(int i = (depth_min+1);i <= depth_max;i++){
                
                const unsigned int x_num_ = x_num[i];
                const unsigned int z_num_ = z_num[i];
                
                const unsigned int x_num_parent = x_num[i-1];
                
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,y_coord) if(z_num_*x_num_ > 100)
                for(z_ = 0;z_ < (z_num_);z_++){
                    
                    for(x_ = 0;x_ < (x_num_);x_++){
                        
                        const size_t z_parent = (z_)/2;
                        const size_t x_parent = (x_)/2;
                        
                        const size_t offset_pc_data = x_num_*z_ + x_;
                        const size_t offset_pc_data_parent = x_num_parent*z_parent + x_parent;
                        
                        //initialization
                        y_coord = (data[i][offset_pc_data][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_parent = (data[i-1][offset_pc_data_parent][0] & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        j_parent = 1;
                        
                        if (data[i-1][offset_pc_data_parent].size() == 1){
                            //set to max so its not checked
                            y_parent = 64000;
                        }
                        
                        y_coord--;
                        
                        const size_t j_num = data[i][offset_pc_data].size();
                        const size_t j_num_parent = data[i-1][offset_pc_data_parent].size();
                        
                        for(j_ = 1;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = data[i][offset_pc_data][j_];
                            
                            if (node_val&1){
                                //get the index gap node
                                
                                y_coord = (node_val & next_prev_mask) >> next_prev_shift;
                                
                                //iterate parent
                                while ((y_parent < (y_coord+y_offset)/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                }
                                
                                if((y_coord+y_offset)/2 == y_parent){
                                    data[i][offset_pc_data][j_] |= (j_parent << index_shift_0);
                                    data[i][offset_pc_data][j_] &= -((depth_mask_0)+1);
                                    data[i][offset_pc_data][j_] |= (  LEVEL_DOWN  << depth_shift_0);
                                    //symmetric (only add it once)
                                    if((y_coord == (y_parent*2 + (y_offset < 0))) & (x_ == x_parent*2) & (z_ == (z_parent*2) )){
                                        data[i-1][offset_pc_data_parent][j_parent-y_offset] |= ( (j_-y_offset) << index_shift_1);
                                    
                                        data[i-1][offset_pc_data_parent][j_parent-y_offset] &= -((depth_mask_1)+1);
                                        data[i-1][offset_pc_data_parent][j_parent-y_offset] |= ( LEVEL_UP  << depth_shift_1);
                                    }
                                } else {
                                    //end node
                                }
                                
                                
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                //iterate parent
                                while ((y_parent < (y_coord+y_offset)/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        j_parent++;
                                        
                                    } else {
                                        //normal node
                                        y_parent++;
                                        
                                    }
                                }
                                
                            }
                        }
                        
                    }
                }
            }
            
            timer.stop_timer();
            
        }
        
    }
    
    


private:
    //  Direction definitions for Particle Cell Neighbours
    //  [+y,-y,+x,-x,+z,-z]
    //  [0,1,2,3,4,5]

    const uint64_t depth_mask_dir[6] = {YP_DEPTH_MASK,YM_DEPTH_MASK,XP_DEPTH_MASK,XM_DEPTH_MASK,ZP_DEPTH_MASK,ZM_DEPTH_MASK};
    const uint64_t depth_shift_dir[6] =  {YP_DEPTH_SHIFT,YM_DEPTH_SHIFT,XP_DEPTH_SHIFT,XM_DEPTH_SHIFT,ZP_DEPTH_SHIFT,ZM_DEPTH_SHIFT};
    
    const uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
    const uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
    
    const int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
    const int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
    const int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};
    
    //the ordering of retrieval of four neighbour cells
    const uint8_t neigh_child_dir[6][3] = {{4,2,5},{4,2,5},{0,4,1},{0,4,1},{0,2,1},{0,2,1}};
    
    
    //variables for neighbour search loops
    const uint8_t x_start_vec[6] = {0,0,0,1,0,0};
    const uint8_t x_stop_vec[6] = {0,0,1,0,0,0};
    
    const uint8_t z_start_vec[6] = {0,0,0,0,0,1};
    const uint8_t z_stop_vec[6] = {0,0,0,0,1,0};
    
    const uint8_t y_start_vec[6] = {0,1,0,0,0,0};
    const uint8_t y_stop_vec[6] = {1,0,0,0,0,0};
    
    //replication of above
    const int8_t x_offset_vec[6] = {0,0,1,-1,0,0};
    const int8_t z_offset_vec[6] = {0,0,0,0,1,-1};
    const int8_t y_offset_vec[6] = {1,-1,0,0,0,0};
    
    const uint64_t index_shift_dir_sym[6] = {YM_INDEX_SHIFT,YP_INDEX_SHIFT,XM_INDEX_SHIFT,XP_INDEX_SHIFT,ZM_INDEX_SHIFT,ZP_INDEX_SHIFT};
    const uint64_t depth_shift_dir_sym[6] = {YM_DEPTH_SHIFT,YP_DEPTH_SHIFT,XM_DEPTH_SHIFT,XP_DEPTH_SHIFT,ZM_DEPTH_SHIFT,ZP_DEPTH_SHIFT};
    
    const uint64_t index_mask_dir_sym[6] = {YM_INDEX_MASK,YP_INDEX_MASK,XM_INDEX_MASK,XP_INDEX_MASK,ZM_INDEX_MASK,ZP_INDEX_MASK};
    const uint64_t depth_mask_dir_sym[6] = {YM_DEPTH_MASK,YP_DEPTH_MASK,XM_DEPTH_MASK,XP_DEPTH_MASK,ZM_DEPTH_MASK,ZP_DEPTH_MASK};
    
    const uint64_t next_prev_mask_vec[6] = {0,0,0,0,PREV_COORD_MASK,NEXT_COORD_MASK};
    const uint64_t next_prev_shift_vec[6] = {0,0,0,0,PREV_COORD_SHIFT,NEXT_COORD_SHIFT};
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP