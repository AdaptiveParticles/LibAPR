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

#define PC_KEY_DEPTH_MASK ((((uint64_t)1) << 4) - 1) << 0
#define PC_KEY_DEPTH_SHIFT 0
#define PC_KEY_X_MASK ((((uint64_t)1) << 13) - 1) << 4
#define PC_KEY_X_SHIFT 4

#define PC_KEY_Z_MASK ((((uint64_t)1) << 13) - 1) << 17
#define PC_KEY_Z_SHIFT 17
#define PC_KEY_J_MASK (((((uint64_t)1) << 13) - 1) << 30)
#define PC_KEY_J_SHIFT 30

#define PC_KEY_INDEX_MASK ((((uint64_t)1) << 15) - 1) << 43
#define PC_KEY_INDEX_SHIFT 43

#define PC_KEY_PARTNUM_MASK ((((uint64_t)1) << 3) - 1) << 58
#define PC_KEY_PARTNUM_SHIFT 58

#define PC_KEY_STATUS_MASK ((((uint64_t)1) << 2) - 1) << 61
#define PC_KEY_STATUS_SHIFT 61

#define PC_KEY_PARTICLE_MASK ((((uint64_t)1) << 20) - 1) << 43

#include "PartCellNeigh.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"


struct pc_key {
    
    static constexpr uint8_t seed_part_y[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    static constexpr uint8_t seed_part_x[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    static constexpr uint8_t seed_part_z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    
    int x,z,j,depth,p,status,y;
    int x_p,z_p,j_p,depth_p,y_p;
    
    void update_cell(uint64_t raw_key){
        depth = (raw_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        x = (raw_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        z = (raw_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        j = (raw_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
        p = (raw_key & PC_KEY_PARTNUM_MASK) >> PC_KEY_PARTNUM_SHIFT;
        status = (raw_key & PC_KEY_STATUS_MASK) >> PC_KEY_STATUS_SHIFT;
        
    }
    
    void update_part(uint64_t raw_key){
        depth = (raw_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        x = (raw_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        z = (raw_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        j = (raw_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
        p = (raw_key & PC_KEY_PARTNUM_MASK) >> PC_KEY_PARTNUM_SHIFT;
        status = (raw_key & PC_KEY_STATUS_MASK) >> PC_KEY_STATUS_SHIFT;

        if(status == SEED){
            int part_num = p;
            
            depth_p = depth + 1;
            x_p = x*2 + seed_part_x[part_num];
            z_p = z*2 + seed_part_z[part_num];
            y_p = 2*y + seed_part_y[part_num];
            
        }
        else {
            x_p = x;
            y_p = y;
            z_p = z;
            depth_p = depth;
        }

        
    }

    bool compare_cell(pc_key comp){
        int same = (comp.x == x) + (comp.z == z) + (comp.j == j) + (comp.depth == depth);

        if(same == 4){
            return true;
        } else {
            return false;
        }


    }


    
    
};




struct node_key {
    
    
    int xp_j,zp_j;
    int xp_dep,zp_dep;
    
    int xm_j,zm_j;
    int xm_dep,zm_dep;
    
    int status;
    int type;
    
    int yp_j,ym_j;
    int yp_dep,ym_dep;
    
    int next_y;
    int prev_y;
    
    void update_node(uint64_t node_val){
        if (node_val&1){
            
            type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
            
            yp_j = (node_val & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
            yp_dep = (node_val & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;
            
            ym_j = (node_val & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
            ym_dep = (node_val & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;
            
            next_y = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
            
            prev_y = (node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT;
            
            
        } else {
            
            status = (node_val & STATUS_MASK) >> STATUS_SHIFT;
            type = (node_val & TYPE_MASK) >> TYPE_SHIFT;
            
            xp_j= (node_val & XP_INDEX_MASK) >> XP_INDEX_SHIFT;
            xp_dep = (node_val & XP_DEPTH_MASK) >> XP_DEPTH_SHIFT;
            
            zp_j = (node_val & ZP_INDEX_MASK) >> ZP_INDEX_SHIFT;
            zp_dep = (node_val & ZP_DEPTH_MASK) >> ZP_DEPTH_SHIFT;
            
            xm_j = (node_val & XM_INDEX_MASK) >> XM_INDEX_SHIFT;
            xm_dep = (node_val & XM_DEPTH_MASK) >> XM_DEPTH_SHIFT;
            
            zm_j = (node_val & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
            zm_dep = (node_val & ZM_DEPTH_MASK) >> ZM_DEPTH_SHIFT;
            
            
        }
    }
    
        
};


template <typename T> // type T data structure base type
class PartCellData {
    
public:
    
    /*
     * Number of layers without the root and the contents.
     */
    

    static constexpr uint64_t depth_mask_dir[6] = {YP_DEPTH_MASK,YM_DEPTH_MASK,XP_DEPTH_MASK,XM_DEPTH_MASK,ZP_DEPTH_MASK,ZM_DEPTH_MASK};
    static constexpr uint64_t depth_shift_dir[6] =  {YP_DEPTH_SHIFT,YM_DEPTH_SHIFT,XP_DEPTH_SHIFT,XM_DEPTH_SHIFT,ZP_DEPTH_SHIFT,ZM_DEPTH_SHIFT};

    static constexpr uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
    static constexpr uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};

    static constexpr int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
    static constexpr int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
    static constexpr int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};

    //the ordering of retrieval of four neighbour cells

    static constexpr uint8_t neigh_child_dir[6][3] = {{4,2,2},{4,2,2},{0,4,4},{0,4,4},{0,2,2},{0,2,2}};

//    constexpr uint8_t whatever::neigh_child_dir[6][3] = {{4,2,2},{4,2,2},{0,4,4},{0,4,4},{0,2,2},{0,2,2}};
//


    static constexpr uint8_t neigh_child_y_offsets[6][4] = {{0,0,0,0},{0,0,0,0},{0,1,0,1},{0,1,0,1},{0,1,0,1},{0,1,0,1}};
//
    //variables for neighbour search loops
    static constexpr uint8_t x_start_vec[6] = {0,0,0,1,0,0};
    static constexpr uint8_t x_stop_vec[6] = {0,0,1,0,0,0};

    static constexpr uint8_t z_start_vec[6] = {0,0,0,0,0,1};
    static constexpr uint8_t z_stop_vec[6] = {0,0,0,0,1,0};

    static constexpr uint8_t y_start_vec[6] = {0,1,0,0,0,0};
    static constexpr uint8_t y_stop_vec[6] = {1,0,0,0,0,0};

    //replication of above
    static constexpr int8_t x_offset_vec[6] = {0,0,1,-1,0,0};
    static constexpr int8_t z_offset_vec[6] = {0,0,0,0,1,-1};
    static constexpr int8_t y_offset_vec[6] = {1,-1,0,0,0,0};

    static constexpr uint64_t index_shift_dir_sym[6] = {YM_INDEX_SHIFT,YP_INDEX_SHIFT,XM_INDEX_SHIFT,XP_INDEX_SHIFT,ZM_INDEX_SHIFT,ZP_INDEX_SHIFT};
    static constexpr uint64_t depth_shift_dir_sym[6] = {YM_DEPTH_SHIFT,YP_DEPTH_SHIFT,XM_DEPTH_SHIFT,XP_DEPTH_SHIFT,ZM_DEPTH_SHIFT,ZP_DEPTH_SHIFT};

    static constexpr uint64_t index_mask_dir_sym[6] = {YM_INDEX_MASK,YP_INDEX_MASK,XM_INDEX_MASK,XP_INDEX_MASK,ZM_INDEX_MASK,ZP_INDEX_MASK};
    static constexpr uint64_t depth_mask_dir_sym[6] = {YM_DEPTH_MASK,YP_DEPTH_MASK,XM_DEPTH_MASK,XP_DEPTH_MASK,ZM_DEPTH_MASK,ZP_DEPTH_MASK};

    static constexpr uint64_t next_prev_mask_vec[6] = {PREV_COORD_MASK,NEXT_COORD_MASK,0,0,0,0};
    static constexpr uint64_t next_prev_shift_vec[6] = {PREV_COORD_SHIFT,NEXT_COORD_SHIFT,0,0,0,0};

    static constexpr uint8_t seed_part_y[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    static constexpr uint8_t seed_part_x[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    static constexpr uint8_t seed_part_z[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    //Edge and corner neighbour look ups

    static constexpr uint8_t edges_face[12] = {0,1,0,1,4,4,4,4,5,5,5,5};
    static constexpr uint8_t edges_face_index[12][2] = {{2,3},{2,3},{0,1},{0,1},{1,3},{0,2},{2,3},{0,1},{1,3},{0,2},{2,3},{0,1}};
    static constexpr uint8_t edges_face_dir[12] = {2,2,3,3,0,1,2,3,0,1,2,3};
    static constexpr uint8_t edges_child_index[12][2] = {{0,2},{1,3},{0,2},{1,3},{0,2},{0,2},{0,1},{0,1},{1,3},{1,3},{2,3},{2,3}};

    static constexpr uint8_t edges_parent_ind[12][2] = {{1,1},{0,1},{1,0},{0,0},{1,1},{0,1},{1,1},{0,1},{1,0},{0,0},{1,0},{0,0}};
    static constexpr uint8_t edges_parent_type[12] = {2,2,2,2,1,1,0,0,1,1,0,0};

    static constexpr uint8_t corner_edge[8] = {6,6,7,7,10,10,11,11};
    static constexpr uint8_t corner_edge_index[8] = {1,0,1,0,1,0,1,0};
    static constexpr uint8_t corner_edge_dir[8] = {0,1,0,1,0,1,0,1};
    static constexpr uint8_t corner_edge_move_index[8] = {0,0,2,2,2,2,3,3};

    static constexpr uint8_t corner_parent_ind[8][3] = {{1,1,1},{0,1,1},{1,0,1},{0,0,1},{1,1,0},{0,1,0},{1,0,0},{0,0,0}};

    uint64_t depth_max;
    uint64_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    std::vector<unsigned int> y_num;
    
    std::vector<unsigned int> org_dims;
    
    std::vector<std::vector<std::vector<T>>> data;
    
    PartCellData(){};
    
    /////////////////////////////////////////
    //
    //  Get and Set methods for PC_KEYS
    //
    //////////////////////////////////////////
    
    T get_j_from_y(const T& x_,const T& z_,const T& depth,const T& y_val){
        
        
        const size_t offset_pc_data = x_num[depth]*z_ + x_;
        
        const size_t j_num =  data[depth][offset_pc_data].size();
        
        T y_coord = 0;
        T node_val = 0;
        T j_;
        
        for(j_ = 0;j_ < j_num;j_++){
            
            //this value encodes the state and neighbour locations of the particle cell
            node_val = data[depth][offset_pc_data][j_];
            
            if (!(node_val&1)){
                y_coord++; //iterate y
                
                if(y_coord == y_val){
                    return j_;
                } else if (y_coord > y_val){
                    //doesn't exist
                    return 0;
                }
                
            } else{
                y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                y_coord--;
            }
            
        }
        
        return 0;
        
    }
    
    
    
    
    inline bool pc_key_cell_isequal(const T& pc_key0,const T& pc_key1){
        //
        //  Checks if the partcell keys address the same cell
        //
        //  Could be different particles! (Ignores the index value)
        //
        
        return (pc_key0 & (-((PC_KEY_PARTICLE_MASK)+1))) == (pc_key1 & (-((PC_KEY_PARTICLE_MASK)+1)));
    
    };
    
    inline bool pc_key_part_isequal(const T& pc_key0,const T& pc_key1){
        //
        // Compares if two particles are the same
        //
        
        return pc_key0 == pc_key1;
    
    };
    
    T& operator ()(int depth, int x_,int z_,int j_){
        // data access
        return data[depth][x_num[depth]*z_ + x_][j_];
    }
    
    inline T pc_key_get_x(const T& pc_key){
        return (pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
    }
    
    inline T pc_key_get_j(const T& pc_key){
        return (pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
    }
    
    inline T pc_key_get_z(const T& pc_key){
        return (pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
    }
    
    inline T pc_key_get_depth(const T& pc_key){
        return (pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
    }
    
    inline T pc_key_get_index(const T& pc_key){
        return (pc_key & PC_KEY_INDEX_MASK) >> PC_KEY_INDEX_SHIFT;
    }
    
    inline T pc_key_get_partnum(const T& pc_key){
        return (pc_key & PC_KEY_PARTNUM_MASK) >> PC_KEY_PARTNUM_SHIFT;
    }
    
    inline T pc_key_get_status(const T& pc_key){
        return (pc_key & PC_KEY_STATUS_MASK) >> PC_KEY_STATUS_SHIFT;
    }
    
    inline void pc_key_set_x(T& pc_key,const T& x_){
        pc_key &= -((PC_KEY_X_MASK) + 1); // clear the current value
        pc_key|= x_  << PC_KEY_X_SHIFT; //set value
    }
    
    inline void pc_key_set_z(T& pc_key,const T& z_){
        pc_key &= -((PC_KEY_Z_MASK) + 1); // clear the current value
        pc_key|= z_  << PC_KEY_Z_SHIFT; //set  value
    }
    
    inline void pc_key_set_j(T& pc_key,const T& j_){
        pc_key &= -((PC_KEY_J_MASK) + 1); // clear the current value
        pc_key|= j_  << PC_KEY_J_SHIFT; //set  value
    }
    
    inline void pc_key_set_depth(T& pc_key,const T& depth_){
        pc_key &= -((PC_KEY_DEPTH_MASK) + 1); // clear the current value
        pc_key|= depth_  << PC_KEY_DEPTH_SHIFT; //set  value
    }
    
    inline void pc_key_set_index(T& pc_key,const T& index_){
        pc_key &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
        pc_key|= index_  << PC_KEY_INDEX_SHIFT; //set  value
    }
    
    inline void pc_key_set_partnum(T& pc_key,const T& index_){
        pc_key &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
        pc_key|= index_  << PC_KEY_PARTNUM_SHIFT; //set  value
    }
    
    inline void pc_key_set_status(T& pc_key,const T& index_){
        pc_key &= -((PC_KEY_STATUS_MASK) + 1); // clear the current value
        pc_key|= index_  << PC_KEY_STATUS_SHIFT; //set  value
    }
    
    
    inline void pc_key_offset_x(T& pc_key,const int& offset){
        T temp = pc_key_get_x(pc_key);
        pc_key &= -((PC_KEY_X_MASK) + 1); // clear the current value
        pc_key|= (temp+offset)  << PC_KEY_X_SHIFT; //set  value
    }
    
    inline void pc_key_offset_z(T& pc_key,const int& offset){
        T temp = pc_key_get_z(pc_key);
        pc_key &= -((PC_KEY_Z_MASK) + 1); // clear the current value
        pc_key|= (temp+offset)  << PC_KEY_Z_SHIFT; //set  value
    }
    
    inline void pc_key_offset_j(T& pc_key,const int& offset){
        T temp = pc_key_get_j(pc_key);
        pc_key &= -((PC_KEY_J_MASK) + 1); // clear the current value
        pc_key|= (temp+offset)  << PC_KEY_J_SHIFT; //set  value
    }
    
    inline void pc_key_offset_depth(T& pc_key,const int& offset){
        T temp = pc_key_get_depth(pc_key);
        pc_key &= -((PC_KEY_DEPTH_MASK) + 1); // clear the current value
        pc_key|= (temp+offset)  << PC_KEY_DEPTH_SHIFT; //set  value
    }
    
    inline void pc_key_offset_index(T& pc_key,const int& offset){
        T temp = pc_key_get_index(pc_key);
        pc_key &= -((PC_KEY_INDEX_MASK) + 1); // clear the current value
        pc_key|= (temp+offset)  << PC_KEY_INDEX_SHIFT; //set  value
    }
    
    inline void pc_key_offset_partnum(T& pc_key,const int& offset){
        T temp = pc_key_get_partnum(pc_key);
        pc_key &= -((PC_KEY_PARTNUM_MASK) + 1); // clear the current value
        pc_key|= (temp+offset)  << PC_KEY_PARTNUM_SHIFT; //set  value
    }
    
    inline T node_get_val(const T& node_val,const T& mask,const T& shift){
        return (node_val & mask) >> shift;
    }
    
    inline void node_set_val(T& node_val,const T& val_,const T& mask,const T& shift){
        node_val &= -((mask) + 1); // clear the current value
        node_val|= val_ << shift; //set  value
    }
    
    inline void node_offset_val(T& node_val,const int& offset,const T& mask,const T& shift){
        T temp = node_get_val(node_val,mask,shift);
        node_val &= -((mask) + 1); // clear the current value
        node_val|= (temp+offset)  << shift; //set  value
    }
    
    uint8_t get_status(T& node_val){
        //
        //  Extracts the status
        //
        
        return (node_val & STATUS_MASK) >> STATUS_SHIFT;
    }

    
    
    T get_val_guarded(const uint64_t& pc_key) {
        // data access

        const uint64_t depth = (pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t j_ = (pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;

        if (x_num[depth] * z_ + x_ >= data[depth].size()) {
            return 0;
        } else if(j_ >= data[depth][x_num[depth]*z_ + x_].size()) {
            return 0;
        } else {
            return data[depth][x_num[depth] * z_ + x_][j_];
        }
        //return data[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT][x_num[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT]*((pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + ((pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT)][(pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT];
    }

    T& get_val(const uint64_t& pc_key) {
        // data access

        const uint64_t depth = (pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t j_ = (pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;

        return data[depth][x_num[depth] * z_ + x_][j_];
        //return data[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT][x_num[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT]*((pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + ((pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT)][(pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT];
    }
    
    
    void get_details_cell(const T& curr_key,T& x_,T& z_,T& j_,T& depth_){
        //
        //  Calculates coordinates and details for a particle, requires pc_key and current y
        //
                
        x_ = pc_key_get_x(curr_key);
        j_ = pc_key_get_j(curr_key);
        z_ = pc_key_get_z(curr_key);
        depth_ = pc_key_get_depth(curr_key);
    }
    
    void set_details_cell(T& curr_key,const T& x_,const T& z_,const T& j_,T& depth_){
        //
        //  Calculates coordinates and details for a particle, requires pc_key and current y
        //
        
        pc_key_set_x(curr_key,x_);
        pc_key_set_j(curr_key,j_);
        pc_key_set_z(curr_key,z_);
        pc_key_set_depth(curr_key,depth_);
    }
    
    void get_coordinates_cell(T current_y,const T& curr_key,T& x_,T& z_,T& y_,T& depth_,T& status_){
        //
        //  Calculates coordinates and details for a particle, requires pc_key and current y
        //
        
        status_ = pc_key_get_status(curr_key);
        
        x_ = pc_key_get_x(curr_key);
        y_ = current_y;
        z_ = pc_key_get_z(curr_key);
        depth_ = pc_key_get_depth(curr_key);
    }
    
    void get_coordinates_part(T current_y,const T& curr_key,T& x_,T& z_,T& y_,T& depth_,T& status_){
        //
        //  Calculates coordinates and details for a particle, requires pc_key and current y
        //
        
        status_ = pc_key_get_status(curr_key);
        
        if(status_ == SEED){
            T part_num = pc_key_get_partnum(curr_key);
            
            depth_ = pc_key_get_depth(curr_key) + 1;
            
            x_ = pc_key_get_x(curr_key)*2 + seed_part_x[part_num];
            z_ = pc_key_get_z(curr_key)*2 + seed_part_z[part_num];
            y_ = 2*current_y + seed_part_y[part_num];
            
        }
        else {
            x_ = pc_key_get_x(curr_key);
            y_ = current_y;
            z_ = pc_key_get_z(curr_key);
            depth_ = pc_key_get_depth(curr_key);
        }
        
    }
    
    void get_coordinates_part_full(T current_y,const T& curr_key,uint16_t& x_,uint16_t& z_,uint16_t& y_,uint8_t& depth_,uint8_t& status_){
        //
        //  Calculates coordinates and details for a particle, requires pc_key and current y
        //
        //  Calculated in a common reference frame
        //
        
        
        status_ = pc_key_get_status(curr_key);
        depth_ = pc_key_get_depth(curr_key);
        T depth_factor = pow(2,depth_max - depth_);
        
        if(status_ == SEED){
            T part_num = pc_key_get_partnum(curr_key);
            
            depth_ = depth_ + 1;
            
            x_ = (4*pc_key_get_x(curr_key) + 1 + 2*seed_part_x[part_num])*depth_factor;
            z_ = (4*pc_key_get_z(curr_key) + 1 + 2*seed_part_z[part_num])*depth_factor;
            y_ = (4*current_y + 1 + 2*seed_part_y[part_num])*depth_factor;
        }
        else {
            x_ = (4*pc_key_get_x(curr_key) + 2)*depth_factor;
            z_ = (4*pc_key_get_z(curr_key) + 2)*depth_factor;
            y_ = (4*current_y + 2)*depth_factor;
        }
        
        
        
        
    }
    
    void get_coordinates_cell_full(T current_y,const T& curr_key,T& x_,T& z_,T& y_,T& depth_,T& status_){
        //
        //  Calculates coordinates and details for a particle, requires pc_key and current y
        //
        //  Calculated in a common reference frame
        //
        
        status_ = pc_key_get_status(curr_key);
        depth_ = pc_key_get_depth(curr_key);
        T depth_factor = pow(2,depth_max - depth_);
        
        x_ = (4*pc_key_get_x(curr_key) + 2)*depth_factor;
        z_ = (4*pc_key_get_z(curr_key) + 2)*depth_factor;
        y_ = (4*current_y + 2)*depth_factor;
        
    }
    
    pc_key get_neigh_coordinates_cell(PartCellNeigh<T>& cell_neigh,T face,T index,T current_y){
        //
        //  Get the coordinates for a cell
        //
        
        T neigh = cell_neigh.neigh_face[face][index];
        
        T neigh_x,neigh_y,neigh_z,neigh_depth;
        
        
        if(neigh > 0){
            neigh_x = pc_key_get_x(neigh);
            neigh_z = pc_key_get_z(neigh);
            neigh_depth = pc_key_get_depth(neigh);
            
            T curr_depth = pc_key_get_depth(cell_neigh.curr);
            
            if(neigh_depth == curr_depth){
                //neigh is on same layer
                neigh_y = current_y + von_neumann_y_cells[face];
            }
            else if (neigh_depth < curr_depth){
                //neigh is on parent layer
                neigh_y = (current_y + von_neumann_y_cells[face])/2;
            }
            else{
                //neigh is on child layer
                neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
            }
            
            
        } else {
            neigh_y = 0;
            neigh_x = 0;
            neigh_z = 0;
            neigh_depth = 0;
        }
        
        pc_key curr_cell;
        
        curr_cell.y = neigh_y;
        curr_cell.x = neigh_x;
        curr_cell.z = neigh_z;
        curr_cell.depth = neigh_depth;
        
        return curr_cell;
        
    };

    
    
    void get_neigh_coordinates_cell(PartCellNeigh<T>& cell_neigh,T face,T index,T current_y,T& neigh_y,T& neigh_x,T& neigh_z,T& neigh_depth){
        //
        //  Get the coordinates for a cell
        //
        
        T neigh = cell_neigh.neigh_face[face][index];
      
        
        if(neigh > 0){
            neigh_x = pc_key_get_x(neigh);
            neigh_z = pc_key_get_z(neigh);
            neigh_depth = pc_key_get_depth(neigh);
            
            T curr_depth = pc_key_get_depth(cell_neigh.curr);
            
            if(neigh_depth == curr_depth){
                //neigh is on same layer
                neigh_y = current_y + von_neumann_y_cells[face];
            }
            else if (neigh_depth < curr_depth){
                //neigh is on parent layer
                neigh_y = (current_y + von_neumann_y_cells[face])/2;
            }
            else{
                //neigh is on child layer
                neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
            }
            
            
        } else {
            neigh_y = 0;
            neigh_x = 0;
            neigh_z = 0;
            neigh_depth = 0;
        }
        
        
        
    };
    
    pc_key get_neigh_coordinates_part(PartCellNeigh<T>& part_cell_neigh,T face,T index,T current_y){
        //
        //  Get the coordinates for a particle
        //
        //
        
        pc_key neigh_key;
        
        T neigh_y,neigh_x,neigh_z,neigh_depth;
        
        T neigh = part_cell_neigh.neigh_face[face][index];
        
        T curr = part_cell_neigh.curr;
        
        T status = pc_key_get_status(curr);
        
        if(neigh > 0){
            
            if(status > SEED){
                //current cell is not seed
                T neigh_status = pc_key_get_status(neigh);
                
                if(neigh_status > SEED){
                    // neighbour cell is not seed, simply use cell coordinates
                    neigh_x = pc_key_get_x(neigh);
                    neigh_z = pc_key_get_z(neigh);
                    neigh_depth = pc_key_get_depth(neigh);
                    
                    
                    
                    T curr_depth = pc_key_get_depth(curr);
                    
                    if(neigh_depth == curr_depth){
                        //neigh is on same layer
                        neigh_y = current_y + von_neumann_y_cells[face];
                    }
                    else if (neigh_depth < curr_depth){
                        //neigh is on parent layer
                        neigh_y = (current_y + von_neumann_y_cells[face])/2;
                    }
                    else{
                        //neigh is on child layer
                        neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
                    }
                    
                } else {
                    // neighbour cell is a seed cell, get cell coords then offset for part
                    
                    T n_depth = pc_key_get_depth(neigh);
                    neigh_depth = n_depth + 1;
                    
                    
                    T part_num = pc_key_get_partnum(neigh);
                    
                    neigh_x = pc_key_get_x(neigh)*2 + seed_part_x[part_num];
                    neigh_z = pc_key_get_z(neigh)*2 + seed_part_z[part_num];
                    
                    T curr_depth = pc_key_get_depth(curr);
                    
                    if(n_depth == curr_depth){
                        //neigh is on same layer
                        neigh_y = current_y + von_neumann_y_cells[face];
                    }
                    else if (n_depth < curr_depth){
                        //neigh is on parent layer
                        neigh_y = (current_y + von_neumann_y_cells[face])/2;
                    }
                    else{
                        //neigh is on child layer
                        neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
                    }
                    
                    neigh_y = neigh_y*2 + seed_part_y[part_num];
                    
                }
                
            } else {
                
                
                //current cell is seed
                
                
                T part_num = pc_key_get_partnum(neigh);
                
                
                
                //check if still in the same cell or not
                if(pc_key_cell_isequal(curr,neigh)){
                    //
                    // The same cell
                    //
                    
                    T n_depth = pc_key_get_depth(neigh);
                    neigh_depth = n_depth + 1;
                    
                    neigh_x = pc_key_get_x(neigh)*2 + seed_part_x[part_num];
                    neigh_z = pc_key_get_z(neigh)*2 + seed_part_z[part_num];
                    
                    neigh_y = 2*current_y + seed_part_y[part_num];
                    
                    
                } else {
                    
                    T neigh_status = pc_key_get_status(neigh);
                    
                    if(neigh_status > SEED){
                        // neighbour cell is not seed, simply use cell coordinates
                        neigh_x = pc_key_get_x(neigh);
                        neigh_z = pc_key_get_z(neigh);
                        neigh_depth = pc_key_get_depth(neigh);
                        
                        
                        
                        T curr_depth = pc_key_get_depth(curr);
                        
                        if(neigh_depth == curr_depth){
                            //neigh is on same layer
                            neigh_y = current_y + von_neumann_y_cells[face];
                        }
                        else if (neigh_depth < curr_depth){
                            //neigh is on parent layer
                            neigh_y = (current_y + von_neumann_y_cells[face])/2;
                        }
                        else{
                            //This case is the 1 -3 match up where this does not work, as only one cell is output and therefore the index needs to be corrected
                            constexpr uint64_t index_offset[6][8] = {{1,0,3,1,5,2,7,3},{0,0,1,2,2,4,3,6},{2,3,0,1,6,7,2,3},{0,1,0,1,2,3,4,5},{4,5,6,7,0,1,2,3},{0,1,2,3,0,1,2,3}};
                            T curr_part_num = pc_key_get_partnum(curr);
                            T adj_index = index_offset[face][curr_part_num];
                            //neigh is on child layer
                            neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][adj_index];
                            
                            
                            
                        }
                        
                    } else {
                        // neighbour cell is a seed cell, get cell coords then offset for part
                        
                        T n_depth = pc_key_get_depth(neigh);
                        neigh_depth = n_depth + 1;
                        
                        
                        T part_num = pc_key_get_partnum(neigh);
                        
                        neigh_x = pc_key_get_x(neigh)*2 + seed_part_x[part_num];
                        neigh_z = pc_key_get_z(neigh)*2 + seed_part_z[part_num];
                        
                        T curr_depth = pc_key_get_depth(curr);
                        
                        if(n_depth == curr_depth){
                            //neigh is on same layer
                            neigh_y = current_y + von_neumann_y_cells[face];
                        }
                        else if (n_depth < curr_depth){
                            //neigh is on parent layer
                            neigh_y = (current_y + von_neumann_y_cells[face])/2;
                        }
                        else{
                            //neigh is on child layer
                            neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
                        }
                        
                        neigh_y = neigh_y*2 + seed_part_y[part_num];
                        
                    }
                    
                }
                
                
            }
            
        }  else {
            neigh_y = 0;
            neigh_x = 0;
            neigh_z = 0;
            neigh_depth = 0;
        }
        
        neigh_key.y_p = neigh_y;
        neigh_key.x_p = neigh_x;
        neigh_key.z_p = neigh_z;
        neigh_key.depth_p = neigh_depth;
        
        return neigh_key;
        
    }
    
    void get_neigh_coordinates_part(PartCellNeigh<T>& part_cell_neigh,T face,T index,T current_y,T& neigh_y,T& neigh_x,T& neigh_z,T& neigh_depth){
        //
        //  Get the coordinates for a particle
        //
        //
        
        T neigh = part_cell_neigh.neigh_face[face][index];
        
        T curr = part_cell_neigh.curr;
        
        T status = pc_key_get_status(curr);
        
        if(neigh > 0){
            
            if(status > SEED){
                //current cell is not seed
                T neigh_status = pc_key_get_status(neigh);
                
                if(neigh_status > SEED){
                    // neighbour cell is not seed, simply use cell coordinates
                    neigh_x = pc_key_get_x(neigh);
                    neigh_z = pc_key_get_z(neigh);
                    neigh_depth = pc_key_get_depth(neigh);
                    
                    
                    
                    T curr_depth = pc_key_get_depth(curr);
                    
                    if(neigh_depth == curr_depth){
                        //neigh is on same layer
                        neigh_y = current_y + von_neumann_y_cells[face];
                    }
                    else if (neigh_depth < curr_depth){
                        //neigh is on parent layer
                        neigh_y = (current_y + von_neumann_y_cells[face])/2;
                    }
                    else{
                        //neigh is on child layer
                        neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
                    }
                    
                } else {
                    // neighbour cell is a seed cell, get cell coords then offset for part
                    
                    T n_depth = pc_key_get_depth(neigh);
                    neigh_depth = n_depth + 1;
                    
                    
                    T part_num = pc_key_get_partnum(neigh);
                    
                    neigh_x = pc_key_get_x(neigh)*2 + seed_part_x[part_num];
                    neigh_z = pc_key_get_z(neigh)*2 + seed_part_z[part_num];
                    
                    T curr_depth = pc_key_get_depth(curr);
                    
                    if(n_depth == curr_depth){
                        //neigh is on same layer
                        neigh_y = current_y + von_neumann_y_cells[face];
                    }
                    else if (n_depth < curr_depth){
                        //neigh is on parent layer
                        neigh_y = (current_y + von_neumann_y_cells[face])/2;
                    }
                    else{
                        //neigh is on child layer
                        neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
                    }
                    
                    neigh_y = neigh_y*2 + seed_part_y[part_num];
                    
                }
                
            } else {
                
                
                //current cell is seed

                T part_num = pc_key_get_partnum(neigh);

                
                //check if still in the same cell or not
                if(pc_key_cell_isequal(curr,neigh)){
                    //
                    // The same cell
                    //
                    
                    T n_depth = pc_key_get_depth(neigh);
                    neigh_depth = n_depth + 1;
                    
                    neigh_x = pc_key_get_x(neigh)*2 + seed_part_x[part_num];
                    neigh_z = pc_key_get_z(neigh)*2 + seed_part_z[part_num];
                    
                    neigh_y = 2*current_y + seed_part_y[part_num];
                    
                    
                } else {
                    
                    T neigh_status = pc_key_get_status(neigh);
                    
                    if(neigh_status > SEED){
                        // neighbour cell is not seed, simply use cell coordinates
                        neigh_x = pc_key_get_x(neigh);
                        neigh_z = pc_key_get_z(neigh);
                        neigh_depth = pc_key_get_depth(neigh);
                        
                        
                        
                        T curr_depth = pc_key_get_depth(curr);
                        
                        if(neigh_depth == curr_depth){
                            //neigh is on same layer
                            neigh_y = current_y + von_neumann_y_cells[face];
                        }
                        else if (neigh_depth < curr_depth){
                            //neigh is on parent layer
                            neigh_y = (current_y + von_neumann_y_cells[face])/2;
                        }
                        else{
                            //This case is the 1 -3 match up where this does not work, as only one cell is output and therefore the index needs to be corrected
                            constexpr uint64_t index_offset[6][8] = {{1,0,3,1,5,2,7,3},{0,0,1,2,2,4,3,6},{2,3,0,1,6,7,2,3},{0,1,0,1,2,3,4,5},{4,5,6,7,0,1,2,3},{0,1,2,3,0,1,2,3}};
                            T curr_part_num = pc_key_get_partnum(curr);
                            T adj_index = index_offset[face][curr_part_num];
                            //neigh is on child layer
                            neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][adj_index];
                            
                            
                          
                        }
                        
                    } else {
                        // neighbour cell is a seed cell, get cell coords then offset for part
                        
                        T n_depth = pc_key_get_depth(neigh);
                        neigh_depth = n_depth + 1;
                        
                        
                        T part_num = pc_key_get_partnum(neigh);
                        
                        neigh_x = pc_key_get_x(neigh)*2 + seed_part_x[part_num];
                        neigh_z = pc_key_get_z(neigh)*2 + seed_part_z[part_num];
                        
                        T curr_depth = pc_key_get_depth(curr);
                        
                        if(n_depth == curr_depth){
                            //neigh is on same layer
                            neigh_y = current_y + von_neumann_y_cells[face];
                        }
                        else if (n_depth < curr_depth){
                            //neigh is on parent layer
                            neigh_y = (current_y + von_neumann_y_cells[face])/2;
                        }
                        else{
                            //neigh is on child layer
                            neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
                        }
                        
                        neigh_y = neigh_y*2 + seed_part_y[part_num];
                        
                    }
                    
                }
                
                
            }
            
        }  else {
            neigh_y = 0;
            neigh_x = 0;
            neigh_z = 0;
            neigh_depth = 0;
        }
    
    }
    
    
    template<typename S>
    void initialize_base_structure(Particle_map<S>& part_map){
        //initializes the partcell data structure based on part_map size
        
        //first add the layers
        depth_max = part_map.k_max;
        depth_min = part_map.k_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        y_num.resize(depth_max+1);
        
        data.resize(depth_max+1);
        
        for(int i = depth_min;i <= depth_max;i++){
            z_num[i] = part_map.layers[i].z_num;
            x_num[i] = part_map.layers[i].x_num;
            y_num[i] = part_map.layers[i].y_num;
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
        
        for(uint64_t i = depth_min;i <= depth_max;i++){
            z_num[i] = part_cell_data.z_num[i];
            x_num[i] = part_cell_data.x_num[i];
            data[i].resize(z_num[i]*x_num[i]);
        }

        y_num = part_cell_data.y_num;

    }


    void create_partcell_structure(std::vector<std::vector<uint8_t>>& p_map){
        //
        //  Bevan Cheeseman 2017
        //
        //  Takes an optimal part_map configuration from the pushing scheme and creates an efficient data structure for procesing using V, instead of V_n as in original (needs to be optimized)
        //

        //initialize the structure
        (*this).data.resize((*this).depth_max + 1);

        for(uint64_t i = (*this).depth_min;i <= (*this).depth_max;i++){

            (*this).data[i].resize((*this).z_num[i]*(*this).x_num[i]);
        }

        Part_timer timer;
        timer.verbose_flag = false;

        //initialize loop variables
        uint64_t x_;
        uint64_t z_;
        uint64_t y_;

        //next initialize the entries;

        uint64_t curr_index;
        uint64_t status;
        uint64_t prev_ind = 0;

        std::vector<unsigned int> x_num = (*this).x_num;
        std::vector<unsigned int> y_num = (*this).y_num;
        std::vector<unsigned int> z_num = (*this).z_num;

        std::vector<uint64_t> status_temp;


        uint64_t prev_coord = 0;

        timer.start_timer("intiialize part_cells");

        const uint8_t seed_us = 4; //deal with the equivalence optimization

        for(uint64_t i = ((*this).depth_min+1);i < (*this).depth_max;i++) {

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            const unsigned int x_num_ds = x_num[i - 1];
            const unsigned int z_num_ds = z_num[i - 1];
            const unsigned int y_num_ds = y_num[i - 1];

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_, x_, y_, curr_index, status, prev_ind) if(z_num_*x_num_ > 100)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {
                    const size_t offset_part_map_ds = (x_ / 2) * y_num_ds + (z_ / 2) * y_num_ds * x_num_ds;
                    const size_t offset_part_map = x_ * y_num_ + z_ * y_num_ * x_num_;

                    for (y_ = 0; y_ < y_num_ds; y_++) {

                        status = p_map[i - 1][offset_part_map_ds + y_];

                        if (status == SEED) {
                            p_map[i][offset_part_map + 2 * y_] = seed_us;
                            p_map[i][offset_part_map + 2 * y_ + 1] = seed_us;
                        }
                    }
                }

            }
        }



        for(uint64_t i = (*this).depth_min;i <= (*this).depth_max;i++){

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            const unsigned int x_num_ds = x_num[i-1];
            const unsigned int z_num_ds = z_num[i-1];
            const unsigned int y_num_ds = y_num[i-1];

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    size_t first_empty = 0;
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_part_map_ds = (x_/2)*y_num_ds + (z_/2)*y_num_ds*x_num_ds;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 0;

                    //first value handle the duplication of the gap node

                    if(i == ((*this).depth_max)) {

                        status = p_map[i-1][ offset_part_map_ds];
                        if(status == SEED){
                            first_empty = 0;
                        } else {
                            first_empty = 1;
                        }

                        for (y_ = 0; y_ < y_num_ds; y_++) {

                            status = p_map[i-1][ offset_part_map_ds + y_];

                            if (status == SEED) {
                                curr_index += 1 + prev_ind;
                                prev_ind = 0;
                                curr_index += 1 + prev_ind;
                            } else {
                                prev_ind = 1;
                            }

                        }

                        if (curr_index == 0) {
                            (*this).data[i][offset_pc_data].resize(
                                    1); //always first adds an extra entry for intialization and extra info
                        } else {

                            (*this).data[i][offset_pc_data].resize(curr_index + 2 - first_empty,
                                                                   0); //gap node to begin, already finishes with a gap node

                        }
                    } else {

                        status = p_map[i][offset_part_map];
                        if((status> 1) & (status < 5)){
                            first_empty = 0;
                        } else {
                            first_empty = 1;
                        }

                        for(y_ = 0;y_ < y_num_;y_++){

                            status = p_map[i][offset_part_map + y_];

                            if((status> 1) & (status < 5)){
                                curr_index+= 1 + prev_ind;
                                prev_ind = 0;
                            } else {
                                prev_ind = 1;
                            }
                        }

                        if(curr_index == 0){
                            (*this).data[i][offset_pc_data].resize(1); //always first adds an extra entry for intialization and extra info
                        } else {

                            (*this).data[i][offset_pc_data].resize(curr_index + 2 - first_empty,0); //gap node to begin, already finishes with a gap node

                        }

                    }


                }
            }

        }

        prev_coord = 0;


        for(uint64_t i = (*this).depth_min;i <= (*this).depth_max;i++){

            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            const unsigned int y_num_ = y_num[i];

            const unsigned int x_num_ds = x_num[i-1];
            const unsigned int z_num_ds = z_num[i-1];
            const unsigned int y_num_ds = y_num[i-1];

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,y_,curr_index,status,prev_ind,prev_coord) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_part_map_ds = (x_/2)*y_num_ds + (z_/2)*y_num_ds*x_num_ds;
                    const size_t offset_part_map = x_*y_num_ + z_*y_num_*x_num_;
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    curr_index = 0;
                    prev_ind = 1;
                    prev_coord = 0;

                    if(i == (*this).depth_max){
                        //initialize the first values type
                        (*this).data[i][offset_pc_data][0] = TYPE_GAP_END;

                        uint64_t y_u;

                        for (y_ = 0; y_ < y_num_ds; y_++) {

                            status = p_map[i - 1][offset_part_map_ds + y_];

                            if (status == SEED) {

                                for (int k = 0; k < 2; ++k) {

                                    y_u = 2*y_ + k;

                                    curr_index++;

                                    //set starting type
                                    if (prev_ind == 1) {
                                        //gap node
                                        //set type
                                        (*this).data[i][offset_pc_data][curr_index - 1] = TYPE_GAP;
                                        (*this).data[i][offset_pc_data][curr_index - 1] |= (y_u << NEXT_COORD_SHIFT);
                                        (*this).data[i][offset_pc_data][curr_index - 1] |= (prev_coord
                                                << PREV_COORD_SHIFT);
                                        (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR
                                                << YP_DEPTH_SHIFT);
                                        (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR
                                                << YM_DEPTH_SHIFT);

                                        curr_index++;
                                    }
                                    prev_coord = y_u;
                                    //set type
                                    (*this).data[i][offset_pc_data][curr_index - 1] = TYPE_PC;

                                    //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);

                                    //set the status
                                    switch (status) {
                                        case SEED: {
                                            (*this).data[i][offset_pc_data][curr_index - 1] |= SEED_SHIFTED;
                                            break;
                                        }
                                        case BOUNDARY: {
                                            (*this).data[i][offset_pc_data][curr_index - 1] |= BOUNDARY_SHIFTED;
                                            break;
                                        }
                                        case FILLER: {
                                            (*this).data[i][offset_pc_data][curr_index - 1] |= FILLER_SHIFTED;
                                            break;
                                        }

                                    }

                                    prev_ind = 0;
                                }
                            } else {
                                //store for setting above
                                if (prev_ind == 0) {
                                    //prev_coord = y_;
                                }

                                prev_ind = 1;

                            }

                        }

                        //Initialize the last value GAP END indicators to no neighbour
                        (*this).data[i][offset_pc_data][(*this).data[i][offset_pc_data].size() - 1] = TYPE_GAP_END;
                        (*this).data[i][offset_pc_data][(*this).data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YP_DEPTH_SHIFT);
                        (*this).data[i][offset_pc_data][(*this).data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YM_DEPTH_SHIFT);



                    } else {

                        //initialize the first values type
                        (*this).data[i][offset_pc_data][0] = TYPE_GAP_END;

                        for (y_ = 0; y_ < y_num_; y_++) {

                            status = p_map[i][offset_part_map + y_];

                            if((status> 1) && (status < 5)) {

                                curr_index++;

                                //set starting type
                                if (prev_ind == 1) {
                                    //gap node
                                    //set type
                                    (*this).data[i][offset_pc_data][curr_index - 1] = TYPE_GAP;
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (y_ << NEXT_COORD_SHIFT);
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (prev_coord << PREV_COORD_SHIFT);
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
                                    (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);

                                    curr_index++;
                                }
                                prev_coord = y_;
                                //set type
                                (*this).data[i][offset_pc_data][curr_index - 1] = TYPE_PC;

                                //initialize the neighbours to empty (to be over-written later if not the case) (Boundary Conditions)
                                (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XP_DEPTH_SHIFT);
                                (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << XM_DEPTH_SHIFT);
                                (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZP_DEPTH_SHIFT);
                                (*this).data[i][offset_pc_data][curr_index - 1] |= (NO_NEIGHBOUR << ZM_DEPTH_SHIFT);

                                //set the status
                                switch (status) {
                                    case seed_us: {
                                        (*this).data[i][offset_pc_data][curr_index - 1] |= SEED_SHIFTED;
                                        break;
                                    }
                                    case BOUNDARY: {
                                        (*this).data[i][offset_pc_data][curr_index - 1] |= BOUNDARY_SHIFTED;
                                        break;
                                    }
                                    case FILLER: {
                                        (*this).data[i][offset_pc_data][curr_index - 1] |= FILLER_SHIFTED;
                                        break;
                                    }

                                }

                                prev_ind = 0;
                            } else {


                                prev_ind = 1;

                            }
                        }

                        //Initialize the last value GAP END indicators to no neighbour
                        (*this).data[i][offset_pc_data][(*this).data[i][offset_pc_data].size() - 1] = TYPE_GAP_END;
                        (*this).data[i][offset_pc_data][(*this).data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YP_DEPTH_SHIFT);
                        (*this).data[i][offset_pc_data][(*this).data[i][offset_pc_data].size() - 1] |= (NO_NEIGHBOUR
                                << YM_DEPTH_SHIFT);
                    }
                }
            }

        }

        timer.stop_timer();


        ///////////////////////////////////
        //
        //  Calculate neighbours
        //
        /////////////////////////////////

        timer.start_timer("set_up_neigh");

        //(+y,-y,+x,-x,+z,-z)
        (*this).set_neighbor_relationships();

        timer.stop_timer();

    }




    void init_from_pulling_scheme(std::vector<MeshData<uint8_t>>& layers){
        //
        //
        //  INITIALIZE THE PARTICLE CELL STRUCTURE FORM THE OUTPUT OF THE PULLING SCHEME
        //
        //

        //INITIALIZE THE DOMAIN SIZES

        (*this).x_num.resize((*this).depth_max+1);
        (*this).y_num.resize((*this).depth_max+1);
        (*this).z_num.resize((*this).depth_max+1);

        for(int i = (*this).depth_min;i < (*this).depth_max;i++){
            (*this).x_num[i] = layers[i].x_num;
            (*this).y_num[i] = layers[i].y_num;
            (*this).z_num[i] = layers[i].z_num;

        }

        (*this).y_num[(*this).depth_max] = (*this).org_dims[0];
        (*this).x_num[(*this).depth_max] = (*this).org_dims[1];
        (*this).z_num[(*this).depth_max] = (*this).org_dims[2];

        //transfer over data-structure to make the same (re-use of function for read-write)

        std::vector<std::vector<uint8_t>> p_map;
        p_map.resize((*this).depth_max);

        for (int k = 0; k < (*this).depth_max; ++k) {
            std::swap(p_map[k],layers[k].mesh);
        }

        (*this).create_partcell_structure(p_map);

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
            
      
        }
        
        
        
    }
    

    
    void get_neigh_0(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
        get_neighs_face_t<0>(curr_key,node_val,neigh_keys);
    }
    
    void get_neigh_1(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
        get_neighs_face_t<1>(curr_key,node_val,neigh_keys);
    }
    
    void get_neigh_2(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
        get_neighs_face_t<2>(curr_key,node_val,neigh_keys);
    }
    
    void get_neigh_3(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
        get_neighs_face_t<3>(curr_key,node_val,neigh_keys);
    }
    
    void get_neigh_4(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
        get_neighs_face_t<4>(curr_key,node_val,neigh_keys);
    }
    
    void get_neigh_5(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
        get_neighs_face_t<5>(curr_key,node_val,neigh_keys);
    }
    
    void get_neighs_axis(const uint64_t& curr_key,uint64_t node_val,PartCellNeigh<uint64_t>& neigh_keys,const unsigned int axis){
        // Selects the neighbour in the correct direction
        
        neigh_keys.curr = curr_key;
        
        switch(axis){
            case(0):{
                //y
                neigh_keys.neigh_face[0].resize(0);
                neigh_keys.neigh_face[1].resize(0);
                
                get_neighs_face_t<0>(curr_key,node_val,neigh_keys.neigh_face[0]);
                get_neighs_face_t<1>(curr_key,node_val,neigh_keys.neigh_face[1]);
                
                break;
            }
            case(1):{
                //x
                neigh_keys.neigh_face[2].resize(0);
                neigh_keys.neigh_face[3].resize(0);
                
                get_neighs_face_t<2>(curr_key,node_val,neigh_keys.neigh_face[2]);
                get_neighs_face_t<3>(curr_key,node_val,neigh_keys.neigh_face[3]);
                
                break;
            }
            case(2):{
                //z
                neigh_keys.neigh_face[4].resize(0);
                neigh_keys.neigh_face[5].resize(0);
                
                get_neighs_face_t<4>(curr_key,node_val,neigh_keys.neigh_face[4]);
                get_neighs_face_t<5>(curr_key,node_val,neigh_keys.neigh_face[5]);
                
                break;
            }
        }
        
        
        
    }
    
    void get_neighs_face(const uint64_t& curr_key,uint64_t node_val,uint64_t face,PartCellNeigh<uint64_t>& neigh_keys){
        // Selects the neighbour in the correct direction

        neigh_keys.curr = curr_key;

        switch(face){
            case 0: {
                neigh_keys.neigh_face[0].resize(0);
                get_neighs_face_t<0>(curr_key,node_val,neigh_keys.neigh_face[0]);
                break;
            }
            case 1: {
                neigh_keys.neigh_face[1].resize(0);
                get_neighs_face_t<1>(curr_key,node_val,neigh_keys.neigh_face[1]);
                break;
            }
            case 2: {
                neigh_keys.neigh_face[2].resize(0);
                get_neighs_face_t<2>(curr_key,node_val,neigh_keys.neigh_face[2]);
                break;
            }
            case 3: {
                neigh_keys.neigh_face[3].resize(0);
                get_neighs_face_t<3>(curr_key,node_val,neigh_keys.neigh_face[3]);
                break;
            }
            case 4: {
                neigh_keys.neigh_face[4].resize(0);
                get_neighs_face_t<4>(curr_key,node_val,neigh_keys.neigh_face[4]);
                break;
            }
            case 5: {
                neigh_keys.neigh_face[5].resize(0);
                get_neighs_face_t<5>(curr_key,node_val,neigh_keys.neigh_face[5]);
                break;
            }

        }

    }

    void get_neighs_face(const uint64_t& curr_key,uint64_t node_val,uint64_t face,std::vector<uint64_t>& neigh_vec){
        // Selects the neighbour in the correct direction


        neigh_vec.resize(0);

        switch(face){
            case 0: {

                get_neighs_face_t<0>(curr_key,node_val,neigh_vec);
                break;
            }
            case 1: {

                get_neighs_face_t<1>(curr_key,node_val,neigh_vec);
                break;
            }
            case 2: {

                get_neighs_face_t<2>(curr_key,node_val,neigh_vec);
                break;
            }
            case 3: {

                get_neighs_face_t<3>(curr_key,node_val,neigh_vec);
                break;
            }
            case 4: {

                get_neighs_face_t<4>(curr_key,node_val,neigh_vec);
                break;
            }
            case 5: {

                get_neighs_face_t<5>(curr_key,node_val,neigh_vec);
                break;
            }

        }

    }
    
    void get_neighs_all(const uint64_t& curr_key,uint64_t node_val,PartCellNeigh<uint64_t>& neigh_keys){
        // Selects the neighbour in the correct direction
        
        neigh_keys.curr = curr_key;
        
        neigh_keys.neigh_face[0].resize(0);
        neigh_keys.neigh_face[1].resize(0);
        neigh_keys.neigh_face[2].resize(0);
        neigh_keys.neigh_face[3].resize(0);
        neigh_keys.neigh_face[4].resize(0);
        neigh_keys.neigh_face[5].resize(0);
        
        get_neighs_face_t<0>(curr_key,node_val,neigh_keys.neigh_face[0]);
        get_neighs_face_t<1>(curr_key,node_val,neigh_keys.neigh_face[1]);
        get_neighs_face_t<2>(curr_key,node_val,neigh_keys.neigh_face[2]);
        get_neighs_face_t<3>(curr_key,node_val,neigh_keys.neigh_face[3]);
        get_neighs_face_t<4>(curr_key,node_val,neigh_keys.neigh_face[4]);
        get_neighs_face_t<5>(curr_key,node_val,neigh_keys.neigh_face[5]);
        
    }
    bool check_parent_edge(std::vector<uint16_t>& coords,uint8_t index){
        //Coords is ordered [y,x,z]
        bool add_parent;

        if (edges_parent_type[index] == 0){
            //xz case
            add_parent = ((coords[1]&1) == edges_parent_ind[index][0]) && ((coords[2]&1) == edges_parent_ind[index][1]);

        } else if(edges_parent_type[index] == 0){
            //yz case
            add_parent = ((coords[0]&1) == edges_parent_ind[index][0]) && ((coords[2]&1) == edges_parent_ind[index][1]);
        } else {
            //yx case
            add_parent = ((coords[0]&1) == edges_parent_ind[index][0]) && ((coords[1]&1) == edges_parent_ind[index][1]);
        }

        return add_parent;

    }


    void get_edge_neighs_all(const uint64_t& curr_key,PartCellNeigh<uint64_t>& neigh_keys_face,PartCellNeigh<uint64_t>& neigh_keys_edge,std::vector<uint16_t>& coords) {
        //
        //  Bevan Cheeseman 2017
        //
        //  Gets the edge neighbours
        //  Coords is ordered [y,x,z]
        //

        neigh_keys_edge.curr = curr_key;

        neigh_keys_edge.neigh_face[0].resize(0);
        neigh_keys_edge.neigh_face[1].resize(0);
        neigh_keys_edge.neigh_face[2].resize(0);
        neigh_keys_edge.neigh_face[3].resize(0);
        neigh_keys_edge.neigh_face[4].resize(0);
        neigh_keys_edge.neigh_face[5].resize(0);
        neigh_keys_edge.neigh_face[6].resize(0);
        neigh_keys_edge.neigh_face[7].resize(0);
        neigh_keys_edge.neigh_face[8].resize(0);
        neigh_keys_edge.neigh_face[9].resize(0);
        neigh_keys_edge.neigh_face[10].resize(0);
        neigh_keys_edge.neigh_face[11].resize(0);

        uint64_t face_key;
        uint64_t curr_node_val;
        uint64_t curr_depth = pc_key_get_depth(curr_key);

        std::vector<uint64_t> neigh_keys_temp;

        for (int i = 0; i < NUM_EDGES; ++i) {
            if(neigh_keys_face.neigh_face[edges_face[i]].size()>1){
                //will be two neighbours
                for (int j = 0; j < 2; ++j) {

                    face_key = neigh_keys_face.neigh_face[edges_face[i]][edges_face_index[i][j]];

                    if(face_key > 0) {

                        curr_node_val = get_val(face_key);


                        get_neighs_face(face_key,curr_node_val,edges_face_dir[i],neigh_keys_temp);

                        if (neigh_keys_temp.size() == 1) {
                            //there can be only one....
                            neigh_keys_edge.neigh_face[i].push_back(neigh_keys_temp[0]);
                        }
                    }

                }


            } else if (neigh_keys_face.neigh_face[edges_face[i]].size() == 1) {
                //same level or parent
                face_key = neigh_keys_face.neigh_face[edges_face[i]][0];

                if(face_key > 0) {

                    curr_node_val = get_val(face_key);

                    get_neighs_face(face_key,curr_node_val,edges_face_dir[i],neigh_keys_temp);

                    if (neigh_keys_temp.size() == 1) {

                        //check if to include the parent or not
                        if(curr_depth < pc_key_get_depth(neigh_keys_temp[0])){
                            //parent need to check
                            if(check_parent_edge(coords,i)){
                                neigh_keys_edge.neigh_face[i].push_back(neigh_keys_temp[0]);
                            }

                        } else {
                            //not parent add
                            neigh_keys_edge.neigh_face[i].push_back(neigh_keys_temp[0]);
                        }

                    } else if (neigh_keys_temp.size() > 1) {
                        neigh_keys_edge.neigh_face[i].push_back(neigh_keys_temp[edges_child_index[i][0]]);
                        neigh_keys_edge.neigh_face[i].push_back(neigh_keys_temp[edges_child_index[i][0]]);
                    }
                }

            }

        }


    }

    bool check_parent_corner(std::vector<uint16_t>& coords,uint8_t index){
        //
        //  Checks if the coordinates are correct, such that this parent is actually the correct neighbour, or has not already been added
        //  Coords is ordered [y,x,z]

        bool add_parent;

        add_parent = ((coords[0]&1) == corner_parent_ind[index][0]) && ((coords[1]&1) == corner_parent_ind[index][1]) && ((coords[2]&1) == corner_parent_ind[index][2]);

        return add_parent;

    }



    void get_corner_neighs_all(const uint64_t& curr_key,PartCellNeigh<uint64_t>& neigh_keys_edge,PartCellNeigh<uint64_t>& neigh_keys_corner,std::vector<uint16_t>& coords) {
        //
        //  Bevan Cheeseman 2017
        //
        //  Gets the edge neighbours
        //  Coords is ordered [y,x,z]
        //

        neigh_keys_corner.curr = curr_key;

        neigh_keys_corner.neigh_face[0].resize(0);


        uint64_t edge_key;
        uint64_t curr_node_val;
        uint64_t curr_depth = pc_key_get_depth(curr_key);

        std::vector<uint64_t> neigh_keys_temp;

        for (int i = 0; i < NUM_CORNERS; ++i) {
            if(neigh_keys_edge.neigh_face[corner_edge[i]].size()>1){
                //will be two neighbours

                edge_key = neigh_keys_edge.neigh_face[corner_edge[i]][corner_edge_index[i]];

                if(edge_key > 0) {

                    curr_node_val = get_val(edge_key);

                    get_neighs_face(edge_key,curr_node_val,corner_edge_dir[i],neigh_keys_temp);

                    if (neigh_keys_temp.size() == 1) {
                        //can only be one

                        neigh_keys_corner.neigh_face[i].push_back(neigh_keys_temp[0]);
                    }

                }


            } else if(neigh_keys_edge.neigh_face[corner_edge[i]].size()==1){
                //same level or parent
                edge_key = neigh_keys_edge.neigh_face[corner_edge[i]][0];

                if(edge_key > 0) {

                    curr_node_val = get_val(edge_key);

                    get_neighs_face(edge_key,curr_node_val,corner_edge_dir[i],neigh_keys_temp);

                    if (neigh_keys_temp.size() == 1) {
                        //can only be one
                        //check if to include the parent or not
                        if(curr_depth < pc_key_get_depth(corner_edge_move_index[i])){
                            //parent need to check
                            if(check_parent_corner(coords,i)){
                                neigh_keys_corner.neigh_face[i].push_back(neigh_keys_temp[corner_edge_move_index[i]]);
                            }

                        } else {
                            //not parent add
                            neigh_keys_corner.neigh_face[i].push_back(neigh_keys_temp[corner_edge_move_index[i]]);
                        }



                    }

                }

            }

        }


    }

    void get_neighs_all_diag(const uint64_t& curr_key,uint64_t node_val,std::vector<PartCellNeigh<uint64_t>>& neigh_keys,std::vector<uint16_t>& coords){
        //
        //  Bevan Cheeseman 2017
        //
        //  Gets all cell neighbours including diagonals
        //
        //  Coords is ordered [y,x,z]
        //

        get_neighs_all(curr_key,node_val,neigh_keys[0]);

        get_edge_neighs_all(curr_key,neigh_keys[0],neigh_keys[1],coords);

        get_corner_neighs_all(curr_key,neigh_keys[1],neigh_keys[2],coords);

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
        PartCellNeigh<uint64_t> neigh_keys;
       
        
        timer.start_timer("get neighbour cells all");
        
        for(uint64_t i = depth_min;i <= depth_max;i++){
            
            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map_inplace)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,neigh_keys) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                pc_key_set_z(curr_key,z_);
                pc_key_set_depth(curr_key,i);
                
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val = data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //get the index gap node
                            
                            pc_key_set_j(curr_key,j_);
                            
                            get_neighs_all(curr_key,node_val,neigh_keys);
                            
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
        
        timer.start_timer("get neighbour cells single face 0");
        
        for(uint64_t i = depth_min;i <= depth_max;i++){
            
            const unsigned int x_num_ = x_num[i];
            const unsigned int z_num_ = z_num[i];
            
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map_inplace)
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,neigh_keys) if(z_num_*x_num_ > 100)
#endif
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                pc_key_set_z(curr_key,z_);
                pc_key_set_depth(curr_key,i);
                
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    pc_key_set_x(curr_key,x_);
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = data[i][offset_pc_data].size();
                    
                    for(j_ = 0;j_ < j_num;j_++){
                        
                        node_val = data[i][offset_pc_data][j_];
                        
                        if (!(node_val&1)){
                            //get the index gap node
                            
                            pc_key_set_j(curr_key,j_);
                            
                            get_neighs_face(curr_key,node_val,0,neigh_keys);
                            
                            
                        } else {
                            
                        }
                        
                    }
                    
                }
                
            }
        }
        
        timer.stop_timer();
        
        
    }
    
    
    void set_neighbor_relationships(){
        //
        // Calls the functions to access the underlying datastructure
        //
        
        set_neighbor_relationships(0);
        set_neighbor_relationships(1);
        set_neighbor_relationships(2);
        set_neighbor_relationships(3);
        set_neighbor_relationships(4);
        set_neighbor_relationships(5);
        
    }


private:
    //  Direction definitions for Particle Cell Neighbours
    //  [+y,-y,+x,-x,+z,-z]
    //  [0,1,2,3,4,5]
    
    
    template<uint8_t face>
    uint64_t get_neighbour_same_level(const uint64_t& curr_key)
    {
        /** Get neighbours of a cell in one of the direction that are guranteed to be on the same level
         *
         *  @param curr_key    input: current key, output: neighbour key
         *  @param face        direction to follow. Possible values are [0,5]
         *                     They stand for [+y,-y,+x,-x,+z,-z] //change this ordering.. (y+ y-) are different,
         */
        
        
        constexpr uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
        constexpr uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
        
        constexpr int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
        constexpr int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
        constexpr int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};
        
        
        //inits
        uint64_t node_val;
        uint64_t neigh_key;
        
        //this is restricted to cells on the same level
        neigh_key = curr_key;
        
        //get the node_val
        if(face < 2){
            //y_values need to use next node
            
            //check if reached end boundary for y
            pc_key_offset_j(neigh_key, von_neumann_y_cells[face]);
            
            return neigh_key;
            
        } else {
            //otherwise
            
            //get the node value
            node_val = get_val(curr_key);
            //set the index
            
            pc_key_set_j(neigh_key, node_get_val(node_val,index_mask_dir[face],index_shift_dir[face]));
            pc_key_offset_x(neigh_key,von_neumann_x_cells[face]);
            pc_key_offset_z(neigh_key,von_neumann_z_cells[face]);
            
        }
        
        return neigh_key;
        
    }
    template<uint64_t face>
    bool check_neigh_exists(uint64_t node_val,uint64_t curr_key){
        //checks if node on same level exists or not
        constexpr uint64_t depth_mask_dir[6] = {YP_DEPTH_MASK,YM_DEPTH_MASK,XP_DEPTH_MASK,XM_DEPTH_MASK,ZP_DEPTH_MASK,ZM_DEPTH_MASK};
        constexpr uint64_t depth_shift_dir[6] =  {YP_DEPTH_SHIFT,YM_DEPTH_SHIFT,XP_DEPTH_SHIFT,XM_DEPTH_SHIFT,ZP_DEPTH_SHIFT,ZM_DEPTH_SHIFT};
        
        constexpr int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
        
        if(face < 2){
            
            //increase key by one
            pc_key_offset_j(curr_key,von_neumann_y_cells[face]);
            
            node_val = get_val(curr_key);
            
            if(!(node_val&1)){
                //same level
                return true;
            } else {
                return false;
            }
            
            
        } else {
            
            if (node_get_val(node_val,depth_mask_dir[face],depth_shift_dir[face]) == NO_NEIGHBOUR){
                return false;
            } else {
                return true;
            }
            
        }
    }

    template<uint64_t face>
    void get_neighs_face_t(const uint64_t& curr_key,uint64_t node_val,std::vector<uint64_t>& neigh_keys){
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
        
        
        constexpr uint64_t depth_mask_dir[6] = {YP_DEPTH_MASK,YM_DEPTH_MASK,XP_DEPTH_MASK,XM_DEPTH_MASK,ZP_DEPTH_MASK,ZM_DEPTH_MASK};
        constexpr uint64_t depth_shift_dir[6] =  {YP_DEPTH_SHIFT,YM_DEPTH_SHIFT,XP_DEPTH_SHIFT,XM_DEPTH_SHIFT,ZP_DEPTH_SHIFT,ZM_DEPTH_SHIFT};
//
        constexpr uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
        constexpr uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
//
        constexpr int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
        constexpr int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
        constexpr int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};
//
//        the ordering of retrieval of four neighbour cells
        constexpr uint8_t neigh_child_dir[6][3] = {{4,2,2},{4,2,2},{0,4,4},{0,4,4},{0,2,2},{0,2,2}};
        
        
        uint64_t neigh_indicator;
        
        uint64_t neigh_key;
        
        uint64_t org_node= node_val;
        
        // +-y direction is different
        if(face < 2){
            
            neigh_key = curr_key;
            
            pc_key_offset_j(neigh_key,von_neumann_y_cells[face]);
            
            node_val = get_val(neigh_key);
            
            if(!(node_val&1)){
                //same level
                neigh_keys.push_back(neigh_key);
                
                return;
            }
            
            
        }
        
        //dir
        neigh_indicator =  node_get_val(node_val,depth_mask_dir[face],depth_shift_dir[face]);
        
        switch(neigh_indicator){
            case(LEVEL_SAME):{
                //same level return single neighbour
                neigh_key = curr_key;
                
                pc_key_set_j(neigh_key, node_get_val(node_val,index_mask_dir[face],index_shift_dir[face]));
                pc_key_offset_x(neigh_key,von_neumann_x_cells[face]);
                pc_key_offset_z(neigh_key,von_neumann_z_cells[face]);
                
                //pc_key_set_status(neigh_key,get_status(get_val(neigh_key)));
                
                neigh_keys.push_back(neigh_key);
                
                return;
            }
            case(LEVEL_DOWN):{
                // Neighbour is on parent level (depth - 1)


                pc_key ck;
                ck.update_cell(curr_key);

                pc_key nk;

                neigh_key = curr_key;
                
                pc_key_set_j(neigh_key, node_get_val(node_val,index_mask_dir[face],index_shift_dir[face]));

                nk.update_cell(neigh_key);

                pc_key_set_x(neigh_key,pc_key_get_x(neigh_key)/2 + von_neumann_x_cells[face]);

                nk.update_cell(neigh_key);

                uint64_t t = pc_key_get_z(neigh_key)/2 + von_neumann_z_cells[face];
                uint64_t zt = pc_key_get_z(neigh_key);
                uint64_t zt2 = pc_key_get_z(neigh_key)/2;
                uint64_t z3 = von_neumann_z_cells[face];

                pc_key_set_z(neigh_key,pc_key_get_z(neigh_key)/2 + von_neumann_z_cells[face]);

                nk.update_cell(neigh_key);

                pc_key_offset_depth(neigh_key,-1);

                nk.update_cell(neigh_key);

               // pc_key_set_status(neigh_key,get_status(get_val(neigh_key)));
                neigh_keys.push_back(neigh_key);


                nk.update_cell(neigh_key);



                return;
            }
            case(LEVEL_UP):{
                // Neighbour is on a lower child level
                
                //first of four children
                
                neigh_key = curr_key;
                pc_key_offset_depth(neigh_key,1);
                pc_key_set_j(neigh_key, node_get_val(node_val,index_mask_dir[face],index_shift_dir[face]));
                pc_key_set_x(neigh_key,(pc_key_get_x(neigh_key) + von_neumann_x_cells[face])*2 + (von_neumann_x_cells[face] < 0));
                pc_key_set_z(neigh_key,(pc_key_get_z(neigh_key) + von_neumann_z_cells[face])*2 + (von_neumann_z_cells[face] < 0));
                
                //pc_key_set_status(neigh_key,get_status(get_val(neigh_key)));
                neigh_keys.push_back(neigh_key);
                
                uint64_t temp = neigh_key;
                
                //check if its two neighbours exist
                bool exist0 = check_neigh_exists<neigh_child_dir[face][0]>(org_node,neigh_key);
                bool exist2 = check_neigh_exists<neigh_child_dir[face][2]>(org_node,neigh_key);
                
                //changed the ordering
                
                if(exist0){
                    neigh_key = get_neighbour_same_level<neigh_child_dir[face][0]>(neigh_key);
                    //pc_key_set_status(neigh_key,get_status(get_val(neigh_key)));
                    neigh_keys.push_back(neigh_key);
                    
                } else {
                    neigh_keys.push_back(0);
                }
                //diagonal will exist only if the other two exist
                
                if(exist2){
                    temp = get_neighbour_same_level<neigh_child_dir[face][2]>(temp);
                    //pc_key_set_status(temp,get_status(get_val(temp)));
                    neigh_keys.push_back(temp);
                } else {
                    neigh_keys.push_back(0);
                }
                
                if(exist0 & exist2){
                    neigh_key = get_neighbour_same_level<neigh_child_dir[face][1]>(neigh_key);
                    //pc_key_set_status(neigh_key,get_status(get_val(neigh_key)));
                    neigh_keys.push_back(neigh_key);
                } else {
                    neigh_keys.push_back(0);
                }
                
                return;
            }
        }
        
        
        
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
        
        const int8_t y_offset = y_offset_vec[face];
        
        
        Part_timer timer;
        timer.verbose_flag = false;
        uint64_t z_;
        uint64_t x_;
        uint64_t j_;
        
        timer.start_timer("Get neighbours dir: " + std::to_string(face));
        
        int y_neigh;
        int y_parent;
        uint64_t j_parent;
        uint64_t j_neigh;
        
        uint64_t node_val;
        uint64_t node_val_parent;
        int y_coord;
        
        if (face > 1){
            
            for(uint64_t i = (depth_min);i <= depth_max;i++){
                
                const unsigned int x_num_ = x_num[i];
                const unsigned int z_num_ = z_num[i];
                
                const unsigned int x_num_parent = x_num[i-1];
                
                if (i == depth_min){
                    
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord) if(z_num_*x_num_ > 100)
#endif
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
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,j_neigh,y_neigh,y_coord) if(z_num_*x_num_ > 100)
#endif
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
                                        if((y_coord == y_parent*2) & (x_ == ((x_parent-x_offset)*2 + (x_offset > 0))) & (z_ == ((z_parent-z_offset)*2 + (z_offset > 0)))){
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
            
            for(uint64_t i = (depth_min+1);i <= depth_max;i++){
                
                const unsigned int x_num_ = x_num[i];
                const unsigned int z_num_ = z_num[i];
                
                const unsigned int x_num_parent = x_num[i-1];
                
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,y_parent,j_parent,y_coord,node_val_parent) if(z_num_*x_num_ > 100)
#endif
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
                        
                        for(j_ = 0;j_ < j_num;j_++){
                            
                            // Parent relation
                            
                            node_val = data[i][offset_pc_data][j_];
                            
                           
                            
                            if (node_val&1){
                                //get the index gap node
                                
                                if(face == 0){
                                    
                                    //iterate parent
                                    while ((y_parent < (y_coord+y_offset)/2) & (j_parent < (j_num_parent-1))){
                                        
                                        j_parent++;
                                        node_val_parent = data[i-1][offset_pc_data_parent][j_parent];
                                        
                                        if (node_val_parent&1){
                                            //get the index gap node
                                            
                                            //if(((node_val_parent & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT) > 0){
                                                y_parent = (node_val_parent & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                           // } else {
                                             //   y_parent = -2;
                                           // }
                                            
                                            j_parent++;
                                            
                                        } else {
                                            //normal node
                                            y_parent++;
                                            
                                        }
                                    }
                                    
                                    if(((y_coord+y_offset)/2 == y_parent) & (y_coord >= 0)){
                                        data[i][offset_pc_data][j_] |= (j_parent << index_shift_0);
                                        data[i][offset_pc_data][j_] &= -((depth_mask_0)+1);
                                        data[i][offset_pc_data][j_] |= (  LEVEL_DOWN  << depth_shift_0);
                                        //symmetric (only add it once)
                                        if((y_coord == ((y_parent-y_offset)*2 + (y_offset > 0))) & (x_ == x_parent*2) & (z_ == (z_parent*2) )){
                                            
                                           data[i-1][offset_pc_data_parent][j_parent-y_offset] |= ( (j_-y_offset) << index_shift_1);
                                            
                                           data[i-1][offset_pc_data_parent][j_parent-y_offset] &= -((depth_mask_1)+1);
                                           data[i-1][offset_pc_data_parent][j_parent-y_offset] |= ( LEVEL_UP  << depth_shift_1);
                                        
                                        }
                                    } else {
                                        //end node
                                    }
                                    
                                    y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                    y_coord--;
                                    
                                    
                                    
                                } else {
                                    
                                    y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                    
                                    //iterate parent
                                    while ((y_parent < (y_coord+y_offset)/2) & (j_parent < (j_num_parent-1))){
                                        
                                        j_parent++;
                                        node_val_parent = data[i-1][offset_pc_data_parent][j_parent];
                                        
                                        if (node_val_parent&1){
                                            //get the index gap node
                                            
                                            //if(((node_val_parent & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT) > 0){
                                            y_parent = (node_val_parent & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                           // } else {
                                            //    y_parent = -2;
                                            //}
                                            
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
                                        if((y_coord == ((y_parent-y_offset)*2 + (y_offset > 0))) & (x_ == x_parent*2) & (z_ == (z_parent*2) )){
                                            data[i-1][offset_pc_data_parent][j_parent-y_offset] |= ( (j_-y_offset) << index_shift_1);
                                            
                                            data[i-1][offset_pc_data_parent][j_parent-y_offset] &= -((depth_mask_1)+1);
                                            data[i-1][offset_pc_data_parent][j_parent-y_offset] |= ( LEVEL_UP  << depth_shift_1);
                                            
                                            
                                        }
                                        
                                        
                                        
                                    } else {
                                        //end node
                                    }
                                    
                                    
                                    y_coord--;
                                    
                                    

                                }
                                
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                //iterate parent
                                while ((y_parent < (y_coord+y_offset)/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val_parent = data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val_parent&1){
                                        //get the index gap node
                                        //if(((node_val_parent & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT) > 0){
                                            y_parent = (node_val_parent & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        //} else {
                                           // y_parent = -2;
                                        //}
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
    
    
};

template<typename T> constexpr uint64_t PartCellData<T>::depth_mask_dir[];
template<typename T> constexpr uint64_t PartCellData<T>::depth_shift_dir[];

template<typename T> constexpr uint64_t PartCellData<T>::index_mask_dir[];
template<typename T> constexpr uint64_t PartCellData<T>::index_shift_dir[];

template<typename T> constexpr int8_t PartCellData<T>::von_neumann_y_cells[];
template<typename T> constexpr int8_t PartCellData<T>::von_neumann_x_cells[];
template<typename T> constexpr int8_t PartCellData<T>::von_neumann_z_cells[];

template<typename T> constexpr uint8_t PartCellData<T>::neigh_child_dir[][3];

template<typename T> constexpr uint8_t PartCellData<T>::neigh_child_y_offsets[][4];
template<typename T> constexpr uint8_t PartCellData<T>::x_start_vec[];
template<typename T> constexpr uint8_t PartCellData<T>::x_stop_vec[];

template<typename T> constexpr uint8_t PartCellData<T>::z_start_vec[];
template<typename T> constexpr uint8_t PartCellData<T>::z_stop_vec[];

template<typename T> constexpr uint8_t PartCellData<T>::y_start_vec[];
template<typename T> constexpr uint8_t PartCellData<T>::y_stop_vec[];

template<typename T> constexpr int8_t PartCellData<T>::x_offset_vec[];
template<typename T> constexpr int8_t PartCellData<T>::z_offset_vec[];
template<typename T> constexpr int8_t PartCellData<T>::y_offset_vec[];

template<typename T> constexpr uint64_t PartCellData<T>::index_shift_dir_sym[];
template<typename T> constexpr uint64_t PartCellData<T>::depth_shift_dir_sym[];

template<typename T> constexpr uint64_t PartCellData<T>::index_mask_dir_sym[];
template<typename T> constexpr uint64_t PartCellData<T>::depth_mask_dir_sym[];

template<typename T> constexpr uint64_t PartCellData<T>::next_prev_mask_vec[];
template<typename T> constexpr uint64_t PartCellData<T>::next_prev_shift_vec[];

template<typename T> constexpr uint8_t PartCellData<T>::seed_part_y[];
template<typename T> constexpr uint8_t PartCellData<T>::seed_part_x[];
template<typename T> constexpr uint8_t PartCellData<T>::seed_part_z[];

template<typename T> constexpr uint8_t PartCellData<T>::edges_face[];
template<typename T> constexpr uint8_t PartCellData<T>::edges_face_index[][2];
template<typename T> constexpr uint8_t PartCellData<T>::edges_face_dir[];
template<typename T> constexpr uint8_t PartCellData<T>::edges_child_index[][2];
template<typename T> constexpr uint8_t PartCellData<T>::edges_parent_ind[][2];
template<typename T> constexpr uint8_t PartCellData<T>::edges_parent_type[];

template<typename T> constexpr uint8_t PartCellData<T>::corner_edge[];
template<typename T> constexpr uint8_t PartCellData<T>::corner_edge_index[];
template<typename T> constexpr uint8_t PartCellData<T>::corner_edge_dir[];
template<typename T> constexpr uint8_t PartCellData<T>::corner_edge_move_index[];
template<typename T> constexpr uint8_t PartCellData<T>::corner_parent_ind[][3];


#endif //PARTPLAY_PARTCELLDATA_HPP