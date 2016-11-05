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
#include "../particle_map.hpp"

template <typename T> // type T data structure base type
class PartCellData {
    
public:
    
    /*
     * Number of layers without the root and the contents.
     */
    
    
    const uint64_t depth_mask_dir[6] = {YP_DEPTH_MASK,YM_DEPTH_MASK,XP_DEPTH_MASK,XM_DEPTH_MASK,ZP_DEPTH_MASK,ZM_DEPTH_MASK};
    const uint64_t depth_shift_dir[6] =  {YP_DEPTH_SHIFT,YM_DEPTH_SHIFT,XP_DEPTH_SHIFT,XM_DEPTH_SHIFT,ZP_DEPTH_SHIFT,ZM_DEPTH_SHIFT};
    
    const uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
    const uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
    
    const int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
    const int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
    const int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};
    
    //the ordering of retrieval of four neighbour cells
    const uint8_t neigh_child_dir[6][3] = {{4,2,2},{4,2,2},{0,4,4},{0,4,4},{0,2,2},{0,2,2}};
    
    const uint8_t neigh_child_y_offsets[6][4] = {{0,0,0,0},{0,0,0,0},{0,1,0,1},{0,1,0,1},{0,1,0,1},{0,1,0,1}};
    
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
    
    
    const uint8_t seed_part_y[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    const uint8_t seed_part_x[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const uint8_t seed_part_z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    
    uint8_t depth_max;
    uint8_t depth_min;
    
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
    
    std::vector<std::vector<std::vector<T>>> data;
    
    PartCellData(){};
    
    /////////////////////////////////////////
    //
    //  Get and Set methods for PC_KEYS
    //
    //////////////////////////////////////////
    

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

    
    
    T&  get_val(const uint64_t& pc_key){
        // data access
        
        const uint64_t depth = (pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t j_ = (pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
       
        
        return data[depth][x_num[depth]*z_ + x_][j_];
        
        //return data[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT][x_num[(pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT]*((pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT) + ((pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT)][(pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT];
    }
    
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
                T temp1 = (von_neumann_y_cells[face] < 0);
                T temp2 = neigh_child_y_offsets[face][index];
                neigh_y = (current_y + von_neumann_y_cells[face])*2 +  (von_neumann_y_cells[face] < 0) + neigh_child_y_offsets[face][index];
            }
            
            
        } else {
            neigh_y = 0;
            neigh_x = 0;
            neigh_z = 0;
            neigh_depth = 0;
        }
        
        
    };
//
//    void get_neigh_coordinates_part(T face,T index,T current_y,T& neigh_y,T& neigh_x,T& neigh_z,T& neigh_depth,T status){
//        //
//        //  Get the coordinates for a particle
//        //
//        //
//        
//        T neigh = neigh_face[face][index];
//        
//        if(neigh > 0){
//            
//            if(status > SEED){
//                
//                neigh_x = pc_data.pc_key_get_x(neigh);
//                neigh_z = pc_data.pc_key_get_z(neigh);
//                neigh_depth = pc_data.pc_key_get_depth(neigh);
//                
//                T curr_depth = pc_data.pc_data.pc_key_get_depth(curr);
//                
//                if(neigh_depth == curr_depth){
//                    //neigh is on same layer
//                    neigh_y = current_y + pc_data.von_neumann_y_cells[face];
//                }
//                else if (neigh_depth > curr_depth){
//                    //neigh is on parent layer
//                    neigh_y = (current_y + pc_data.von_neumann_y_cells[face])/2;
//                }
//                else{
//                    //neigh is on child layer
//                    neigh_y = (current_y + pc_data.von_neumann_y_cells[face])*2 + pc_data.neigh_child_y_offsets[face][index];
//                }
//                
//            } else {
//                
//                
//                
//                T part_num = pc_data.pc_key_get_partnum(neigh);
//                
//                neigh_x = pc_data.pc_key_get_x(neigh)*2 + pc_data.seed_part_x(part_num);
//                neigh_z = pc_data.pc_key_get_z(neigh)*2 + pc_data.seed_part_z(part_num);
//                
//                //check if still in the same cell or not
//                if(pc_key_cell_isequal(curr_key,neigh)){
//                    
//                    
//                } else {
//                    
//                    T neigh_status = pc_data.get_status(pc_data.get_val(neigh));
//                    T curr_depth = pc_data.pc_data.pc_key_get_depth(curr);
//                    
//                    if(neigh_depth == curr_depth){
//                        //neigh is on same layer
//                        neigh_y = current_y + pc_data.von_neumann_y_cells[face];
//                    }
//                    else if (neigh_depth > curr_depth){
//                        //neigh is on parent layer
//                        neigh_y = (current_y + pc_data.von_neumann_y_cells[face])/2;
//                    }
//                    else{
//                        //neigh is on child layer
//                        neigh_y = (current_y + pc_data.von_neumann_y_cells[face])*2 + pc_data.neigh_child_y_offsets[face][index];
//                    }
//                }
//                
//                neigh_y = neigh_y*2 + pc_data.seed_part_y(part_num);
//            }
//        } else {
//            neigh_y = 0;
//            neigh_x = 0;
//            neigh_z = 0;
//            neigh_depth = 0;
//        }
//        
//    }
    
    
    
    
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
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,neigh_keys) if(z_num_*x_num_ > 100)
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
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,neigh_keys) if(z_num_*x_num_ > 100)
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
    
    
    template<uint64_t face>
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
    
    bool check_neigh_exists(uint64_t node_val,uint64_t curr_key,uint8_t face){
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
        
        constexpr uint64_t index_mask_dir[6] = {YP_INDEX_MASK,YM_INDEX_MASK,XP_INDEX_MASK,XM_INDEX_MASK,ZP_INDEX_MASK,ZM_INDEX_MASK};
        constexpr uint64_t index_shift_dir[6] = {YP_INDEX_SHIFT,YM_INDEX_SHIFT,XP_INDEX_SHIFT,XM_INDEX_SHIFT,ZP_INDEX_SHIFT,ZM_INDEX_SHIFT};
        
        constexpr int8_t von_neumann_y_cells[6] = { 1,-1, 0, 0, 0, 0};
        constexpr int8_t von_neumann_x_cells[6] = { 0, 0, 1,-1, 0, 0};
        constexpr int8_t von_neumann_z_cells[6] = { 0, 0, 0, 0, 1,-1};
        
        //the ordering of retrieval of four neighbour cells
        constexpr uint8_t neigh_child_dir[6][3] = {{4,2,2},{4,2,2},{0,4,4},{0,4,4},{0,2,2},{0,2,2}};
        
        
        uint64_t neigh_indicator;
        
        uint64_t neigh_key;
        
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
                
                neigh_key = curr_key;
                
                pc_key_set_j(neigh_key, node_get_val(node_val,index_mask_dir[face],index_shift_dir[face]));
                pc_key_set_x(neigh_key,pc_key_get_x(neigh_key)/2 + von_neumann_x_cells[face]);
                pc_key_set_z(neigh_key,pc_key_get_z(neigh_key)/2 + von_neumann_z_cells[face]);
                
                pc_key_offset_depth(neigh_key,-1);
                
               // pc_key_set_status(neigh_key,get_status(get_val(neigh_key)));
                neigh_keys.push_back(neigh_key);
                
                
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
                bool exist0 = check_neigh_exists(node_val,neigh_key,neigh_child_dir[face][0]);
                bool exist2 = check_neigh_exists(node_val,neigh_key,neigh_child_dir[face][2]);
                
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
        
        const uint64_t next_prev_mask = next_prev_mask_vec[face];
        const uint64_t next_prev_shift= next_prev_shift_vec[face];
        const int8_t y_offset = y_offset_vec[face];
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
                                
                                if(((node_val & next_prev_mask) >> next_prev_shift) > 0){
                                    y_coord = (node_val & next_prev_mask) >> next_prev_shift;
                                } else {
                                    y_coord = -1;
                                }
                                //iterate parent
                                while ((y_parent < (y_coord+y_offset)/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        
                                        if(((node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT) > 0){
                                            y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        } else {
                                            y_parent = -2;
                                        }
                                        
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
                                
                                
                                
                            } else {
                                //normal node
                                y_coord++;
                                
                                //iterate parent
                                while ((y_parent < (y_coord+y_offset)/2) & (j_parent < (j_num_parent-1))){
                                    
                                    j_parent++;
                                    node_val = data[i-1][offset_pc_data_parent][j_parent];
                                    
                                    if (node_val&1){
                                        //get the index gap node
                                        if(((node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT) > 0){
                                            y_parent = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                                        } else {
                                            y_parent = -2;
                                        }
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

#endif //PARTPLAY_PARTCELLDATA_HPP