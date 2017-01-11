///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_NEIGHITERATOR_HPP
#define PARTPLAY_NEIGHITERATOR_HPP
// type T data structure base type

#define SAME_LEVEL 1
#define PARENT_LEVEL 2
#define CHILD_LEVEL 3


#include "../../data_structures/Tree/PartCellStructure.hpp"
#include "CurrLevel.hpp"

template<typename V,typename T>
class NeighIterator {
private:
    
    const int8_t child_y[6] = { 1, 0, 0, 1, 0, 1};
    const int8_t child_x[6] = { 0, 1, 1, 0, 0, 1};
    const int8_t child_z[6] = { 0, 1, 0, 1, 1, 0};
    
    //the ordering of retrieval of four neighbour cells
    const uint8_t neigh_child_dir[6][3] = {{2,4,1},{2,4,1},{0,4,3},{0,4,3},{0,2,5},{0,2,5}};
    
    T depth_it;
    T x_it;
    T z_it;
    T j_it;
    T pc_offset_it;
    int y_it;
    T j_num_it;
    T x_num_it;
    T z_num_it;
    T part_offset_it;
    T node_val_it;
    T status_it;
    
    //offsets
    const int offset_x;
    const int offset_z;
    const int offset_y;
    const int offset_depth;
    
    const int child_offset_x;
    const int child_offset_y;
    const int child_offset_z;
    
    
    const int type; // 0 - same, 1 - parent, -1 - child_1, -2 - child 2, -3 child 3, -4 child 4
    
    const float depth_factor;
    const float depth_factor_local;
    
    bool active_depth;
    bool active_row;
    
    
    uint64_t not_exist;
    
public:
    
    int current_flag;
    
    NeighIterator(int type,int offset_x,int offset_z,int offset_y): type(type), offset_x(offset_x), offset_z(offset_z),offset_y(offset_y){
        
        if(type < 0){
            //child iterator
            depth_factor = 0.5;
            offset_depth = 1;
            
            if(type == -1){
                
                child_offset_x = 0;
                child_offset_z = 0;
                child_offset_y = 0;
                
            } else {
                int dir = (offset_y == -1) + (offset_x == 1)*2 + (offset_x == -1)*3 + (offset_z == 1)*4 + (offset_z == -1)*5;
                
                int child_num = -type - 2;
                
                child_offset_x = child_x[neigh_child_dir[dir][child_num]];
                child_offset_z = child_z[neigh_child_dir[dir][child_num]];
                child_offset_y = child_y[neigh_child_dir[dir][child_num]];
                
            }
            
        } else if (type == 0) {
            // same iterator
            depth_factor = 1.0;
            offset_depth = 0;
            
            child_offset_x = 0;
            child_offset_y = 0;
            child_offset_z = 0;
            
        } else if (type == 1){
            // parent iterator
            depth_factor = 2.0;
            offset_depth = -1;
            
            child_offset_x = 0;
            child_offset_y = 0;
            child_offset_z = 0;
            
        }
        
        
    };
    
    
    template<typename U>
    void set_new_depth(T curr_depth,ParticleDataNew<U, T>& part_data){
        
        depth_it = curr_depth + offset_depth;
        x_num_it = part_data.access_data.x_num[depth_it];
        z_num_it = part_data.access_data.z_num[depth_it];
        
        //put in case for parent or child/active deactive here
        
        if( (depth_it > part_data.depth_max) | (depth_it < part_data.depth_min) ){
            //this iterator is not active for this depth
            active_depth = false;
            current_flag = 0;
        } else{
            active_depth = true;
        }
        
    }
    
    template<typename U>
    void incriment_y(const int current_y,const ParticleDataNew<U, T>& part_data){
        //
        //  Incriments the array, according to the desired offset, returns true, if the offset value exist on the layer.
        //
        
        current_flag = 0;
        //need to deal with offset and
        int y_input = depth_factor*(current_y + offset_y) + child_offset_y;
        
        //iterate forward
        while ((y_it < y_input) & (j_it < (j_num_it-1))){
            
            j_it++;
            node_val_it = part_data.access_data.data[depth_it][pc_offset_it][j_it];
            
            if (node_val_it&1){
                //get the index gap node
                y_it += (node_val_it & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE;
                j_it++;
                
                if(j_it < j_num_it){
                    node_val_it = part_data.access_data.data[depth_it][pc_offset_it][j_it];
                    status_it = ((node_val_it & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                    part_offset_it = ((node_val_it & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
                }
                
            } else {
                //normal node
                y_it++;
                status_it = ((node_val_it & STATUS_MASK_PARTICLE) >> STATUS_SHIFT_PARTICLE);
                
                part_offset_it = ((node_val_it & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE);
            }
            
        }
        
        if (y_it == y_input){
            current_flag = 1;
        }
        
        
        
        
    }
    
    template<typename U>
    void new_row(CurrentLevel<U,T>& curr_level,ParticleDataNew<U, T>& part_data){
        //
        //  Updates for new a new row
        //
        
        x_it = (curr_level.x + offset_x)*depth_factor + child_offset_x;
        z_it = (curr_level.z + offset_z)*depth_factor + child_offset_z;
        
        if ((z_it < 0) | (z_it >= z_num_it) | (z_it < 0) | (z_it >= z_num_it)){
            //outside of boundaries
            active_row = false;
            return;
        } else {
            active_row = true;
        }
        
        part_offset_it = x_num_it*z_it + x_it;
        y_it = -1; //initialize
        
    }

    
    
};

#endif //PARTPLAY_NEIGHITERATOR_HPP