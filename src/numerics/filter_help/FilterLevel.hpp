///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for iterating over neighbour arrays
//
///////////////

#ifndef PARTPLAY_PARTCELLFILTERLEVEL_HPP
#define PARTPLAY_PARTCELLFILTERLEVEL_HPP
// type T data structure base type

#include "../../data_structures/Tree/PartCellStructure.hpp"

template<typename T,typename V>
class FilterLevel {
    
public:
    
    inline T j_num_(){
        return j_num;
    }
    
    inline T y_(){
        return y;
    }
    
    inline T depth_(){
        return depth;
    }
    
    inline T pc_offset_(){
        return pc_offset;
    }
    
    inline T status_(){
        return status;
    }
    
    inline T part_offset_(){
        return part_offset;
    }
    
    FilterLevel(){
        depth = 0;
        x = 0;
        z = 0;
        j = 0;
        pc_offset = 0;
        y = 0;
        j_num = 0;
        x_num = 0;
        z_num = 0;
        
    };
    
    
    template<typename U>
    void set_new_depth(T depth_,PartCellStructure<U,T>& pc_struct){
        
        depth = depth_;
        x_num = pc_struct.x_num[depth];
        z_num = pc_struct.x_num[depth];
        
        depth_factor = pow(2,pc_struct.depth_max - depth + 1);
        
    }
    
    
    template<typename U>
    void set_new_xz(T x_,T z_,PartCellStructure<U,T>& pc_struct){
        
        x_global = x_*depth_factor;
        z_global = x_*depth_factor;
        
        x = x_;
        z = z_;
        
        pc_offset = x_num*z + x;
        j_num = pc_struct.pc_data.data[depth][pc_offset].size();
        part_offset = 0;
        y = 0;
        y_global = 0;
        
    }
    
    template<typename U>
    bool new_j(T j_,PartCellStructure<U,T>& pc_struct){
        j = j_;
        node_val = pc_struct.part_data.access_data.data[depth][pc_offset][j_];
        
        //returns if it is a cell or not
        return !(node_val&1);
        
    }
    
    template<typename U>
    void update_cell(PartCellStructure<U,T>& pc_struct){
        
        status = pc_struct.part_data.access_node_get_status(node_val);
        
        y++;
        
        y_global = y*depth_factor; // on seed level
        
        //seed offset accoutns for which (x,z) you are doing
        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val);
        
    }
    
    template<typename U>
    void update_gap(PartCellStructure<U,T>& pc_struct){
        
        y += ((node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
        y--;
        y_global = y*depth_factor;
    }
    
    template<typename U>
    void initialize_temp_vecs(std::vector<V>& filter_input,PartCellStructure<U,T>& pc_struct){
        
        filter.resize(filter_input.size());
        
        std::copy(filter_input.begin(),filter_input.end(),filter.begin());
        
        temp_vec_s0.resize(filter.size(),0);
        temp_vec_s1.resize(filter.size(),0);
        temp_vec_s2.resize(filter.size(),0);
        temp_vec_s3.resize(filter.size(),0);
        
        temp_vec_ns.resize(filter.size(),0);
        
    }
    
    void iterate_y_seed(){
        //
        //  Moving forward through the particle in the cell right
        //
        
        part_offset++;
        y++;
        y_global = y*depth_factor;
    }
    
    void iterate_temp_vecs(){
        //
        //  Iterates forward these arrays
        //
        //  Copying the last value in
        //
        
        //shuffle values to the left, and then place end value, copied to end. Repeat for both
        std::rotate(temp_vec_s0.begin(),temp_vec_s0.begin() + 1,temp_vec_s0.end());
        temp_vec_s0.back() = temp_vec_s0[0];
        
        std::rotate(temp_vec_s1.begin(),temp_vec_s1.begin() + 1,temp_vec_s1.end());
        temp_vec_s1.back() = temp_vec_s1[0];
        
        std::rotate(temp_vec_s2.begin(),temp_vec_s2.begin() + 1,temp_vec_s2.end());
        temp_vec_s2.back() = temp_vec_s2[0];
        
        std::rotate(temp_vec_ns.begin(),temp_vec_ns.begin() + 1,temp_vec_ns.end());
        temp_vec_s3.back() = temp_vec_s3[0];
        
        std::rotate(temp_vec_ns.begin(),temp_vec_ns.begin() + 1,temp_vec_ns.end());
        temp_vec_ns.back() = temp_vec_ns[0];
        
    }
    
    void compute_filter(ExtraPartCellData<V>& filter_output){
        
        
        if(status ==SEED){
            //perform the filter
            
            for(uint64_t f = 0;f < filter.size();f++){
                filter_output.data[depth][pc_offset][part_offset] += temp_vec_s0[f]*filter[f];
            }
            
            for(uint64_t f = 0;f < filter.size();f++){
                filter_output.data[depth][pc_offset][part_offset+2] += temp_vec_s1[f]*filter[f];
            }
            
            for(uint64_t f = 0;f < filter.size();f++){
                filter_output.data[depth][pc_offset][part_offset+4] += temp_vec_s2[f]*filter[f];
            }
            
            for(uint64_t f = 0;f < filter.size();f++){
                filter_output.data[depth][pc_offset][part_offset+6] += temp_vec_s3[f]*filter[f];
            }
        } else {
            // not seed
            for(uint64_t f = 0;f < filter.size();f++){
                filter_output.data[depth][pc_offset][part_offset] += temp_vec_ns[f]*filter[f];
            }
            
        }

        
        
        
    }
    
    
    //seed temp vectors
    std::vector<V> temp_vec_s0;
    std::vector<V> temp_vec_s1;
    std::vector<V> temp_vec_s2;
    std::vector<V> temp_vec_s3;
    
    //non seed vectors
    std::vector<V> temp_vec_ns;
    
    T x_global;
    T z_global;
    T y_global;
    
    T status;
    
private:
    
    T j_num;
    T pc_offset;
    T depth;
    
    T x;
    T z;
    T j;

    T y;
    
    T x_num;
    T z_num;
    T part_offset;
    T node_val;
    

    //offsets

    float depth_factor;
    float depth_factor_local;
  
    
    std::vector<V> filter;
    
    T filter_offset;
    
};

#endif //PARTPLAY_PARTCELLOFFSET_HPP