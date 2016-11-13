///////////////////
//
//  Bevan Cheeseman 2016
//
//     Class for storing extra cell or particle data, that can then be accessed using the access data from pc_structure or parent cells
//
///////////////

#ifndef PARTPLAY_EXTRAPARTCELLDATA_HPP
#define PARTPLAY_EXTRAPARTCELLDATA_HPP
// type T data structure base type

#include "PartCellData.hpp"
#include "ParticleData.hpp"

template<typename T>
class ExtraPartCellData {
    
public:
    
    //the neighbours arranged by face
    
    
    
    ExtraPartCellData(){
    };
    
    ExtraPartCellData(PartCellData<T>& pc_data){
        initialize_structure(pc_data);
    };
    
    
    uint64_t depth_max;
    uint64_t depth_min;
    
    std::vector<uint64_t> z_num;
    std::vector<uint64_t> x_num;
    
    std::vector<std::vector<std::vector<T>>> data;
    
    void initialize_structure(PartCellData<T>& pc_data){
        //
        //  Initialize the structure to the same size as the given structure
        //
        
        //first add the layers
        depth_max = pc_data.depth_max;
        depth_min = pc_data.depth_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        
        data.resize(depth_max+1);
        
        for(int i = depth_min;i <= depth_max;i++){
            z_num[i] = pc_data.z_num[i];
            x_num[i] = pc_data.x_num[i];
            data[i].resize(z_num[i]*x_num[i]);
            
            for(int j = 0;j < pc_data.data[i].size();j++){
                data[i][j].resize(pc_data.data[i][j].size(),0);
            }
            
        }
        
    }
    
    T&  get_val(const uint64_t& pc_key){
        // data access
        
        const uint64_t depth = (pc_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (pc_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (pc_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t j_ = (pc_key & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
        
        
        return data[depth][x_num[depth]*z_ + x_][j_];
    }
    T&  get_part(const uint64_t part_key){
        // data access
        
        const uint64_t depth = (part_key & PC_KEY_DEPTH_MASK) >> PC_KEY_DEPTH_SHIFT;
        const uint64_t x_ = (part_key & PC_KEY_X_MASK) >> PC_KEY_X_SHIFT;
        const uint64_t z_ = (part_key & PC_KEY_Z_MASK) >> PC_KEY_Z_SHIFT;
        const uint64_t index = (part_key & PC_KEY_INDEX_MASK) >> PC_KEY_INDEX_SHIFT;
        
        return data[depth][x_num[depth]*z_ + x_][index];
        
    }
    
private:
    
};

#endif //PARTPLAY_PARTNEIGH_HPP