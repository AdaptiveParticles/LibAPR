///////////////////
//
//  Bevan Cheeseman 2016
//
//  PartCellData class, the data container for CRS sparse format for APR
//
///////////////

#ifndef PARTPLAY_PARTCELLDATA_HPP
#define PARTPLAY_PARTCELLDATA_HPP

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
        depth_max = part_cell_data.k_max;
        depth_min = part_cell_data.k_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        
        data.resize(depth_max+1);
        
        for(int i = depth_min;i <= depth_max;i++){
            z_num[i] = part_cell_data.z_num[i];
            x_num[i] = part_cell_data.x_num[i];
            data[i].resize(z_num[i]*x_num[i]);
        }

    }

    
    
private:
    
    
                           
    
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP