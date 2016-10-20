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

template <typename T> // type T data structure base type
class PartCellData {
    
public:
    
    /*
     * Number of layers without the root and the contents.
     */
    
    
    PartCellData(){};
    
    T& operator ()(int depth, int z_,int x_,int j_){
        // data access
        return data[depth][z_num[depth]*x_ + z_][j_];
    }
    
    T& operator ()(const PartCellKey& key){
        // data access
        return data[depth][z_num[key.depth]*key.x + key.z][key.j];
    }

    
    
private:
    uint8_t depth;
                           
    std::vector<unsigned int> z_num;
    std::vector<unsigned int> x_num;
                           
    std::vector<std::vector<T>> data;
    
};

#endif //PARTPLAY_PARTCELLDATA_HPP