///////////////////
//
//  Bevan Cheeseman 2016
//
//     Class for storing extra cell or particle data, that can then be accessed using the access data from pc_structure or parent cells
//
///////////////

#ifndef PARTPLAY_EXTRAPARTCELLDATA_HPP
#define PARTPLAY_EXTRAPARTCELLDATA_HPP

#include <vector>

#include "data_structures/APR/APR.hpp"

#include "data_structures/APR/APRAccessStructures.hpp"

template<typename T>
class PartCellData {
    
public:
    uint64_t level_max;
    uint64_t level_min;

    std::vector<uint64_t> z_num;
    std::vector<uint64_t> x_num;

    std::vector<std::vector<std::vector<T>>> data; // [level][x_num(level) * z + x][y]

    PartCellData() {}
    template<typename S>
    PartCellData(const PartCellData<S> &part_data) { initialize_structure_parts(part_data); }

    PartCellData(APR &apr) { initialize_structure_parts_empty(apr); }

    T& operator[](LinearIterator it) { return data[it.level][it.offset][it.current_index - it.begin_index]; }

    T& operator[](PCDKey& pcdKey){
      return data[pcdKey.level][pcdKey.offset][pcdKey.local_ind];
    }

    void initialize_structure_parts_empty(APR& apr);

    void initialize_structure_parts(APR& apr);



private:

    template<typename S>
    void initialize_structure_parts(const PartCellData<S>& part_data) {
        // Initialize the structure to the same size as the given structure

        level_max = part_data.level_max;
        level_min = part_data.level_min;
        
        z_num.resize(level_max+1);
        x_num.resize(level_max+1);
        data.resize(level_max+1);

        for (uint64_t i = level_min; i <= level_max; ++i) {
            z_num[i] = part_data.z_num[i];
            x_num[i] = part_data.x_num[i];
            data[i].resize(z_num[i]*x_num[i]);
            for (uint64_t j = 0; j < part_data.data[i].size(); ++j) {
                data[i][j].resize(part_data.data[i][j].size(),0);
            }
        }
    }
};

template<typename T>
void PartCellData<T>::initialize_structure_parts_empty(APR& apr) {
    // Initialize the structure to the same size as the given structure
    level_max = apr.level_max();
    level_min = apr.level_min();

    z_num.resize(level_max+1);
    x_num.resize(level_max+1);
    data.resize(level_max+1);

    for (uint64_t i = level_min; i <= level_max; ++i) {
        z_num[i] = apr.spatial_index_z_max(i);
        x_num[i] = apr.spatial_index_x_max(i);
        data[i].resize(z_num[i]*x_num[i]);

        for (int j = 0; j < data[i].size(); ++j) {
            data[i][j].resize(0);
        }
    }
}


template<typename T>
void PartCellData<T>::initialize_structure_parts(APR& apr) {

    auto apr_it = apr.iterator();

    // Initialize the structure to the same size as the given structure
    level_max = apr_it.level_max();
    level_min = apr_it.level_min();

    z_num.resize(level_max+1);
    x_num.resize(level_max+1);
    data.resize(level_max+1);

    for (unsigned int l = level_min; l <= level_max; ++l) {
        z_num[l] = apr_it.z_num(l);
        x_num[l] = apr_it.x_num(l);
        data[l].resize(z_num[l]*x_num[l]);

        for (int z = 0; z < apr_it.z_num(l); z++) {
            for (int x = 0; x < apr_it.x_num(l); ++x) {

                apr_it.begin(l,z,x);

                auto sz = apr_it.end() - apr_it;

                if(apr_it.end() > 0) {
                    auto off = x_num[l]*z + x;
                    data[l][off].resize(sz,0);
                }

            }
        }

    }
}

#endif //PARTPLAY_PARTNEIGH_HPP
