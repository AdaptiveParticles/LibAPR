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
class ExtraPartCellData {
    
public:
    uint64_t depth_max;
    uint64_t depth_min;

    std::vector<uint64_t> z_num;
    std::vector<uint64_t> x_num;

    std::vector<std::vector<std::vector<T>>> data; // [level][x_num(level) * z + x][y]

    ExtraPartCellData() {}
    template<typename S>
    ExtraPartCellData(const ExtraPartCellData<S> &part_data) { initialize_structure_parts(part_data); }

    ExtraPartCellData(APR &apr) { initialize_structure_parts_empty(apr); }

    T& operator[](PCDKey& pcdKey){
      return data[pcdKey.level][pcdKey.offset][pcdKey.local_ind];
    }

    void initialize_structure_parts_empty(APR& apr);



private:

    template<typename S>
    void initialize_structure_parts(const ExtraPartCellData<S>& part_data) {
        // Initialize the structure to the same size as the given structure

        depth_max = part_data.depth_max;
        depth_min = part_data.depth_min;
        
        z_num.resize(depth_max+1);
        x_num.resize(depth_max+1);
        data.resize(depth_max+1);

        for (uint64_t i = depth_min; i <= depth_max; ++i) {
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
void ExtraPartCellData<T>::initialize_structure_parts_empty(APR& apr) {
    // Initialize the structure to the same size as the given structure
    depth_max = apr.level_max();
    depth_min = apr.level_min();

    z_num.resize(depth_max+1);
    x_num.resize(depth_max+1);
    data.resize(depth_max+1);

    for (uint64_t i = depth_min; i <= depth_max; ++i) {
        z_num[i] = apr.spatial_index_z_max(i);
        x_num[i] = apr.spatial_index_x_max(i);
        data[i].resize(z_num[i]*x_num[i]);

        for (int j = 0; j < data[i].size(); ++j) {
            data[i][j].resize(0);
        }
    }
}
#endif //PARTPLAY_PARTNEIGH_HPP
