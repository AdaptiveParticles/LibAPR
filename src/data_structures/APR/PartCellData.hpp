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

#include "APR.hpp"

#include "APRAccessStructures.hpp"

#include "GenData.hpp"

template<typename DataType>
class PartCellData: public GenData<DataType> {

    uint64_t number_elements=0;

public:
    uint64_t level_max;
    uint64_t level_min;

    std::vector<uint64_t> z_num;
    std::vector<uint64_t> x_num;

    std::vector<std::vector<std::vector<DataType>>> data; // [level][x_num(level) * z + x][y]

    PartCellData() {}
    template<typename S>
    PartCellData(const PartCellData<S> &part_data) { initialize_structure_parts(part_data); }

    PartCellData(APR &apr) { initialize_structure_parts_empty(apr); }

    DataType& operator[](LinearIterator& it) override {
        return data[it.level][it.offset][it.current_index - it.begin_index];
    }

    DataType& operator[](PCDKey& pcdKey){
      return data[pcdKey.level][pcdKey.offset][pcdKey.local_ind];
    }

    uint64_t size() const override {
        return number_elements;
    }

    void initialize_structure_parts_empty(APR& apr);

    void init(APR& apr) override {

        auto it = apr.iterator();
        initialize_structure_parts(it,it.level_max());
    };

    void init(APR& apr,unsigned int level) override {

        auto it = apr.iterator();
        initialize_structure_parts(it,level);
    };

    void init_tree(APR& apr,unsigned int level) override {

        auto it = apr.tree_iterator();
        initialize_structure_parts(it,level);
    };

    void init_tree(APR& apr) override {

        auto it = apr.tree_iterator();
        initialize_structure_parts(it,it.level_max());
    };


    void set_to_zero() override {

        for (uint64_t i = level_min; i <= level_max; ++i) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (uint64_t j = 0; j < data[i].size(); ++j) {
                std::fill(data[i][j].begin(),data[i][j].end(),0);
            }
        }
    }

private:

    void initialize_structure_parts(LinearIterator& it,int level_init);

    template<typename S>
    void initialize_structure_parts(const PartCellData<S>& part_data) {
        // Initialize the structure to the same size as the given structure

        number_elements = 0;

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
                number_elements += part_data.data[i][j].size();
            }
        }
    }
};

template<typename DataType>
void PartCellData<DataType>::initialize_structure_parts_empty(APR& apr) {
    // Initialize the structure to the same size as the given structure
    level_max = apr.level_max();
    level_min = apr.level_min();

    z_num.resize(level_max+1);
    x_num.resize(level_max+1);
    data.resize(level_max+1);

    for (uint64_t i = level_min; i <= level_max; ++i) {
        z_num[i] = apr.z_num(i);
        x_num[i] = apr.x_num(i);
        data[i].resize(z_num[i]*x_num[i]);

        for (int j = 0; j < data[i].size(); ++j) {
            data[i][j].resize(0);
        }
    }
}


template<typename DataType>
void PartCellData<DataType>::initialize_structure_parts(LinearIterator& it,int level_init) {

    if(level_init == 0){
        level_init = it.level_max();
    }

    // Initialize the structure to the same size as the given structure
    level_max = it.level_max();
    level_min = it.level_min();

    z_num.resize(level_max+1);
    x_num.resize(level_max+1);
    data.resize(level_max+1);

    for (unsigned int l = level_min; l <= level_init; ++l) {
        z_num[l] = it.z_num(l);
        x_num[l] = it.x_num(l);
        data[l].resize(z_num[l]*x_num[l]);

        for (int z = 0; z < it.z_num(l); z++) {
            for (int x = 0; x < it.x_num(l); ++x) {

                it.begin(l,z,x);

                auto sz = it.end() - it;

                if(it.end() > 0) {
                    auto off = x_num[l]*z + x;
                    data[l][off].resize(sz,0);
                }

                number_elements += sz;

            }
        }

    }
}

#endif //PARTPLAY_PARTNEIGH_HPP
