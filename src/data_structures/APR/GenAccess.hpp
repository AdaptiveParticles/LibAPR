//
// Created by cheesema on 2019-06-03.
//

#ifndef LIBAPR_GENACCESS_HPP
#define LIBAPR_GENACCESS_HPP


class GenAccess {

public:
    uint64_t l_min;
    uint64_t l_max;
    uint64_t org_dims[3]={0,0,0};

    uint8_t number_dimensions = 3;

    // TODO: SHould they be also saved as uint64 in HDF5? (currently int is used)
    std::vector<uint64_t> x_num;
    std::vector<uint64_t> y_num;
    std::vector<uint64_t> z_num;

    uint64_t total_number_particles;

    std::vector<unsigned int> level_size; // precomputation of the size of each level, used by the iterators.

    uint64_t level_max() const { return l_max; }
    uint64_t level_min() const { return l_min; }
    uint64_t spatial_index_x_max(const unsigned int level) const { return x_num[level]; }
    uint64_t spatial_index_y_max(const unsigned int level) const { return y_num[level]; }
    uint64_t spatial_index_z_max(const unsigned int level) const { return z_num[level]; }

};


#endif //LIBAPR_GENACCESS_HPP
