//
// Created by cheesema on 2019-06-03.
//

#ifndef LIBAPR_GENACCESS_HPP
#define LIBAPR_GENACCESS_HPP

#include "GenInfo.hpp"

class GenAccess {

public:

    GenInfo* genInfo;

    uint64_t level_max() const { return genInfo->l_max; }
    uint64_t level_min() const { return genInfo->l_min; }

    uint64_t spatial_index_x_max(const unsigned int level) const { return genInfo->x_num[level]; }
    uint64_t spatial_index_y_max(const unsigned int level) const { return genInfo->y_num[level]; }
    uint64_t spatial_index_z_max(const unsigned int level) const { return genInfo->z_num[level]; }

    uint64_t x_num(const unsigned int level) const { return genInfo->x_num[level]; }
    uint64_t y_num(const unsigned int level) const { return genInfo->y_num[level]; }
    uint64_t z_num(const unsigned int level) const { return genInfo->z_num[level]; }

    unsigned int org_dims(int dim) const { return genInfo->org_dims[dim]; }

};


#endif //LIBAPR_GENACCESS_HPP
