//
// Created by cheesema on 2019-07-09.
//

#ifndef LIBAPR_GPUACCESS_HPP
#define LIBAPR_GPUACCESS_HPP

#include "data_structures/APR/GenInfo.hpp"

class GPUAccess {

    class GPUAccessImpl;
    std::unique_ptr<GPUAccessImpl> data;

public:

    void copy2Device();
    void copy2Host();

    void init_y_vec(std::vector<uint16_t>& y_vec_);
    void init_xz_end_vec(std::vector<uint64_t>& xz_end_vec);
    void init_level_xz_vec(std::vector<uint64_t>& level_xz_vec);

    GenInfo* genInfo;

    int level_max() const { return genInfo->l_max; }
    int level_min() const { return genInfo->l_min; }

    int x_num(const unsigned int level) const { return genInfo->x_num[level]; }
    int y_num(const unsigned int level) const { return genInfo->y_num[level]; }
    int z_num(const unsigned int level) const { return genInfo->z_num[level]; }

    int org_dims(int dim) const { return genInfo->org_dims[dim]; }

    ~GPUAccess();
    GPUAccess();
    GPUAccess(GPUAccess&&);

};

#endif //LIBAPR_GPUACCESS_HPP