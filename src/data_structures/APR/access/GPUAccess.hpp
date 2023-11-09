//
// Created by cheesema on 2019-07-09.
//

#ifndef LIBAPR_GPUACCESS_HPP
#define LIBAPR_GPUACCESS_HPP

#include "data_structures/APR/access/LinearAccess.hpp"
#include "data_structures/Mesh/PixelData.hpp"
//#include "data_structures/APR/GenInfo.hpp"

class GPUAccess {

    friend class GPUAccessHelper;

protected:

    class GPUAccessImpl;
    std::unique_ptr<GPUAccessImpl> data;
    bool initialized = false;

public:

    void copy2Device();
    void copy2Device(size_t numElements, GPUAccess* tree_access);

    void copy2Host();

    void init_y_vec(VectorData<uint16_t>& y_vec_);
    void init_xz_end_vec(VectorData<uint64_t>& xz_end_vec);
    void init_level_xz_vec(VectorData<uint64_t>& level_xz_vec);

    GenInfo* genInfo;
    uint64_t total_number_particles() const { return genInfo->total_number_particles; }

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

class GPUAccessHelper {

public:


    GPUAccess* gpuAccess;
    LinearAccess* linearAccess;

    GPUAccessHelper(GPUAccess& gpuAccess_,LinearAccess& linearAccess_);

    uint16_t* get_y_vec_ptr();
    uint64_t* get_xz_end_vec_ptr();
    uint64_t* get_level_xz_vec_ptr();

    void init_gpu(bool force=false){
        if(force || !gpuAccess->initialized) {
            gpuAccess->init_y_vec(linearAccess->y_vec);
            gpuAccess->init_level_xz_vec(linearAccess->level_xz_vec);
            gpuAccess->init_xz_end_vec(linearAccess->xz_end_vec);
            gpuAccess->genInfo = linearAccess->genInfo;
            gpuAccess->copy2Device();
            gpuAccess->initialized = true;
        }
    }

    void init_gpu(GPUAccessHelper& tree_access, bool force=false){
        if(force || !gpuAccess->initialized) {
            gpuAccess->init_y_vec(linearAccess->y_vec);
            gpuAccess->init_level_xz_vec(linearAccess->level_xz_vec);
            gpuAccess->init_xz_end_vec(linearAccess->xz_end_vec);
            gpuAccess->genInfo = linearAccess->genInfo;
            gpuAccess->copy2Device(total_number_particles(tree_access.level_max()), tree_access.gpuAccess);
            gpuAccess->initialized = true;
        }
    }

    void copy2Host() {
        gpuAccess->copy2Host();
    }

    uint64_t total_number_particles() const { return gpuAccess->total_number_particles(); }

    uint64_t total_number_particles(const int level) const {
        uint64_t index = linearAccess->level_xz_vec[level] + linearAccess->x_num(level) - 1 + (linearAccess->z_num(level)-1)*linearAccess->x_num(level);
        return linearAccess->xz_end_vec[index];
    }

    int level_max() const { return gpuAccess->level_max(); }
    int level_min() const { return gpuAccess->level_min(); }

    int x_num(const unsigned int level) const { return gpuAccess->x_num(level); }
    int y_num(const unsigned int level) const { return gpuAccess->y_num(level); }
    int z_num(const unsigned int level) const { return gpuAccess->z_num(level); }

    int org_dims(int dim) const { return gpuAccess->org_dims(dim); }

};


#endif //LIBAPR_GPUACCESS_HPP
