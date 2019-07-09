//
// Created by cheesema on 2019-07-08.
//

#ifndef LIBAPR_GPUAPRTEST_HPP
#define LIBAPR_GPUAPRTEST_HPP

class GPUAccess{

    class GPUAccessImpl;
    std::unique_ptr<GPUAccessImpl> data;

    void init_y_vec(std::vector<uint16_t>& y_vec_);

public:

    ~GPUAccess();
    GPUAccess();
    GPUAccess(GPUAccess&&);


};

bool run_simple_test();


#endif //LIBAPR_GPUAPRTEST_CUH_HPP
