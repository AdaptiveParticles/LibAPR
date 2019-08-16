//
// Created by cheesema on 2019-07-09.
//

#ifndef LIBAPR_PARTICLEDATAGPU_HPP
#define LIBAPR_PARTICLEDATAGPU_HPP

#include <cstdint>
#include <memory>
#include "data_structures/Mesh/PixelData.hpp"

template<typename DataType>
class ParticleDataGpu {

protected:

    template<typename T>
    class ParticleDataGpuImpl;

    std::unique_ptr<ParticleDataGpuImpl<DataType>> data;

public:

    ParticleDataGpu();
    ~ParticleDataGpu();

    void init(VectorData<DataType>& cpu_data);

    DataType* getGpuData();

    void sendDataToGpu();

    void getDataFromGpu();

};



#endif //LIBAPR_PARTICLEDATAGPU_HPP
