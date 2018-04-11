//
// Created by gonciarz on 4/10/18.
//

#ifndef LIBAPR_TESTTOOLS_HPP
#define LIBAPR_TESTTOOLS_HPP


#include "data_structures/Mesh/MeshData.hpp"


/**
 * Compares mesh with provided data
 * @param mesh
 * @param data - data with [Z][Y][X] structure
 * @return true if same
 */
template<typename T>
inline bool compare(MeshData<T> &mesh, const float *data, const float epsilon) {
    size_t dataIdx = 0;
    for (size_t z = 0; z < mesh.z_num; ++z) {
        for (size_t y = 0; y < mesh.y_num; ++y) {
            for (size_t x = 0; x < mesh.x_num; ++x) {
                bool v = std::abs(mesh(y, x, z) - data[dataIdx]) < epsilon;
                if (v == false) {
                    std::cerr << "Mesh and expected data differ. First place at (Y, X, Z) = " << y << ", " << x
                              << ", " << z << ") " << mesh(y, x, z) << " vs " << data[dataIdx] << std::endl;
                    return false;
                }
                ++dataIdx;
            }
        }
    }
    return true;
}


template<typename T>
inline bool initFromZYXarray(MeshData<T> &mesh, const float *data) {
    size_t dataIdx = 0;
    for (size_t z = 0; z < mesh.z_num; ++z) {
        for (size_t y = 0; y < mesh.y_num; ++y) {
            for (size_t x = 0; x < mesh.x_num; ++x) {
                mesh(y, x, z) = data[dataIdx];
                ++dataIdx;
            }
        }
    }
    return true;
}

#endif //LIBAPR_TESTTOOLS_HPP
