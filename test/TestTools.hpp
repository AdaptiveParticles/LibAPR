//
// Created by gonciarz on 4/10/18.
//

#ifndef LIBAPR_TESTTOOLS_HPP
#define LIBAPR_TESTTOOLS_HPP


#include "data_structures/Mesh/PixelData.hpp"
#include <random>

/**
 * Compares mesh with provided data
 * @param mesh
 * @param data - data with [Z][Y][X] structure
 * @return true if same
 */
template<typename T>
inline bool compare(PixelData<T> &mesh, const float *data, const float epsilon) {
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
inline bool initFromZYXarray(PixelData<T> &mesh, const float *data) {
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

/**
 * Compares two meshes
 * @param expected
 * @param tested
 * @param maxNumOfErrPrinted - how many error values should be printed (-1 for all)
 * @return number of errors detected
 */
template <typename T>
inline int compareMeshes(const PixelData<T> &expected, const PixelData<T> &tested, double maxError = 0.0001, int maxNumOfErrPrinted = 3) {
    int cnt = 0;
    for (size_t i = 0; i < expected.mesh.size(); ++i) {
        if (std::abs(expected.mesh[i] - tested.mesh[i]) > maxError || std::isnan(expected.mesh[i]) ||
            std::isnan(tested.mesh[i])) {
            if (cnt < maxNumOfErrPrinted || maxNumOfErrPrinted == -1) {
                std::cout << "ERROR expected vs tested mesh: " << expected.mesh[i] << " vs " << tested.mesh[i] << " IDX:" << tested.getStrIndex(i) << std::endl;
            }
            cnt++;
        }
    }
    std::cout << "Number of errors / all points: " << cnt << " / " << expected.mesh.size() << std::endl;
    return cnt;
}

/**
 * Generates mesh with provided dims with random values in range [0, 1] * multiplier
 * @param y
 * @param x
 * @param z
 * @param multiplier
 * @return
 */
template <typename T>
inline PixelData<T> getRandInitializedMesh(int y, int x, int z, float multiplier = 2.0f, bool useIdxNumbers = false) {
    PixelData<T> m(y, x, z);
    std::cout << "Mesh info: " << m << std::endl;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    #ifdef HAVE_OPENMP
    #pragma omp parallel for default(shared)
    #endif
    for (size_t i = 0; i < m.mesh.size(); ++i) {
        m.mesh[i] = useIdxNumbers ? i : dist(mt) * multiplier;
    }
    return m;
}


#endif //LIBAPR_TESTTOOLS_HPP
