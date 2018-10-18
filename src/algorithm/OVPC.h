//
// Created by gonciarz on 10/17/18.
//

#ifndef LIBAPR_OVPC_H
#define LIBAPR_OVPC_H


#include <vector>
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/APRAccess.hpp"
#include "algorithm/PullingScheme.hpp"

class OVPC {
    // Element big enouth to keep all the levels + 2 highest bits for type
    // for uint8_t we have [ 2 bit - type(empty, seed, boundary, filler) |  6 bit - level(0-63) ]
    using ElementType = uint8_t;
    static constexpr int BIT_SHIFT = 6;
    static constexpr ElementType OVPC_SEED = SEED_TYPE;
    static constexpr ElementType OVPC_BOUNDARY = BOUNDARY_TYPE;
    static constexpr ElementType OVPC_FILLER = FILLER_TYPE;

    static constexpr ElementType  SEED_MASK = SEED_TYPE << BIT_SHIFT;
    static constexpr ElementType  BOUNDARY_MASK = BOUNDARY_TYPE << BIT_SHIFT;
    static constexpr ElementType  FILLER_MASK = FILLER_TYPE << BIT_SHIFT;
    static constexpr ElementType  MASK = 0x03 << BIT_SHIFT;

    int iLevelMax;
    int iLevelMin;
    std::vector<PixelData<ElementType>> iParticleCellTree;

public:
    std::vector<PixelData<ElementType>>& getParticleCellTree() { return iParticleCellTree; }

    template <typename T>
    OVPC(const APRAccess &apr_access, const PixelData<T> &inLevels) {
        // Level Max is one less since we are working on downsampled version
        iLevelMax = apr_access.l_max - 1;
        iLevelMin = apr_access.l_min;


        // Initialize particle cell tree on maximum level with input level data
        iParticleCellTree.resize(iLevelMax + 1);
        iParticleCellTree[iLevelMax].init(inLevels.y_num, inLevels.x_num, inLevels.z_num);
        fillLevel(iLevelMax, inLevels);

        // Downsample with max reduction to levelMin
        for(int level = iLevelMax - 1; level >= iLevelMin; level--) {
            downsample(iParticleCellTree[level + 1], iParticleCellTree[level],
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
        }
    }

    void generateTree() {
        for (int l = iLevelMin; l <= iLevelMax; ++l) {
            firstStep2(l);
        }
        for (int l = iLevelMax - 1; l >= iLevelMin; --l) {
            secondStep(l);
        }
    }

private:
    void firstStep(ElementType level) {
        auto &currLevel = iParticleCellTree[level];
        const size_t xLen = currLevel.x_num;
        const size_t yLen = currLevel.y_num;
        const size_t zLen = currLevel.z_num;

        PixelData<ElementType> &currData = iParticleCellTree[level];

        short boundaries[3][2] = {{0,2},{0,2},{0,2}};
        #ifdef HAVE_OPENMP
        #pragma omp parallel for default(shared) firstprivate(boundaries) if (zLen * xLen * yLen > 100000)
        #endif
        for (size_t j = 0; j < zLen; ++j) {
            CHECKBOUNDARIES(0, j, zLen - 1, boundaries);
            for (size_t i = 0; i < xLen; ++i) {
                CHECKBOUNDARIES(1, i, xLen - 1, boundaries);
                size_t index = j * xLen * yLen + i * yLen;
                for (size_t k = 0; k < yLen; ++k) {
                    uint8_t status = iParticleCellTree[level].mesh[index + k];
                    if (status <= level) {
                        bool ok = true;
                        bool neig = false;
                        CHECKBOUNDARIES(2, k, yLen - 1, boundaries);
                        int64_t jn, in, kn;
                        NEIGHBOURLOOP(jn, in, kn, boundaries) {
                                    size_t neighbour_index = index + jn * xLen * yLen + in * yLen + kn + k;
                                    ElementType currentLevel = ~MASK & currLevel.mesh[neighbour_index];
                                    if (currentLevel > level) { ok = false; break; }
                                    else if (currentLevel == level) neig = true;
                                }

                        if (ok) {
                            if (status == level) currData.mesh[index + k] |= SEED_MASK;
                            else if (neig) currData.mesh[index + k] |= BOUNDARY_MASK;
                            else currData.mesh[index + k] |= FILLER_MASK;
                        }
                    }
                }
            }
        }
    }

    void firstStep2(ElementType level) {
        auto &currLevel = iParticleCellTree[level];
        const size_t xLen = currLevel.x_num;
        const size_t yLen = currLevel.y_num;
        const size_t zLen = currLevel.z_num;

        PixelData<ElementType> &currData = iParticleCellTree[level];

        short boundaries[3][2] = {{0,2},{0,2},{0,2}};
        #ifdef HAVE_OPENMP
        #pragma omp parallel for default(shared) firstprivate(boundaries) if (zLen * xLen * yLen > 100000)
        #endif
        for (size_t j = 0; j < zLen; ++j) {
            CHECKBOUNDARIES(0, j, zLen - 1, boundaries);
            for (size_t i = 0; i < xLen; ++i) {
                CHECKBOUNDARIES(1, i, xLen - 1, boundaries);
                size_t index = j * xLen * yLen + i * yLen;

                bool p = true, c, n, pn = false, cn ,nn;
                {
                    c = true;
                    cn = false;
                    int64_t jn, in;
                    NEIGHBOURLOOP2(jn, in, boundaries) {
                                size_t neighbour_index = index + jn * xLen * yLen + in * yLen + 0;
                                ElementType currentLevel = static_cast<ElementType>(~MASK) & currLevel.mesh[neighbour_index];
                                if (currentLevel > level) { c = false; break; }
                                else if (currentLevel == level) cn = true;
                            }
                }
                for (size_t k = 0; k < yLen; ++k) {

                    if (k < yLen - 1) {
                        bool ok = true;
                        bool neig = false;
                        int64_t jn, in;
                        NEIGHBOURLOOP2(jn, in, boundaries) {
                                size_t neighbour_index = index + jn * xLen * yLen + in * yLen + k + 1;
                                ElementType currentLevel = static_cast<ElementType>(~MASK) & currLevel.mesh[neighbour_index];
                                if (currentLevel > level) { ok = false; break; }
                                else if (currentLevel == level) neig = true;
                            }
                        n = ok;
                        nn = neig;
                    }

                    if (p && c && n) {
                        uint8_t status = iParticleCellTree[level].mesh[index + k];

                            if (status == level) currData.mesh[index + k] |= SEED_MASK;
                            else if (pn || cn || nn) currData.mesh[index + k] |= BOUNDARY_MASK;
                            else currData.mesh[index + k] |= FILLER_MASK;
                    }

                    p = c; pn = cn;
                    c = n; cn = nn;
                }
            }
        }
    }

    void secondStep(int level) {
        short children_boundaries[3] = {2, 2, 2};
        const int64_t x_num = iParticleCellTree[level].x_num;
        const int64_t y_num = iParticleCellTree[level].y_num;
        const int64_t z_num = iParticleCellTree[level].z_num;

        int64_t prev_x_num = iParticleCellTree[level + 1].x_num;
        int64_t prev_y_num = iParticleCellTree[level + 1].y_num;
        int64_t prev_z_num = iParticleCellTree[level + 1].z_num;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) if (z_num * x_num * y_num > 10000) firstprivate(level, children_boundaries)
#endif
        for (int64_t j = 0; j < z_num; ++j) {
            if (j == z_num - 1 && prev_z_num % 2) {
                children_boundaries[0] = 1;
            }

            for (int64_t i = 0; i < x_num; ++i) {
                if (i == x_num - 1 && prev_x_num % 2) {
                    children_boundaries[1] = 1;
                } else if (i == 0) {
                    children_boundaries[1] = 2;
                }

                size_t index = j * x_num * y_num + i * y_num;

                for (int64_t k = 0; k < y_num; ++k) {
                    if (k == y_num - 1 && prev_y_num % 2) {
                        children_boundaries[2] = 1;
                    } else if (k == 0) {
                        children_boundaries[2] = 2;
                    }

                    uint8_t status = iParticleCellTree[level].mesh[index + k];
                    int64_t jn, in, kn;
                    CHILDRENLOOP(jn, in, kn, children_boundaries) {
                        size_t children_index = jn * prev_x_num * prev_y_num + in * prev_y_num + kn;
                        ElementType  v = iParticleCellTree[level + 1].mesh[children_index];
                        iParticleCellTree[level + 1].mesh[children_index] = status >= (OVPC_SEED << BIT_SHIFT) ? 0 : v >> BIT_SHIFT;
                    }
                    if (level == iLevelMin) iParticleCellTree[level].mesh[index + k] = status >> BIT_SHIFT;
                }
            }
        }
    }

    template<typename T>
    void fillLevel(int level, const PixelData<T> &input) {
        auto &inMesh = input.mesh;
        auto &outMesh = iParticleCellTree[level].mesh;

        #ifdef HAVE_OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < inMesh.size(); ++i) {
            T v = inMesh[i];
            if (v > iLevelMax) v = iLevelMax;
            else if (v < iLevelMin) v = iLevelMin;
            outMesh[i] = v;
        }
    }
};

#endif //LIBAPR_OVPC_H
