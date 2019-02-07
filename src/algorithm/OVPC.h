//
// Created by gonciarz on 10/17/18.
//
// CPU implementation of GPU pulling scheme algorithm

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

    static constexpr ElementType  SEED = OVPC_SEED << BIT_SHIFT;
    static constexpr ElementType  BOUNDARY = OVPC_BOUNDARY << BIT_SHIFT;
    static constexpr ElementType  FILLER = OVPC_FILLER << BIT_SHIFT;
    static constexpr ElementType  MASK = 0x03 << BIT_SHIFT;

    int iLevelMax;
    int iLevelMin;
    std::vector<PixelData<ElementType>> iParticleCellTree;

public:
    template <typename T>
    OVPC(const APRAccess &aAprAccess, const PixelData<T> &aInputLevels) {
        // Level Max is one less since we are working on downsampled version
        iLevelMax = aAprAccess.l_max - 1;
        iLevelMin = aAprAccess.l_min;

        // Initialize particle cell tree on maximum level with input level data
        iParticleCellTree.resize(iLevelMax + 1);
        iParticleCellTree[iLevelMax].init(aInputLevels.y_num, aInputLevels.x_num, aInputLevels.z_num);
        fillLevel(iLevelMax, aInputLevels);

        // Downsample with max reduction to levelMin to fill the rest of the tree
        for(int level = iLevelMax - 1; level >= iLevelMin; --level) {
            downsample(iParticleCellTree[level + 1], iParticleCellTree[level],
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
        }
    }

    std::vector<PixelData<ElementType>>& getParticleCellTree() { return iParticleCellTree; }

    void generateTree() {

        for (int level = iLevelMin; level <= iLevelMax; ++level) {
            firstStep(level);
        }
        for (int level = iLevelMax - 1; level >= iLevelMin; --level) {
            secondStep(level);
        }
    }

private:

    /**
     * Sets SEED, BOUNDARY or FILER type to each element of the tree.
     * @param aCurrentLevel
     */
    void firstStep(ElementType aCurrentLevel) {
        auto &currData = iParticleCellTree[aCurrentLevel];
        const size_t xLen = currData.x_num;
        const size_t yLen = currData.y_num;
        const size_t zLen = currData.z_num;

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
                    ElementType level = currData.mesh[index + k];
                    if (level <= aCurrentLevel) {
                        bool hasNeighHigherLevel = false;
                        bool hasNeighSameLevel = false;

                        CHECKBOUNDARIES(2, k, yLen - 1, boundaries);
                        int64_t jn, in, kn;
                        NEIGHBOURLOOP(jn, in, kn, boundaries) {
                            size_t neighbourIndex = index + k + jn * xLen * yLen + in * yLen + kn ;
                            ElementType neighbourLevel = ~MASK & currData.mesh[neighbourIndex];
                            if (neighbourLevel > aCurrentLevel) { hasNeighHigherLevel = true; break; }
                            else if (neighbourLevel == aCurrentLevel) hasNeighSameLevel = true;
                        }

                        if (!hasNeighHigherLevel) {
                            if (level == aCurrentLevel) currData.mesh[index + k] |= SEED;
                            else if (hasNeighSameLevel) currData.mesh[index + k] |= BOUNDARY;
                            else currData.mesh[index + k] |= FILLER;
                        }
                    }
                }
            }
        }
    }

    /*
     * Zeros all level values in children when parent has a level set.
     * Shifts all values to keep only type value (input level are not needed anymore).
     */
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
            children_boundaries[0] = (j == z_num - 1 && prev_z_num % 2 == 1) ? 1 : 2;

            for (int64_t i = 0; i < x_num; ++i) {
                children_boundaries[1] = (i == x_num - 1 && prev_x_num % 2 == 1) ? 1 : 2;

                size_t index = j * x_num * y_num + i * y_num;

                for (int64_t k = 0; k < y_num; ++k) {
                    children_boundaries[2] = (k == y_num - 1 && prev_y_num % 2 == 1) ? 1 : 2;

                    const ElementType status = iParticleCellTree[level].mesh[index + k];
                    int64_t jn, in, kn;
                    CHILDRENLOOP(jn, in, kn, children_boundaries) {
                        size_t children_index = jn * prev_x_num * prev_y_num + in * prev_y_num + kn;
                        // If there is any of SEED, FILLER or BOUNDARY type set children to 0
                        // otherwise just shift to keep type value only
                        iParticleCellTree[level + 1].mesh[children_index] = (status >= SEED) ?
                                0 : (iParticleCellTree[level + 1].mesh[children_index] >> BIT_SHIFT);
                    }

                    // shift values in min level since they are not handled by childrenloop
                    if (level == iLevelMin) iParticleCellTree[level].mesh[index + k] = status >> BIT_SHIFT;
                }
            }
        }
    }

    /*
     * Fills level with provided data, clamps values to allowed one [minLevel; maxLevel]
     */
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
