//
// Created by Krzysztof Gonciarz on 6/25/18.
//

#include <gtest/gtest.h>
#include "data_structures/Mesh/PixelData.hpp"
//TODO: only APRAccess.hpp should be included here but currently because of dependencies it does not work :(
#include "data_structures/APR/APR.hpp"
//#include "data_structures/APR/APRAccess.hpp"
#include "algorithm/PullingScheme.hpp"
#include "TestTools.hpp"
#ifdef APR_USE_CUDA
#include "algorithm/ComputeGradientCuda.hpp"
#endif

namespace {
    template <typename T>
    PixelData<float> generateLevels(const PixelData<T> &dimsMesh, int maxLevel) {
        PixelData<float> levels(dimsMesh, false);
        for (int i = 0; i < levels.mesh.size(); ++i) {
            levels.mesh[i] = ( i/2 ) % (maxLevel + 2);
        }
//        std::cout << "LEVELS: " << std::endl;
        levels.printMesh(3, 0);
        return levels;
    }

    void printParticleCellTree(const std::vector<PixelData<uint8_t>> &particleCellTree) {
        for (int l = 0; l < particleCellTree.size(); ++l) {
            auto &tree = particleCellTree[l];
            std::cout << "-- level = " << l << ",  " << tree << std::endl;
            tree.printMeshT(3,0);
        }
    }

    TEST(PullingSchemeTest, Init) {

        APRAccess access;
        access.l_max = 4;
        access.l_min = 1;
        access.org_dims[0] = 16;
        access.org_dims[1] = 1;
        access.org_dims[2] = 1;

        PullingScheme ps;
        ps.initialize_particle_cell_tree(access);
        std::vector<PixelData<uint8_t>> &pctree = ps.getParticleCellTree();
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>> Initialized tree:\n";
        printParticleCellTree(pctree);
        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";

        // TEST: check if zeroed and correct number of levels
        ASSERT_EQ(access.l_max, pctree.size()); // all levels [0, access.level_max - 1]
        for (int l = 0; l < pctree.size(); ++l) {
            auto &tree = pctree[l];
            for (auto &e : tree.mesh) {
                ASSERT_EQ(0, e);
            }
        }

        // Generate mesh with test levels
        PixelData<float> levels(pctree.back(), false);// = generateLevels(pctree[access.l_max - 1], access.l_max);
        float values[] = {9,0,0,0, 0,0,0,0};
        initFromZYXarray(levels, values);

        // Fill particle cell tree with levels
        int l_max = access.l_max - 1;
        int l_min = access.l_min;
        ps.fill(l_max, levels);

        levels.printMeshT(3,0);

        PixelData<float> levelsDS;
        for(int l_ = l_max - 1; l_ >= l_min; l_--){
            //down sample the resolution level k, using a max reduction
            downsample(levels, levelsDS,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
            ps.fill(l_,levelsDS);
            levelsDS.printMeshT(3,0);
            levels.swap(levelsDS);
        }

        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      Filled tree:\n";
        printParticleCellTree(pctree);
        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";

//        ps.fill_neighbours(l_max);
//        pctree[l_max].printMesh(3, 0);


        ps.pulling_scheme_main();
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      MAIN   tree:\n";
        printParticleCellTree(pctree);
        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";

        access.initialize_structure_from_particle_cell_tree(false, ps.getParticleCellTree());
        std::cout << "NUM OF PARTICLES: " << access.get_total_number_particles() << std::endl;


        APRIterator apr_iterator(access);
        std::cout << "Total number of particles: " << apr_iterator.total_number_particles() << std::endl;

        int prev = 0;
        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            std::cout << "Level: " << level << std::endl;
            int w = (int) (std::pow(2, 5-level) * 3);
            for (int z = 0; z < apr_iterator.spatial_index_z_max(level); ++z) {
                for (int x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {
                        for (int i = prev; i < apr_iterator.y(); ++i )  std::cout << std::setw(w) << ".";
                        std::cout << std::setw(w) << apr_iterator.y();
                        prev = apr_iterator.y() + 1;
                    }
                    for (int pp = prev; pp < apr_iterator.spatial_index_y_max(level); ++pp)
                        std::cout << std::setw(w) << ".";

                    prev = 0;
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }

    }

#ifdef APR_USE_CUDA
    TEST(PullingSchemeTest, computeLevels) {
        using ImgType = float;
        const int maxLevel = 3;
        const float relError = 0.1;

        PixelData<ImgType> grad = getRandInitializedMesh<ImgType>(10, 20, 33);
        PixelData<float> localIntensityScaleCpu = getRandInitializedMesh<float>(10, 20, 33);

        PixelData<float> localIntensityScaleGpu(localIntensityScaleCpu, true);
        PixelData<float> elo(localIntensityScaleCpu, true);
        APRTimer timer(true);

        timer.start_timer("CPU PS FULL");
        APRConverter<ImgType>().computeLevels(grad, localIntensityScaleCpu, maxLevel, relError);
        timer.stop_timer();

        timer.start_timer("GPU PS FULL");
        computeLevelsCuda(grad, localIntensityScaleGpu, maxLevel, relError);
        timer.stop_timer();

        EXPECT_EQ(compareMeshes(localIntensityScaleCpu, localIntensityScaleGpu), 0);
    }
#endif
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
