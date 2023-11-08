//
// Created by Krzysztof Gonciarz on 6/25/18.
//

#include <gtest/gtest.h>
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/access/APRAccessStructures.hpp"
#include "algorithm/PullingScheme.hpp"
#include "algorithm/OVPC.h"
#include "TestTools.hpp"
#include "algorithm/APRConverter.hpp"


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

    template <typename T>
    void printParticleCellTree(const std::vector<PixelData<T>> &particleCellTree) {
        for (uint64_t  l = 0; l < particleCellTree.size(); ++l) {
            auto &tree = particleCellTree[l];
//            std::cout << "-- level = " << l << ",  " << tree << std::endl;
            tree.printMeshT(3,0);
        }
    }

    template <typename T>
    inline int compareParticleCellTrees(const PixelData<T> &expected, const PixelData<T> &tested, double maxError = 0.0001, int maxNumOfErrPrinted = 3) {
        int cnt = 0;
        int numOfParticles = 0;
        for (size_t i = 0; i < expected.mesh.size(); ++i) {
            if (expected.mesh[i] < 8) {
            if (std::abs(expected.mesh[i] - tested.mesh[i]) > maxError || std::isnan(expected.mesh[i]) ||
                std::isnan(tested.mesh[i])) {
                if (cnt < maxNumOfErrPrinted || maxNumOfErrPrinted == -1) {
                    std::cout << "ERROR expected vs tested mesh: " << (float)expected.mesh[i] << " vs " << (float)tested.mesh[i] << " IDX:" << tested.getStrIndex(i) << std::endl;
                }
                cnt++;
            }
                if (expected.mesh[i] > 0) numOfParticles++;
            }
        }
        std::cout << "Number of errors / all points: " << cnt << " / " << expected.mesh.size() << " Particles:" << numOfParticles << std::endl;
        return cnt;
    }

    // ------------------------------------------------------------------------

    TEST(PullingSchemeTest, DeleteMeAfterDeevelopment) {
        // TODO: delete me after development
        // Full 'get apr' pipeline to test imp. on different stages
        // Useful during debugging and can be removed once finished

        // Prepare input data (image)
      int values[] = {9,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

//        int values[] = {3,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 3,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, };
        // PS values for above 'image': int values[] = {4,0,0,0, 0,0,0,0, 4,0,0,0, 0,0,0,0};

        int len = sizeof(values)/sizeof(int);
        PixelData<int> data(len, 1, 1);
        initFromZYXarray(data, values);
        std::cout << "----- Input image:\n";
        data.printMeshT(3, 1);

        // Produce APR
        APR apr;
        APRConverter<uint16_t> aprConverter;
        aprConverter.par.rel_error = 0.01;
        aprConverter.par.lambda = 0.1;
        aprConverter.get_apr(apr, data);

        // Print information about APR and all particles
        std::cout << "APR level min/max: " << apr.level_max() << "/" << apr.level_min() << std::endl;
        for (int l = apr.level_min(); l <= apr.level_max(); ++l) {
            std::cout << "    level[" << l << "] size: " << apr.level_size(l) << std::endl;
        }
        std::cout << "APR particles z x y level:\n";
        auto it = apr.iterator();
        for (int level = it.level_min(); level <= it.level_max(); ++level) {
            for (int z = 0; z < it.z_num(level); z++) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end(); it++) {
                        std::cout << "              " << z << " " << x << " " << it.y() << " " << level << std::endl;
                    }
                }
            }
        }
        std::cout << std::endl;

        // Sample input
        ParticleData<uint16_t> particleIntensities;
        particleIntensities.sample_image(apr, data);

        // Reconstruct image from particles
        PixelData<uint16_t> reconstructImg;
        APRReconstruction::reconstruct_constant(apr, reconstructImg, particleIntensities);
        std::cout << "----- Reconstructed image:"<<std::endl;
        reconstructImg.printMeshT(3, 1);

        // Show level assigned to each pixel in reconstructed image
        PixelData<uint16_t> levelImg;
        APRReconstruction::reconstruct_level(apr, levelImg);
        std::cout << "----- Image levels:" << std::endl;
        levelImg.printMeshT(3, 1);

        // Show intensities and levels of each particle
        std::cout << "----- Particle intensities:\n";
        for (uint64_t i = 0; i < particleIntensities.size(); i++) std::cout << particleIntensities.data[i] << " ";
        std::cout << std::endl;

        particleIntensities.fill_with_levels(apr);

        std::cout << "----- Particle levels:\n";
        for (uint64_t  i = 0; i < particleIntensities.size(); i++) std::cout << particleIntensities.data[i] << " ";
        std::cout << std::endl;

        // Show some general information about generated APR
        double computational_ratio = (1.0 * apr.org_dims(0) * apr.org_dims(1) * apr.org_dims(2)) / (1.0 * apr.total_number_particles());
        std::cout << std::endl;
        std::cout << "#pixels: " << (apr.org_dims(0) * apr.org_dims(1) * apr.org_dims(2)) << " #particles: " << (apr.total_number_particles()) << std::endl;
        std::cout << "Computational Ratio (Pixels/Particles): " << std::setprecision(2) << computational_ratio << std::endl;
    }

    TEST(PullingSchemeTest, PullingScheme1D) {

        //int values[] = {4,4,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };
//        int values[] = {3,2,2,2, 2,2,1,1};
//        int values[] = {3,0,0,0, 0,0,0,0};
//        int values[] = {3,0,0,0, 0,0,0,0};
//        int values[] = {4,0,0,0, 0,0,0,0, 4,0,0,0, 0,0,0,0};
        int values[] = {0,2,2,3, 4,5,6,7};
        int len = sizeof(values)/sizeof(int);
        PixelData<int> levels(len ,1, 1);
        initFromZYXarray(levels, values);
        levels.printMeshT(3, 1);

        GenInfo gi;
        const PixelDataDim dim = levels.getDimension();
        gi.init(dim.y * 2, dim.x, dim.z); // time two in y-direction since PS container is downsized.
        std::cout << gi << std::endl;

        APRTimer t(true);

        t.start_timer("PS1");
        PullingScheme ps;
        ps.initialize_particle_cell_tree(gi);
        int l_max = gi.l_max - 1;
        int l_min = gi.l_min;
        std::cout << "PS: max/max min/min" << l_max << " " << ps.pct_level_max() << "  " << l_min << " " << ps.pct_level_min() << std::endl;
        ps.fill(l_max, levels);
        std::cout << "LEVEL: " << l_max << std::endl; levels.printMeshT(3, 1);
        PixelData<int> levelsDS;
        for(int l = l_max - 1; l >= l_min; l--){
            downsample(levels, levelsDS,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
            ps.fill(l, levelsDS);
            std::cout << "LEVEL: " << l << std::endl; levelsDS.printMeshT(3, 1);
            levels.swap(levelsDS);
        }
        printParticleCellTree(ps.getParticleCellTree());
        ps.pulling_scheme_main();
        t.stop_timer();

        std::cout << "----------PS:\n";
        printParticleCellTree(ps.getParticleCellTree());
        std::cout << "-------------\n";

        LinearAccess linearAccess;
        linearAccess.genInfo = &gi;
        APRParameters par;
        std::cout << "1\n";
        linearAccess.initialize_linear_structure(par, ps.getParticleCellTree());
        std::cout << "2\n";
        LinearIterator it(linearAccess, gi);

        std::cout << "===========================\n";
        for (int level = it.level_min(); level <= it.level_max(); ++level) {
            for (int z = 0; z < it.z_num(level); z++) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end(); it++) {
                        std::cout << "              " << z << " " << x << " " << it.y() << " " << level << std::endl;
                    }
                }
            }
        }
        std::cout << std::endl;
    }

    TEST(PullingSchemeTest, Simple) {
        GenInfo gi;
        // TODO: Investigate why OVPC fails if one of the dimension is equal to 1
        //       Investigate why sub-dimension in printParticleCellTree is different in OVPC nad PS
        gi.init(8, 1, 2);

        std::cout << gi << std::endl;

        PixelData<int> levels = getRandInitializedMesh<int>(
                std::ceil(gi.org_dims[0]/2),
                std::ceil(gi.org_dims[1]/2),
                std::ceil(gi.org_dims[2]/2),
                gi.l_max + 1);
        PixelData<int> levels2(levels, true);
//        float values[] = {1, 1, 4, 1,    1, 1, 1, 1,    1, 1, 3, 3,    1, 1, 1, 1, 1, 1, 4, 1,    1, 1, 1, 1,    1, 1, 3, 3,    1, 1, 1, 1};
//        initFromZYXarray(levels, values);

//        levels.printMeshT(3, 1);

        APRTimer t(true);

        t.start_timer("PS1");
        PullingScheme ps;
        ps.initialize_particle_cell_tree(gi);
        int l_max = gi.l_max - 1;
        int l_min = gi.l_min;
        ps.fill(l_max, levels);
        PixelData<int> levelsDS;
        for(int l_ = l_max - 1; l_ >= l_min; l_--){
            downsample(levels, levelsDS,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
            ps.fill(l_,levelsDS);
            levels.swap(levelsDS);
        }
        ps.pulling_scheme_main();
        t.stop_timer();

        t.start_timer("OVPC1");
        OVPC nps(gi, levels2);
        t.stop_timer();
        t.start_timer("OVPC2");
        nps.generateTree();
        t.stop_timer();

        std::cout << "----------OVPC:\n";
        printParticleCellTree(nps.getParticleCellTree());
        std::cout << "----------PS:\n";
        printParticleCellTree(ps.getParticleCellTree());
        std::cout << "-------------\n";

        for (int l = l_min; l <= l_max; ++l)
            compareParticleCellTrees(ps.getParticleCellTree()[l], nps.getParticleCellTree()[l]);

    }


    TEST(PullingSchemeTest, NEWvsOLD) {
        GenInfo access;
        access.l_max = 9;
        access.l_min = 1;
        access.org_dims[0] = std::pow(2, access.l_max);
        access.org_dims[1] = std::pow(2, access.l_max);
        access.org_dims[2] = std::pow(2, access.l_max);
        int l = access.l_max - 1;

        PixelData<int> levels = getRandInitializedMesh<int>(access.org_dims[0]/2,access.org_dims[1]/2,access.org_dims[2]/2, access.l_max + 1);
        PixelData<int> levels2(levels, true);
//        float values[] = {1, 1, 4, 1,    1, 1, 1, 1,    1, 1, 3, 3,    1, 1, 1, 1, 1, 1, 4, 1,    1, 1, 1, 1,    1, 1, 3, 3,    1, 1, 1, 1};
//        initFromZYXarray(levels, values);

//        levels.printMeshT(3, 1);

        APRTimer t(true);

        t.start_timer("PS1");
        PullingScheme ps;
        ps.initialize_particle_cell_tree(access);
        int l_max = access.l_max - 1;
        int l_min = access.l_min;
        ps.fill(l_max, levels);
        PixelData<int> levelsDS;
        for(int l_ = l_max - 1; l_ >= l_min; l_--){
            downsample(levels, levelsDS,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);
            ps.fill(l_,levelsDS);
            levels.swap(levelsDS);
        }
        ps.pulling_scheme_main();
        t.stop_timer();

        t.start_timer("OVPC1");
        OVPC nps(access, levels2);
        t.stop_timer();
        t.start_timer("OVPC2");
        nps.generateTree();
        t.stop_timer();

//        printParticleCellTree(nps.getParticleCellTree());
//        printParticleCellTree(ps.getParticleCellTree());

        for (l = l_min; l <= l_max; ++l)
        compareParticleCellTrees(ps.getParticleCellTree()[l], nps.getParticleCellTree()[l]);

    }

//    TEST(PullingSchemeTest, Init) {
//
//        GenInfo access;
//        access.l_max = 5;
//        access.l_min = 1;
//        access.org_dims[0] = 32;
//        access.org_dims[1] = 1;
//        access.org_dims[2] = 1;
//
//        PullingScheme ps;
//        ps.initialize_particle_cell_tree(access);
//        std::vector<PixelData<uint8_t>> &pctree = ps.getParticleCellTree();
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>> Initialized tree:\n";
//        printParticleCellTree(pctree);
//        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
//
//        // TEST: check if zeroed and correct number of levels
//        ASSERT_EQ(access.l_max, pctree.size()); // all levels [0, access.level_max - 1]
//        for (int l = 0; l < pctree.size(); ++l) {
//            auto &tree = pctree[l];
//            for (auto &e : tree.mesh) {
//                ASSERT_EQ(0, e);
//            }
//        }
//
//        // Generate mesh with test levels
//        PixelData<float> levels(pctree.back(), false);// = generateLevels(pctree[access.l_max - 1], access.l_max);
////        float values[] = {4, 1, 1, 1,    1, 1, 1, 2};
//        float values[] = {1, 1, 4, 1,    1, 1, 1, 1,    1, 1, 3, 3,    1, 1, 1, 1};
//        initFromZYXarray(levels, values);
//
//
//        OVPC nps(access, levels);
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      NPS1:\n";
//        printParticleCellTree(nps.getParticleCellTree());
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      NPS1:\n";
//        nps.generateTree();
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      NPS2:\n";
//        printParticleCellTree(nps.getParticleCellTree());
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      NPS2:\n";
//        // Fill particle cell tree with levels
//        int l_max = access.l_max - 1;
//        int l_min = access.l_min;
//        ps.fill(l_max, levels);
//
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      LEVELS:\n";
//        levels.printMeshT(3,0);
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      LEVELS:\n";
//
//        PixelData<float> levelsDS;
//        for(int l_ = l_max - 1; l_ >= l_min; l_--){
//            //down sample the resolution level k, using a max reduction
//            downsample(levels, levelsDS,
//                       [](const float &x, const float &y) -> float { return std::max(x, y); },
//                       [](const float &x) -> float { return x; }, true);
//            levelsDS.printMeshT(3, 0);
//            ps.fill(l_,levelsDS);
//            levelsDS.printMeshT(3,0);
//            levels.swap(levelsDS);
//        }
//
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      Filled tree:\n";
//        printParticleCellTree(pctree);
//        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
//
////        ps.fill_neighbours(l_max);
////        pctree[l_max].printMesh(3, 0);
//
//
//        ps.pulling_scheme_main();
//        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>      MAIN   tree:\n";
//        printParticleCellTree(pctree);
//        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
//
//        access.initialize_structure_from_particle_cell_tree(false, ps.getParticleCellTree());
//        std::cout << "NUM OF PARTICLES: " << access.get_total_number_particles() << std::endl;
//
//
//        APRIterator apr_iterator(access);
//        std::cout << "Total number of particles: " << apr_iterator.total_number_particles() << std::endl;
//
//        int prev = 0;
//        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
//            std::cout << "Level: " << level << std::endl;
//            int w = (int) (std::pow(2, 5-level) * 3);
//            for (int z = 0; z < apr_iterator.spatial_index_z_max(level); ++z) {
//                for (int x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
//                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {
//                        for (int i = prev; i < apr_iterator.y(); ++i )  std::cout << std::setw(w) << ".";
//                        std::cout << std::setw(w) << apr_iterator.y();
//                        prev = apr_iterator.y() + 1;
//                    }
//                    for (int pp = prev; pp < apr_iterator.spatial_index_y_max(level); ++pp)
//                        std::cout << std::setw(w) << ".";
//
//                    prev = 0;
//                    std::cout << std::endl;
//                }
//                std::cout << std::endl;
//            }
//        }
//
//    }


}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
