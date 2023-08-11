#include <gtest/gtest.h>

#include "algorithm/PullingScheme.hpp"
#include "algorithm/OVPC.h"

#include "algorithm/PullingSchemeCuda.hpp"
#include "algorithm/ComputeGradientCuda.hpp"

#include "TestTools.hpp"

//    TEST(PullingSchemeTest, computeLevels) {
//        using ImgType = float;
//        const int maxLevel = 3;
//        const float relError = 0.1;
//
//        PixelData<ImgType> grad = getRandInitializedMesh<ImgType>(10, 20, 33);
//        PixelData<float> localIntensityScaleCpu = getRandInitializedMesh<float>(10, 20, 33);
//
//        PixelData<float> localIntensityScaleGpu(localIntensityScaleCpu, true);
//        PixelData<float> elo(localIntensityScaleCpu, true);
//        APRTimer timer(true);
//
//        timer.start_timer("CPU Levels");
//        APRConverter<ImgType>().computeLevels(grad, localIntensityScaleCpu, maxLevel, relError);
//        timer.stop_timer();
//
//        timer.start_timer("GPU Levels");
//        computeLevelsCuda(grad, localIntensityScaleGpu, maxLevel, relError);
//        timer.stop_timer();
//
//        EXPECT_EQ(compareMeshes(localIntensityScaleCpu, localIntensityScaleGpu), 0);
//    }



TEST(PullingSchemeTest, DS) {
    GenInfo access;
    access.l_max = 11;
    access.l_min = 1;
    access.org_dims[0] = std::pow(2, access.l_max)/2;
    access.org_dims[1] = std::pow(2, access.l_max)/2;
    access.org_dims[2] = std::pow(2, access.l_max);


    PixelData<float> levels = getRandInitializedMesh<float>(access.org_dims[0]/2,access.org_dims[1]/2,access.org_dims[2]/2, access.l_max + 1);
    PixelData<float> levels2(levels, true);

    //        PixelData<float> levels(16,1,1);
    //        float values[] = {4, 1, 1, 1,   1, 1, 1, 2,   3, 1, 1, 1,   1, 1, 1, 2};
    //        initFromZYXarray(levels, values);

    APRTimer t(true);
    if (false)    {
        t.start_timer("PS1");
        PullingScheme ps;
        ps.initialize_particle_cell_tree(access);
        int l_max = access.l_max - 1;
        int l_min = access.l_min;
        ps.fill(l_max, levels2);
        PixelData<float> levelsDS;
        for (int l_ = l_max - 1; l_ >= l_min; l_--) {
            downsample(levels, levelsDS,
            [](const float &x, const float &y) -> float { return std::max(x, y); },
            [](const float &x) -> float { return x; }, true);
            ps.fill(l_, levelsDS);
            levels2.swap(levelsDS);
        }
        t.stop_timer();
    }
    {
        t.start_timer("CUDA");
        int levelMax = access.l_max - 1;
        int levelMin = access.l_min;
        PixelData<TreeElementType> ds(levels.y_num, levels.x_num, levels.z_num * (levelMax - levelMin + 1), 0);
        std::cout << levels << std::endl;
        //        std::cout << ds << std::endl;
        computeOVPC(levels, ds, levelMin, levelMax);
        //        ds.printMeshT(3,1);
        t.stop_timer();
    }
    {
        t.start_timer("OVPC1");
        OVPC nps(access, levels);
        nps.generateTree();
        t.stop_timer();
        //        printParticleCellTree(nps.getParticleCellTree());
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
