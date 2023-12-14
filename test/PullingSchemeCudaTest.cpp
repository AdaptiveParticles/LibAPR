#include <gtest/gtest.h>

#include "algorithm/PullingScheme.hpp"
#include "algorithm/OVPC.h"

#include "algorithm/PullingSchemeCuda.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "algorithm/APRConverter.hpp"

#include "TestTools.hpp"

/**
 * Prints PCT
 * @param particleCellTree
 */
template <typename T>
void printParticleCellTree(const std::vector<PixelData<T>> &particleCellTree) {
    for (uint64_t  l = 0; l < particleCellTree.size(); ++l) {
        auto &tree = particleCellTree[l];
//            std::cout << "-- level = " << l << ",  " << tree << std::endl;
        tree.printMeshT(3,0);
    }
}
/**
 * Compare
 * @param expected - expected levels
 * @param tested - levels to verify
 * @param maxError
 * @param maxNumOfErrPrinted - how many error outputs should be printed
 * @return
 */
template <typename T, typename W>
int compareParticleCellTrees(const std::vector<PixelData<T>> &expected, const std::vector<PixelData<W>> &tested, int maxNumOfErrPrinted = 3) {
    int cntGlobal = 0;
    for (size_t level = 0; level < expected.size(); level++) {
        int cnt = 0;
        int numOfParticles = 0;
        for (size_t i = 0; i < expected[level].mesh.size(); ++i) {
            if (expected[level].mesh[i] < 8 && tested[level].mesh[i] <= FILLER_TYPE) {
                if (std::abs(expected[level].mesh[i] - tested[level].mesh[i]) > 0 || std::isnan(expected[level].mesh[i]) ||
                    std::isnan(tested[level].mesh[i])) {
                    if (cnt < maxNumOfErrPrinted || maxNumOfErrPrinted == -1) {
                        std::cout << "Level: " << level <<" ERROR expected vs tested mesh: " << (float) expected[level].mesh[i] << " vs "
                                  << (float) tested[level].mesh[i] << " IDX:" << tested[level].getStrIndex(i) << std::endl;
                    }
                    cnt++;
                }
                if (expected[level].mesh[i] > 0) numOfParticles++;
            }
        }
        cntGlobal += cnt;
        if (cnt > 0) std::cout << "Level: " << level << ", Number of errors / all points: " << cnt << " / " << expected[level].mesh.size() << " Particles:" << numOfParticles << std::endl;
    }
    return cntGlobal;
}

template<typename DataType>
void fillPS(PullingScheme &aPS, PixelData<DataType> &levels) {
    auto l_max = aPS.pct_level_max();
    auto l_min = aPS.pct_level_min();

//        std::cout << "LEVEL: " << l_max << std::endl; levels.printMeshT(3, 1);

    aPS.fill(l_max, levels);
    PixelData<int> levelsDS;
    for (int l = l_max - 1; l >= l_min; l--) {
        downsample(levels, levelsDS,
                   [](const float &x, const float &y) -> float { return std::max(x, y); },
                   [](const float &x) -> float { return x; }, true);
        aPS.fill(l, levelsDS);
//            std::cout << "LEVEL: " << l << std::endl; levelsDS.printMeshT(3, 1);
        levels.swap(levelsDS);
    }
}

TEST(PullingSchemeTest, DeleteMeAfterDevelopment) {
    // TODO: delete me after development
    // Full 'get apr' pipeline to test imp. on different stages
    // Useful during debugging and can be removed once finished

    // Prepare input data (image)
    int values[] = {9,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    // PS input values = 5  0  0  0  0  0  0  0

//         int values[] = {3,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 3,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, };
//         PullingScheme input values (local_scale_temp) for above 'image' = {6  0  0  0  0  0  0  0  6  0  0  0  0  0  0  0};

    int len = sizeof(values)/sizeof(int);
    PixelData<int> data(len, 1, 1);
    initFromZYXarray(data, values);
    std::cout << "----- Input image:\n";
    data.printMeshT(3, 1);

    // Produce APR
    APR apr;
    APRConverter<uint16_t> aprConverter;
    aprConverter.par.rel_error = 0.1;
    aprConverter.par.lambda = 0.1;
    aprConverter.par.sigma_th = 0.0001;
    aprConverter.par.neighborhood_optimization = true;
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

    int values[] = {0,0,0,5, 0,0,0,0};
    int len = sizeof(values)/sizeof(int);

    PixelData<int> levels(3,3,3, 0);
    levels(2,2,2) = 11;

//        initFromZYXarray(levels, values);
    levels.printMeshT(3, 1);

    GenInfo gi;
    const PixelDataDim dim = levels.getDimension();
    std::cout << "Levels dim: " << dim << std::endl;
    gi.init(dim.y * 2, dim.x * 2, dim.z * 2); // time two in y-direction since PS container is downsized.
    std::cout << gi << std::endl;

    APRTimer t(true);

    t.start_timer("PS1");
    PullingScheme ps;
    ps.initialize_particle_cell_tree(gi);
    int l_max = gi.l_max - 1;
    int l_min = gi.l_min;
    std::cout << "PS: max/max min/min" << l_max << " " << ps.pct_level_max() << "  " << l_min << " " << ps.pct_level_min() << std::endl;

    fillPS(ps, levels);

    std::cout << "---------- Filled PS tree\n";
    printParticleCellTree(ps.getParticleCellTree());
    std::cout << "---------------\n";

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

//    TEST(PullingSchemeCudaTest, computeLevels) {
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



TEST(PullingSchemeCudaTest, DS) {
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
