#include <gtest/gtest.h>

#include "algorithm/PullingScheme.hpp"
#include "algorithm/OVPC.h"

#include "algorithm/PullingSchemeCuda.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "algorithm/APRConverter.hpp"
#include "algorithm/LocalParticleCellSet.hpp"

#include "TestTools.hpp"


// Class for storing expected values for one element of Particle Cell Tree (output of Pulling Scheme)
class LevelData  {
public:
    int level;
    int y;
    int x;
    int z;
    uint8_t expectedType; // seed, boundary, filler...
};

/**
 * Verify computed Particle Cell Tree (PCT) vs expected values
 * Expected values should list all data for types=1,2,3 (seed, boundary filler) which are used to generate particles:
 * {levels, y,x,z(position), type}
 * All other values are ignored (and used by Pulling Scheme (PS) only for intermediate calculations)
 * @param aPCT - PCT produces by PS (note: values in PCT will be changed during verification!)
 * @param expectedValues expected values
 * @return true if correct, false otherwise
 */
template<typename ElementType>
bool verifyParticleCellTree(std::vector<PixelData<ElementType>> &aPCT, const std::vector<LevelData> &expectedValues) {

    const uint8_t AlreadyCheckedMark = 255;
    const uint8_t MaxValueOfImportantType = FILLER_TYPE; // All types above are used by PS during computation phase only

    for (const auto &r : expectedValues) {
        // std::cout << r.level << " " << r.y << "," << r.x << "," << r.z << " " << (int)r.expectedType << std::endl;

        auto &v = aPCT[r.level](r.y, r.x, r.z);
        // Add dim. checks for accessing pct
        if (v == r.expectedType) {
            v = AlreadyCheckedMark;
        }
        else {
            std::cout << "Error! Data on level=" << r.level << " at (" << r.y << "," << r.x << "," << r.z << ") expected=" << (int)r.expectedType << " got=" << (int)v << std::endl;
            return false;
        }
    }

    for (size_t level = 0; level < aPCT.size(); level++) {
        auto &d = aPCT[level];
        auto y_num = d.y_num;
        auto x_num = d.x_num;
        auto z_num = d.z_num;

        for (int j = 0; j < z_num; j++) {
            for (int i = 0; i < x_num; i++) {
                for (int k = 0; k < y_num; k++) {
                    const auto &v = d(k, i, j);
                    if (v != AlreadyCheckedMark && v <= MaxValueOfImportantType && v > 0) {
                        std::cout << "Error! Data on level = " << level << " at (" << k << "," << i << "," << j << ") with value = " << (int)v << " not verified or bad!" << std::endl;
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

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
    PixelData<float> levelsDS(ceil(levels.y_num/2.0), ceil(levels.x_num/2.0), ceil(levels.z_num/2.0));
    LocalParticleCellSet().get_local_particle_cell_set(aPS, levels, levelsDS, APRParameters());
}

TEST(PullingSchemeTest, DeleteMeAfterDevelopment_fullAprPipeline) {
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



TEST(PullingSchemeTest, DeleteMeAfterDevelopment_PS) {
    // TODO: delete me after development
    // Runs PS to test imp. on different stages
    // Useful during debugging and can be removed once finished
    int values[] = {0,0,0,5, 0,0,0,0};
    int len = sizeof(values)/sizeof(int);

    PixelData<float> levels(3,3,3, 0);
    levels(2,2,2) = 11;

//        initFromZYXarray(levels, values);
    levels.printMeshT(3, 1);

    GenInfo gi;
    const PixelDataDim dim = levels.getDimension();
    std::cout << "Levels dim: " << dim << std::endl;
    gi.init(dim.y * 2, dim.x * 2, dim.z * 2); // time two in y-direction since PS container is downsized.
    std::cout << gi << std::endl;

    APRTimer t(false);

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


TEST(PullingSchemeTest, PSvsOVPCCUDA) {
    // Generates random levels in a 3D cube and then compares generated output levels in PS and OVPC
    GenInfo gi;
    gi.init(255, 257, 199);

    // Generate random levels for PS and OVPC
    PixelData<float> levels(std::ceil(gi.org_dims[0]/2.0),
                            std::ceil(gi.org_dims[1]/2.0),
                            std::ceil(gi.org_dims[2]/2.0),
                            0);
    // Add a few particles only - it will end up with Pulling Scheme generate particles on (almost) all
    // levels - good case to compare with OVPC
    const int numOfParticles = 3;
    std::srand(std::time(nullptr));
    for (int i = 0; i < numOfParticles; ++i) {
        levels(std::rand() % levels.y_num, std::rand() % levels.x_num, std::rand() % levels.z_num) = gi.l_max;
    }
    PixelData<float> levelsOVPC(levels, true); // just copy 'levels'
    PixelData<float> levelsPS(levels, true);

    // Initialize all needed objects
    APRTimer t(false);

    t.start_timer("PS - init");
    PullingScheme ps;
    ps.initialize_particle_cell_tree(gi);
    fillPS(ps, levelsPS);
    t.stop_timer();
    t.start_timer("PS - compute");
    ps.pulling_scheme_main();
    t.stop_timer();

    // Run test methods and compare results
    t.start_timer("OVPCCUDA - init");
    int levelMax = gi.l_max - 1;
    int levelMin = gi.l_min;
    std::vector<PixelData<uint8_t>> pct = PullingScheme::generateParticleCellTree(gi);
    t.stop_timer();
    t.start_timer("OVPCCUDA - compute");
    computeOvpcCuda(levelsOVPC, pct, levelMin, levelMax);
    t.stop_timer();

    // -------------- Verify result
    ASSERT_EQ(compareParticleCellTrees(ps.getParticleCellTree(), pct), 0);
}

TEST(PullingSchemeTest, OVPCCUDA_Ydir) {
    // Prepare input data for PS
    float values[] = {9,0,0,0, 0,0,0,0};
    int len = sizeof(values)/sizeof(int);
    PixelData<float> levels(len, 1, 1);  // <-- Y-dir
    initFromZYXarray(levels, values); // <-- Y-dir

    // Prepare GenInfo structure -
    // remember: data for PS is downsampled so is representing image twice bigger so Y-dir size need to be multiplied by 2
    GenInfo gi;
    const PixelDataDim dim = levels.getDimension();
    gi.init(2 * dim.y, dim.x, dim.z); // <-- Y-dir

    int levelMax = gi.l_max - 1;
    int levelMin = gi.l_min;

    // Initialize all needed objects
    APRTimer t(false);

    t.start_timer("OVPCCUDA - initialize");
    std::vector<PixelData<uint8_t>> pct = PullingScheme::generateParticleCellTree(gi);
    t.stop_timer();

    t.start_timer("OVPCCUDA - compute");
    computeOvpcCuda(levels, pct, levelMin, levelMax);
    t.stop_timer();

    // List of expected types
    std::vector<LevelData> ev = {
            {3, 0,0,0, 1},
            {3, 1,0,0, 2},
            {3, 2,0,0, 3},
            {3, 3,0,0, 3},

            {2, 2,0,0, 3},
            {2, 3,0,0, 3}
    };

    // -------------- Verify result
    EXPECT_TRUE(verifyParticleCellTree(pct, ev));
}

TEST(PullingSchemeTest, OVPCCUDA_Xdir) {
    // Prepare input data for PS
    int values[] = {9,0,0,0, 0,0,0,0};
    int len = sizeof(values)/sizeof(int);
    PixelData<int> levels(1, len, 1);  // <-- X-dir
    initFromZYXarray(levels, values);

    // Prepare GenInfo structure -
    // remember: data for PS is downsampled so is representing image twice bigger so Y-dir size need to be multiplied by 2
    GenInfo gi;
    const PixelDataDim dim = levels.getDimension();
    gi.init(dim.y, 2 * dim.x, dim.z); // <-- X-dir

    int levelMax = gi.l_max - 1;
    int levelMin = gi.l_min;

    // Initialize all needed objects
    APRTimer t(false);

    t.start_timer("OVPCCUDA - initialize");
    std::vector<PixelData<uint8_t>> pct = PullingScheme::generateParticleCellTree(gi);
    t.stop_timer();

    t.start_timer("OVPCCUDA - compute");
    computeOvpcCuda(levels, pct, levelMin, levelMax);
    t.stop_timer();

    // List of expected types
    std::vector<LevelData> ev = {
            {3, 0,0,0, 1},
            {3, 0,1,0, 2},
            {3, 0,2,0, 3},
            {3, 0,3,0, 3}  ,

            {2, 0,2,0, 3},
            {2, 0,3,0, 3}
    };

    // -------------- Verify result
    EXPECT_TRUE(verifyParticleCellTree(pct, ev));
}

TEST(PullingSchemeTest, OVPCCUDA_Zdir) {
    // Prepare input data for PS
    int values[] = {9,0,0,0, 0,0,0,0};
    int len = sizeof(values)/sizeof(int);
    PixelData<int> levels(1, 1, len);  // <-- Z-dir
    initFromZYXarray(levels, values);

    // Prepare GenInfo structure -
    // remember: data for PS is downsampled so is representing image twice bigger so Y-dir size need to be multiplied by 2
    GenInfo gi;
    const PixelDataDim dim = levels.getDimension();
    gi.init(dim.y, dim.x, 2 * dim.z); // <-- Z-dir

    int levelMax = gi.l_max - 1;
    int levelMin = gi.l_min;

    // Initialize all needed objects
    APRTimer t(false);

    t.start_timer("OVPCCUDA - initialize");
    std::vector<PixelData<uint8_t>> pct = PullingScheme::generateParticleCellTree(gi);
    t.stop_timer();

    t.start_timer("OVPCCUDA - compute");
    computeOvpcCuda(levels, pct, levelMin, levelMax);
    t.stop_timer();

    // List of expected types
    std::vector<LevelData> ev = {
            {3, 0,0,0, 1},
            {3, 0,0,1, 2},
            {3, 0,0,2, 3},
            {3, 0,0,3, 3}  ,

            {2, 0,0,2, 3},
            {2, 0,0,3, 3}
    };

    // -------------- Verify result
    EXPECT_TRUE(verifyParticleCellTree(pct, ev));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
