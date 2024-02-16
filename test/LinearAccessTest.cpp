#include <gtest/gtest.h>

#include "algorithm/PullingScheme.hpp"
#include "algorithm/LocalParticleCellSet.hpp"
#include "algorithm/APRConverter.hpp"

#include "TestTools.hpp"

template<typename DataType>
void fillPS(PullingScheme &aPS, PixelData<DataType> &levels) {
    PixelData<DataType> levelsDS(ceil(levels.y_num/2.0), ceil(levels.x_num/2.0), ceil(levels.z_num/2.0));
    LocalParticleCellSet().get_local_particle_cell_set(aPS, levels, levelsDS, APRParameters());
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
//    int values[] = {0,0,0,5, 0,0,0,0};
//    int len = sizeof(values)/sizeof(int);

    PixelData<int> levels(2, 2, 2, 0);
    levels(0,0,0) = 4;

//    initFromZYXarray(levels, values);
    std::cout << "---------------\n";
    levels.printMeshT(3, 1);
    std::cout << "---------------\n";

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
    linearAccess.initialize_linear_structure(par, ps.getParticleCellTree());

    std::cout << gi << std::endl;
    auto prt = [&](const auto& v){ std::cout << "size=" << v.size() << " data="; for (size_t i = 0; i < v.size(); i++) std::cout << v[i] << " "; std::cout << std::endl; };
    prt(linearAccess.y_vec);
    prt(linearAccess.xz_end_vec);
    prt(linearAccess.level_xz_vec);

    LinearIterator it(linearAccess, gi);
    for (int l = 0; l <= 3; l++) {
        std::cout << it.particles_level_begin(l) << " " << it.particles_level_end(l) << std::endl;
    }
    std::cout << "NumOfParticles: " << gi.total_number_particles << std::endl;

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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}