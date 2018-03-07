#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRIterator.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/ExtraParticleData.hpp"


struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

bool command_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv) {
    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }
    if(command_option_exists(argv, argv + argc, "-i")) {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    if(command_option_exists(argv, argv + argc, "-d")) {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-o")) {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct DenseRepresentation {
    ExtraPartCellData<uint16_t> yRow;
    ExtraPartCellData<uint64_t> globalIdxOfYrow;
    int levelMin;
    int levelMax;

    friend std::ostream & operator<<(std::ostream &os, const DenseRepresentation &obj) {
        os << "DenseRepresentation: levelMin=" << obj.levelMin << ", levelMax=" << obj.levelMax
           << ", min/max num of yRows=" << obj.yRow.data[obj.levelMin].size() << "/" << obj.yRow.data[obj.levelMax].size();
        return os;
    }
};

/**
 * Produce dense representation from provided APR iterator (it may come from APR or APR tree)
 */
template<typename T>
DenseRepresentation getDenseRepresentation(APRIterator<T> &aprIt) {
    // Create structures to hold y-rows and starting global indices of each row.
    ExtraPartCellData<uint16_t> yRow;
    yRow.data.resize(aprIt.level_max() + 1);
    ExtraPartCellData<uint64_t> globalIdxOfYrow;
    globalIdxOfYrow.data.resize(aprIt.level_max() + 1);
    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
        const size_t x_num = aprIt.spatial_index_x_max(level);
        const size_t z_num = aprIt.spatial_index_z_max(level);

        const size_t numberOfRows = x_num * z_num;
        yRow.data[level].resize(numberOfRows);
        globalIdxOfYrow.data[level].resize(numberOfRows);

        for (size_t z = 0; z < z_num; ++z) {
            for (size_t x = 0; x < x_num; ++x) {
                if (aprIt.set_new_lzx(level, z, x) < UINT64_MAX) { // It means that we have y-values in row (TODO: do it better than UINT64_MAX)
                    //Storing the first global index in each row (level,x,z row)
                    globalIdxOfYrow.data[level][aprIt.z()*x_num + aprIt.x()].push_back(aprIt.global_index());
                    //Storing the non-zero y values in each row (level,x,z row)
                    const auto maxGlobalIdxInRow = aprIt.particles_zx_end(level, z, x);
                    const auto numOfElementsInRow = maxGlobalIdxInRow - aprIt.global_index();
                    yRow.data[level][z * x_num + x].reserve(numOfElementsInRow);
                    while (aprIt.global_index() < maxGlobalIdxInRow) {
                        yRow.data[level][z * x_num + x].push_back(aprIt.y());
                        aprIt.set_iterator_to_particle_next_particle();
                    }
                }
            }
        }
    }

    return DenseRepresentation{ yRow, globalIdxOfYrow, aprIt.level_min(), aprIt.level_max() };
}

template<typename T>
ExtraParticleData<float> maximumDownsampling(APR<T> &aInputApr, const DenseRepresentation &dr, const DenseRepresentation &drTree, APRTree<uint16_t> &aprTree) {
    APRIterator<uint16_t> aprIt(aInputApr);
    const auto &intensities = aInputApr.particles_intensities;
    ExtraParticleData<float> partData(aprTree);
    APRTreeIterator<uint16_t> treeIterator(aprTree);

    // Update tree from APR
    for (int level = dr.levelMax; level >= dr.levelMin; --level) {
        const size_t x_num = aprIt.spatial_index_x_max(level);
        const size_t z_num = aprIt.spatial_index_z_max(level);
        const size_t x_num_ds = treeIterator.spatial_index_x_max(level - 1);
        const size_t y_num_ds = treeIterator.spatial_index_y_max(level - 1);
        const size_t z_num_ds = treeIterator.spatial_index_z_max(level - 1);

        for (size_t zds = 0; zds < z_num_ds; ++zds) {
            for (size_t xds = 0; xds < x_num_ds; ++xds) {
                // Calculate max of downsampled y-row
                std::vector<float> dsData(y_num_ds, 0);
                for (size_t z = zds*2; z <= std::min(zds*2+1,z_num-1); ++z) {
                    for (size_t x = xds*2; x <= std::min(xds*2+1,x_num-1); ++x) {
                        if (dr.globalIdxOfYrow.data[level][z * x_num + x].size() > 0) { // we have some y?
                            auto currentGlobalIdx = dr.globalIdxOfYrow.data[level][z * x_num + x][0];
                            auto &currentRow = dr.yRow.data[level][z * x_num + x];
                            for (auto y : currentRow) {
                                auto val = intensities.data[currentGlobalIdx];
                                if (val > dsData[y/2]) dsData[y/2] = val;  // compute maximum
                                ++currentGlobalIdx;
                            }
                        }
                    }
                }

                // Update a tree
                if (drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds].size() > 0) { // we have some y?
                    auto currentGlobalIdx = drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds][0];
                    auto &currentRow = drTree.yRow.data[level-1][zds * x_num_ds + xds];
                    for (auto &y : currentRow) {
                        partData.data[currentGlobalIdx] = dsData[y];
                        ++currentGlobalIdx;
                    }
                }
            }
        }
    }


    // propagate changes up to the lowest level in the tree
    for (int level = drTree.levelMax; level > drTree.levelMin; --level) {
        const size_t x_num = aprIt.spatial_index_x_max(level);
        const size_t z_num = aprIt.spatial_index_z_max(level);
        const size_t x_num_ds = treeIterator.spatial_index_x_max(level - 1);
        const size_t y_num_ds = treeIterator.spatial_index_y_max(level - 1);
        const size_t z_num_ds = treeIterator.spatial_index_z_max(level - 1);


        for (size_t zds = 0; zds < z_num_ds; ++zds) {
            for (size_t xds = 0; xds < x_num_ds; ++xds) {
                // Calculate max of downsampled y-row
                std::vector<float> dsData(y_num_ds, 0);
                for (size_t z = zds*2; z <= std::min(zds*2+1,z_num-1); ++z) {
                    for (size_t x = xds*2; x <= std::min(xds*2+1,x_num-1); ++x) {
                        if (drTree.globalIdxOfYrow.data[level][z * x_num + x].size() > 0) { // we have some y?
                            auto currentGlobalIdx = drTree.globalIdxOfYrow.data[level][z * x_num + x][0];
                            auto &currentRow = drTree.yRow.data[level][z * x_num + x];
                            for (auto y : currentRow) {
                                auto val = partData.data[currentGlobalIdx];
                                if (val > dsData[y/2]) dsData[y/2] = val;  // compute maximum
                                ++currentGlobalIdx;
                            }
                        }
                    }
                }

                // Update a tree
                if (drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds].size() > 0) { // we have some y?
                    auto currentGlobalIdx = drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds][0];
                    auto &currentRow = drTree.yRow.data[level-1][zds * x_num_ds + xds];
                    for (auto &y : currentRow) {
                        auto val = dsData[y];
                        if (val > partData.data[currentGlobalIdx]) partData.data[currentGlobalIdx] = val;
                        ++currentGlobalIdx;
                    }
                }
            }
        }
    }
    return partData;
}

template<typename T>
ExtraParticleData<float> meanDownsampling(APR<T> &aInputApr, const DenseRepresentation &dr, const DenseRepresentation &drTree, APRTree<uint16_t> &aprTree) {
    APRIterator<uint16_t> aprIt(aInputApr);
    const auto &intensities = aInputApr.particles_intensities;
    ExtraParticleData<float> ouputTree(aprTree);
    ExtraParticleData<uint8_t> childCnt(aprTree);
    APRTreeIterator<uint16_t> treeIterator(aprTree);

    // Update tree from APR
    for (int level = dr.levelMax; level >= dr.levelMin; --level) {
        const size_t x_num = aprIt.spatial_index_x_max(level);
        const size_t z_num = aprIt.spatial_index_z_max(level);
        const size_t x_num_ds = treeIterator.spatial_index_x_max(level - 1);
        const size_t y_num_ds = treeIterator.spatial_index_y_max(level - 1);
        const size_t z_num_ds = treeIterator.spatial_index_z_max(level - 1);

        for (size_t zds = 0; zds < z_num_ds; ++zds) {
            for (size_t xds = 0; xds < x_num_ds; ++xds) {
                // Calculate max of downsampled y-row
                std::vector<float> dsData(y_num_ds, 0);
                std::vector<uint8_t> cnt(y_num_ds, 0);
                for (size_t z = zds*2; z <= std::min(zds*2+1,z_num-1); ++z) {
                    for (size_t x = xds*2; x <= std::min(xds*2+1,x_num-1); ++x) {
                        if (dr.globalIdxOfYrow.data[level][z * x_num + x].size() > 0) { // we have some y?
                            auto currentGlobalIdx = dr.globalIdxOfYrow.data[level][z * x_num + x][0];
                            auto &currentRow = dr.yRow.data[level][z * x_num + x];
                            for (auto y : currentRow) {
                                auto val = intensities.data[currentGlobalIdx];
                                dsData[y/2] += val;
                                cnt[y/2]++;
                                ++currentGlobalIdx;
                            }
                        }
                    }
                }

                // Update a tree
                if (drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds].size() > 0) { // we have some y?
                    auto currentGlobalIdx = drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds][0];
                    auto &currentRow = drTree.yRow.data[level-1][zds * x_num_ds + xds];
                    for (auto &y : currentRow) {
                        ouputTree.data[currentGlobalIdx] += dsData[y]; // / (childCnt[y] > 0 ? (1.0 * childCnt[y]) : 1.0);
                        childCnt.data[currentGlobalIdx] += cnt[y];
                        ++currentGlobalIdx;
                    }
                }
            }
        }
    }

    // propagate changes up to the lowest level in the tree
    for (int level = drTree.levelMax; level >= drTree.levelMin; --level) {
        const size_t x_num = aprIt.spatial_index_x_max(level);
        const size_t z_num = aprIt.spatial_index_z_max(level);
        const size_t x_num_ds = treeIterator.spatial_index_x_max(level - 1);
        const size_t y_num_ds = treeIterator.spatial_index_y_max(level - 1);
        const size_t z_num_ds = treeIterator.spatial_index_z_max(level - 1);


        for (size_t particleNumber = treeIterator.particles_level_begin(level);
             particleNumber < treeIterator.particles_level_end(level);
             ++particleNumber)
        {
            ouputTree.data[particleNumber] /= (1.0*childCnt.data[particleNumber]);
        }

        for (size_t zds = 0; zds < z_num_ds; ++zds) {
            for (size_t xds = 0; xds < x_num_ds; ++xds) {
                // Calculate max of downsampled y-row
                std::vector<float> dsData(y_num_ds, 0);
                std::vector<uint8_t> cnt(y_num_ds, 0);
                for (size_t z = zds*2; z <= std::min(zds*2+1,z_num-1); ++z) {
                    for (size_t x = xds*2; x <= std::min(xds*2+1,x_num-1); ++x) {
                        if (drTree.globalIdxOfYrow.data[level][z * x_num + x].size() > 0) { // we have some y?
                            auto currentGlobalIdx = drTree.globalIdxOfYrow.data[level][z * x_num + x][0];
                            auto &currentRow = drTree.yRow.data[level][z * x_num + x];
                            for (auto y : currentRow) {
                                auto val = ouputTree.data[currentGlobalIdx];
                                dsData[y/2] += val;
                                cnt[y/2]++;
                                ++currentGlobalIdx;
                            }
                        }
                    }
                }

                // Update a tree
                if (drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds].size() > 0) { // we have some y?
                    auto currentGlobalIdx = drTree.globalIdxOfYrow.data[level-1][zds * x_num_ds + xds][0];
                    auto &currentRow = drTree.yRow.data[level-1][zds * x_num_ds + xds];
                    for (auto &y : currentRow) {
                        auto val = dsData[y];
                        childCnt.data[currentGlobalIdx] += cnt[y];
                        ouputTree.data[currentGlobalIdx] += val;
                        ++currentGlobalIdx;
                    }
                }
            }
        }
    }
    return ouputTree;
}

template<typename T>
ExtraParticleData<float> maximumDownsamplingOld(APR<T> &aInputApr, APRTree<uint16_t> &aprTree) {
    APRIterator<uint16_t> aprIt(aInputApr);
    APRTreeIterator<uint16_t> treeIt(aprTree);
    APRTreeIterator<uint16_t> parentTreeIt(aprTree);
    ExtraParticleData<float> outputTree(aprTree);
    auto &intensities = aInputApr.particles_intensities;


    for (unsigned int level = aprIt.level_max(); level >= aprIt.level_min(); --level) {
        for (size_t particle_number = aprIt.particles_level_begin(level);
             particle_number < aprIt.particles_level_end(level);
             ++particle_number)
        {
            aprIt.set_iterator_to_particle_by_number(particle_number);
            treeIt.set_iterator_to_parent(aprIt);

            auto val = intensities[aprIt];
            if (val > outputTree[treeIt]) outputTree[treeIt] = val;  // compute maximum
        }
    }


    for (unsigned int level = treeIt.level_max(); level > treeIt.level_min(); --level) {
        for (size_t particleNumber = treeIt.particles_level_begin(level);
             particleNumber < treeIt.particles_level_end(level);
             ++particleNumber)
        {
            treeIt.set_iterator_to_particle_by_number(particleNumber);
            parentTreeIt.set_iterator_to_parent(treeIt);
            auto val = outputTree.data[treeIt.global_index()];
            if (val > outputTree[parentTreeIt]) outputTree[parentTreeIt] = val;  // compute maximum
        }
    }
    return outputTree;
}

template<typename T>
ExtraParticleData<float> meanDownsamplingOld(APR<T> &aInputApr, APRTree<uint16_t> &aprTree) {
    APRIterator<uint16_t> aprIt(aInputApr);
    APRTreeIterator<uint16_t> treeIt(aprTree);
    APRTreeIterator<uint16_t> parentTreeIt(aprTree);
    ExtraParticleData<float> outputTree(aprTree);
    ExtraParticleData<uint8_t> childCnt(aprTree);
    auto &intensities = aInputApr.particles_intensities;

    for (unsigned int level = aprIt.level_max(); level >= aprIt.level_min(); --level) {
        for (size_t particle_number = aprIt.particles_level_begin(level);
             particle_number < aprIt.particles_level_end(level);
             ++particle_number)
        {
            aprIt.set_iterator_to_particle_by_number(particle_number);
            parentTreeIt.set_iterator_to_parent(aprIt);

            auto val = intensities[aprIt];
            outputTree[parentTreeIt] += val;
            childCnt[parentTreeIt]++;
        }
    }

    //then do the rest of the tree where order matters (it goes to level_min since we need to eventually average data there).
    for (unsigned int level = treeIt.level_max(); level >= treeIt.level_min(); --level) {
        // average intensities first
        for (size_t particleNumber = treeIt.particles_level_begin(level);
             particleNumber < treeIt.particles_level_end(level);
             ++particleNumber)
        {
            treeIt.set_iterator_to_particle_by_number(particleNumber);
            outputTree[treeIt] /= (1.0*childCnt[treeIt]);
        }

        // push changes
        if (level > treeIt.level_min())
        for (uint64_t parentNumber = treeIt.particles_level_begin(level);
             parentNumber < treeIt.particles_level_end(level);
             ++parentNumber)
        {
            treeIt.set_iterator_to_particle_by_number(parentNumber);
            if (parentTreeIt.set_iterator_to_parent(treeIt)) {
                outputTree[parentTreeIt] += outputTree[treeIt];
                childCnt[parentTreeIt]++;
            }
        }
    }
    return outputTree;
}

template<typename T>
void compareTreeIntensities(APRTreeIterator<uint16_t>& treeIterator, ExtraParticleData<T> newData, ExtraParticleData<T> oldData) {
    int errCnt = 0;
    const int maxErrorPirintNum =10;
    if (oldData.data.size() != newData.data.size()) {
        std::cout << "ERROR: size of compared containers differ!" << std::endl;
    }
    for (unsigned int level = treeIterator.level_max(); level >= treeIterator.level_min(); --level) {
        for (size_t particle_number = treeIterator.particles_level_begin(level);
             particle_number < treeIterator.particles_level_end(level);
             ++particle_number) {
            treeIterator.set_iterator_to_particle_by_number(particle_number);
            if (newData.data[particle_number] != oldData.data[particle_number]) {
                if (errCnt < maxErrorPirintNum) std::cout << "ERROR: " << " idx: " << particle_number << " {" << newData.data[particle_number] << " vs old: " << oldData.data[particle_number] << "} on level: " << level << std::endl;
                errCnt++;
            }
        }
    }

    std::cout << "Number of errors: " << errCnt << ", sizeOfTree: " << newData.data.size() << std::endl;
}


int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);
    std::string fileName = options.directory + options.input;
    APR<uint16_t> apr;
    apr.read_apr(fileName);

    // Get dense representation of APR
    APRIterator<uint16_t> aprIt(apr);
    DenseRepresentation dr = getDenseRepresentation(aprIt);
    std::cout << "APR: " << dr << std::endl;

    // Get dense representation of APR tree
    APRTree<uint16_t> aprTree(apr);
    APRTreeIterator<uint16_t> treeIt(aprTree);
    DenseRepresentation drTree = getDenseRepresentation(treeIt);
    std::cout << "APR tree: " << drTree << std::endl;

    // Do maximum downsampling on DenseRepresentation and old representation
    std::cout << "\nMax Downsampling on new dense representation.\n";
    ExtraParticleData<float> newMaxDsTree = maximumDownsampling(apr, dr, drTree, aprTree);
    std::cout << "Max Downsampling on old representation.\n";
    ExtraParticleData<float> oldMaxDsTree = maximumDownsamplingOld(apr, aprTree);
    // Check if old and new way give same result
    std::cout << "Compare.\n";
    compareTreeIntensities(treeIt, newMaxDsTree, oldMaxDsTree);

    // Do mean downsampling on DenseRepresentation and old representation
    std::cout << "\nMean Downsampling on new dense representation.\n";
    ExtraParticleData<float> newMeanDsTree = meanDownsampling(apr, dr, drTree, aprTree);
    std::cout << "Mean Downsampling on old representation.\n";
    ExtraParticleData<float> oldMeanDsTree = meanDownsamplingOld(apr, aprTree);
    // Check if old and new way give same result
    std::cout << "Compare.\n";
    compareTreeIntensities(treeIt, newMeanDsTree, oldMeanDsTree);
}
