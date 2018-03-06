#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"
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
ExtraParticleData<float> doWork(APR<T> &aInputApr, const DenseRepresentation &dr, const DenseRepresentation &drTree, APRTree<uint16_t> &aprTree) {
    APRIterator<uint16_t> aprIt(aInputApr);
    const auto &intensities = aInputApr.particles_intensities;
    ExtraParticleData<float> partData(aprTree);

    for (int level = dr.levelMax; level >= dr.levelMin; --level) {
        const size_t x_num = aprIt.spatial_index_x_max(level);
        const size_t x_num_ds = aprIt.spatial_index_x_max(level - 1);
        const size_t y_num_ds = aprIt.spatial_index_y_max(level - 1);
        const size_t z_num_ds = aprIt.spatial_index_z_max(level - 1);

        for (size_t zds = 0; zds < z_num_ds; ++zds) {
            for (size_t xds = 0; xds < x_num_ds; ++xds) {

                // Calculate max of downsampled y-row
                std::vector<float> dsData(y_num_ds, 0);
                for (size_t z = zds*2; z < zds*2+1; ++z) {
                    for (size_t x = xds*2; x < xds*2+1; ++x) {
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

    for (int level = drTree.levelMax; level > drTree.levelMin; --level) {
        const size_t x_num = aprIt.spatial_index_x_max(level);
        const size_t x_num_ds = aprIt.spatial_index_x_max(level - 1);
        const size_t y_num_ds = aprIt.spatial_index_y_max(level - 1);
        const size_t z_num_ds = aprIt.spatial_index_z_max(level - 1);

        for (size_t zds = 0; zds < z_num_ds; ++zds) {
            for (size_t xds = 0; xds < x_num_ds; ++xds) {

                // Calculate max of downsampled y-row
                std::vector<float> dsData(y_num_ds, 0);
                for (size_t z = zds*2; z < zds*2+1; ++z) {
                    for (size_t x = xds*2; x < xds*2+1; ++x) {
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
                        partData.data[currentGlobalIdx] = dsData[y];
                        ++currentGlobalIdx;
                    }
                }
            }
        }
    }
    return partData;
}

template<typename T>
ExtraParticleData<float> doWorkOld(APR<T> &aInputApr, APRTree<uint16_t> &aprTree) {
    APRIterator<uint16_t> apr_iterator(aInputApr);
    APRTreeIterator<uint16_t> treeIterator(aprTree);
    ExtraParticleData<float> tree_intensity(aprTree);
    ExtraParticleData<uint8_t> tree_counter(aprTree);

    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        for (size_t particle_number = apr_iterator.particles_level_begin(level);
             particle_number < apr_iterator.particles_level_end(level);
             ++particle_number)
        {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            treeIterator.set_iterator_to_parent(apr_iterator);
            tree_intensity[treeIterator]= (tree_intensity[treeIterator]*tree_counter[treeIterator]*1.0f + aInputApr.particles_intensities[apr_iterator])/(1.0f*(tree_counter[treeIterator]+1.0f));
            tree_counter[treeIterator]++;
        }
    }

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

    // Do some op on DenseRepresentation
    doWork(apr, dr, drTree, aprTree);
    // Do some op on old representation
    doWorkOld(apr, aprTree);

}
