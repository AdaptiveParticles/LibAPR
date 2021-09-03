//
// Created by joel on 01.09.21.
//

#ifndef APR_LAZYACCESS_HPP
#define APR_LAZYACCESS_HPP

#include "GenAccess.hpp"
#include "io/APRFile.hpp"
#include "io/APRWriter.hpp"
#include "data_structures/APR/particles/LazyData.hpp"
#include "data_structures/Mesh/PixelData.hpp"


class LazyAccess : public GenAccess {

public:

    GenInfo aprInfo;

    VectorData<uint64_t> level_xz_vec; // the starting location of each level in the xz_end_vec structure
    VectorData<uint64_t> xz_end_vec; // total number of particles up to and including the current sparse row
    LazyData<uint16_t> y_vec;

    LazyAccess() = default;

    void init(APRFile& aprFile) {
        aprFile.read_metadata(aprInfo);
        genInfo = &aprInfo;

        initialize_xz_linear();

        uint64_t begin_index = 0;
        uint64_t end_index = level_xz_vec[genInfo->l_max+1];
        auto objectId = aprFile.get_fileStructure()->objectId;
        APRWriter::readData("xz_end_vec", objectId, xz_end_vec.data(), begin_index, end_index);

        y_vec.init_file(aprFile, "y_vec", true);
    }


    void init_tree(APRFile& aprFile) {
        aprFile.read_metadata_tree(aprInfo);
        genInfo = &aprInfo;

        initialize_xz_linear();

        uint64_t begin_index = 0;
        uint64_t end_index = level_xz_vec[genInfo->l_max+1];
        auto objectId = aprFile.get_fileStructure()->objectIdTree;
        APRWriter::readData("xz_end_vec", objectId, xz_end_vec.data(), begin_index, end_index);

        y_vec.init_file(aprFile, "y_vec", false);
    }


    void initialize_xz_linear(){
        uint64_t counter_total = 1; //the buffer val to allow -1 calls without checking.
        level_xz_vec.resize(level_max()+2,0); //includes a buffer for -1 calls, and therefore needs to be called with level + 1;
        level_xz_vec[0] = 1; //allowing for the offset.

        for (int i = 0; i <= level_max(); ++i) {
            counter_total += x_num(i)*z_num(i);
            level_xz_vec[i+1] = counter_total;
        }
        xz_end_vec.resize(counter_total,0);
    }

    void open() { y_vec.open(); }
    void close() { y_vec.close(); }

};

#endif //APR_LAZYACCESS_HPP
