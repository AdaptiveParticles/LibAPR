//
// Created by cheesema on 2019-06-15.
//

#ifndef LIBAPR_LAZYDATA_HPP
#define LIBAPR_LAZYDATA_HPP

#include <vector>

#include "APR.hpp"

#include "APRAccessStructures.hpp"

#include "GenData.hpp"

#include "io/APRFile.hpp"

template<typename DataType>
class LazyData: public GenData<DataType>  {

    uint64_t current_offset;

    std::vector<DataType> data;

    APRFile * aprFile;
    std::string parts_name;
    bool apr_or_tree;
    uint64_t parts_start;
    uint64_t parts_end;


    void init(APRFile parts_file,std::string name,bool apr_or_tree_){
        parts_name = name;
        apr_or_tree = apr_or_tree_;
        aprFile = &parts_file;
    }

    void get_row(int level,int z,int x,LinearIterator& it){

        it.begin(level,z,x);
        parts_start = it.begin_index;
        parts_end = it.end();

        data.resize(parts_end - parts_start);

        if (data.size() > 0) {
            APRWriter::readData(parts_name, aprFile., data.data() + parts_start,parts_start,parts_end);
        }



//        // ------------ decompress if needed ---------------------
//        if (compress_type > 0) {
//            aprCompress.set_compression_type(compress_type);
//            aprCompress.set_quantization_factor(quantization_factor);
//            aprCompress.decompress(apr, particles,parts_start);
//        }
    }


    DataType& operator[](LinearIterator& it) override {
        return data[it.current_index - it.begin_index];
    }




};


#endif //LIBAPR_LAZYDATA_HPP
