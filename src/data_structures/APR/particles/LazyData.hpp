//
// Created by cheesema on 2019-06-15.
//

#ifndef LIBAPR_LAZYDATA_HPP
#define LIBAPR_LAZYDATA_HPP

#include <vector>
#include "io/APRFile.hpp"
#include "numerics/APRCompress.hpp"

template<typename DataType>
class LazyData {

    uint64_t current_offset;

    std::vector<DataType> data;

    APRWriter::FileStructure* fileStructure;
    std::string parts_name;
    bool apr_or_tree;
    uint64_t parts_start;
    uint64_t parts_end;
    hid_t group_id;

public:

    APRCompress compressor;

    void init_file(APRFile& parts_file,std::string name,bool apr_or_tree_){

        parts_name = name;
        apr_or_tree = apr_or_tree_;
        fileStructure = parts_file.get_fileStructure();
        fileStructure->create_time_point(0,apr_or_tree_,"t");

        if(apr_or_tree) {
            group_id = fileStructure->objectId;
        } else {
            group_id = fileStructure->objectIdTree;
        }

        set_hdf5_cache();

        dataSet.init(group_id,parts_name.c_str());

    }

    void open(){
        dataSet.open();
    }

    void create_file(uint64_t size){

        hid_t type = APRWriter::Hdf5Type<DataType>::type();

        dataSet.create(type,size);

    }

    void load_row(int level,int z,int x,LinearIterator& it){

        it.begin(level,z,x);
        parts_start = it.begin_index;
        parts_end = it.end();

        if ((parts_end - parts_start) > 0) {
            data.resize(parts_end - parts_start);

            read_data(data.data(),parts_start,parts_end);

        }

        // ------------ decompress if needed ---------------------
        if (this->compressor.get_compression_type() > 0) {
            this->compressor.decompress( data,parts_start);
        }
    }

    void load_slice(int level,int z,LinearIterator& it){

        it.begin(level,z,0); //begining of slice
        parts_start = it.begin_index;
        it.begin(level,z,it.x_num(level)-1); //to end of slice
        parts_end = it.end();

        if ((parts_end - parts_start) > 0) {
            data.resize(parts_end - parts_start);
            read_data(data.data(),parts_start,parts_end);

        }


        // ------------ decompress if needed ---------------------
        if (this->compressor.get_compression_type() > 0) {
            this->compressor.decompress( data,parts_start);
        }
    }

    void write_slice(int level,int z,LinearIterator& it){

        it.begin(level,z,0); //begining of slice
        parts_start = it.begin_index;
        it.begin(level,z,it.x_num(level)-1); //to end of slice
        parts_end = it.end();

        if ((parts_end - parts_start) > 0) {
            //compress if needed
            if (this->compressor.get_compression_type() > 0){
                this->compressor.compress(data);
            }

            write_data(data.data(),parts_start,parts_end);

        }

    }


    inline DataType& operator[](const LinearIterator& it) {
        return data[it.current_index - parts_start];
    }

    uint64_t dataset_size(){

        std::vector<uint64_t> dims = dataSet.get_dimensions();

        return dims[0];

    }

    LazyData(){};

    void close(){
        dataSet.close();

    }

private:

    Hdf5DataSet dataSet;


    void set_hdf5_cache(){

        int mdc_nelmts;
        size_t rdcc_nelmts;
        size_t rdcc_nbytes;
        double rdcc_w0;
        hid_t  fapl_idChunked;
        herr_t status;

        fapl_idChunked = H5Pcreate(H5P_FILE_ACCESS);

        status = H5Pget_cache (fapl_idChunked, &mdc_nelmts, &rdcc_nelmts,
                               &rdcc_nbytes, &rdcc_w0);

        rdcc_nbytes = 1000000*400;
        rdcc_nelmts = 1000000;

        status = H5Pset_cache (fapl_idChunked, mdc_nelmts, rdcc_nelmts,
                               rdcc_nbytes, rdcc_w0);
    }


    void read_data( void* buff,uint64_t elements_start,uint64_t elements_end) {

        dataSet.read(buff, elements_start, elements_end);

    }

    void write_data( void* buff,uint64_t elements_start,uint64_t elements_end) {

        dataSet.write(buff,elements_start,elements_end);

    }

};


#endif //LIBAPR_LAZYDATA_HPP
