//
// Created by cheesema on 2019-05-20.
//

#ifndef LIBAPR_APRFILE_HPP
#define LIBAPR_APRFILE_HPP

#include "hdf5functions_blosc.h"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/RandomAccess.hpp"
#include "ConfigAPR.h"
#include <numeric>
#include <memory>
#include <data_structures/APR/APR.hpp>
#include "numerics/APRCompress.hpp"
#include "APRWriter.hpp"
#include "misc/APRTimer.hpp"

class APRFile {

public:

    APRFile(){

    }

    // setup
    bool open(std::string file_name,std::string read_write = "WRITE");
    void close();

    // write
    void write_apr(APR &apr,uint64_t t = 0,std::string channel_name = "t");
    void write_apr_append(APR &apr);
    template<typename DataType>
    void write_particles(APR &apr,std::string particles_name,ParticleData<DataType>& particles,uint64_t t = 0,bool apr_or_tree = true,std::string channel_name = "t");

    // read
    void read_apr(APR &apr,uint64_t t = 0,std::string channel_name = "t");
    template<typename DataType>
    void read_particles(APR apr,std::string particles_name,ParticleData<DataType>& particles,uint64_t t = 0,bool apr_or_tree = true,std::string channel_name = "t");

    //set helpers
    bool get_read_write_tree(){
        return with_tree_flag;
    }

    //set helpers
    /**
   * Set whether the internal APR Tree internal access should also be written and read.
   * @param write_with_tree_flag_ indicate whether the APRTree should be written and read. (True = save both APR and APR Tree)
   */
    void set_read_write_tree(bool with_tree_flag_){
        with_tree_flag = with_tree_flag_;
    }

    std::vector<std::string> get_channel_names();

    //get helpers
    uint64_t get_number_time_steps(std::string channel_name = "t");
    std::vector<std::string> get_particles_names(uint64_t t,bool apr_or_tree = true,std::string channel_name = "t");

    APRTimer timer;

    float current_file_size(){
        return fileStructure.getFileSize();
    }

    void set_blosc_access_settings(unsigned int blosc_comp_type_access_,unsigned int blosc_comp_level_access_,unsigned int blosc_shuffle_access_){
        blosc_comp_type_access = blosc_comp_type_access_;
        blosc_comp_level_access = blosc_comp_level_access_;
        blosc_shuffle_access= blosc_shuffle_access_;
    }

    void set_blosc_parts_settings(unsigned int blosc_comp_type_parts_,unsigned int blosc_comp_level_parts_,unsigned int blosc_shuffle_parts_){
        blosc_comp_type_parts = blosc_comp_type_parts_;
        blosc_comp_level_parts = blosc_comp_level_parts_;
        blosc_shuffle_parts= blosc_shuffle_parts_;
    }

    APRCompress aprCompress;
private:

    //Basic Properties.
    uint64_t current_t=0;
    bool with_tree_flag = true;
    std::string file_name = "noname";
    APRWriter::FileStructure fileStructure;

    //HDF5 - BLOSC parameters
    unsigned int blosc_comp_type_parts = BLOSC_ZSTD;
    unsigned int blosc_comp_level_parts = 2;
    unsigned int blosc_shuffle_parts=1;

    unsigned int blosc_comp_type_access = BLOSC_ZSTD;
    unsigned int blosc_comp_level_access = 2;
    unsigned int blosc_shuffle_access=1;

    //Advanced Parameters.

    //Maximum and minimum levels to read.
//    unsigned int maximum_level_read = 0; //have a set method either by level or by delta
//    int max_level_delta = 0; #TODO: add this functionality back in for lazy loading particles..

};

/**
   * Open the file, creates it if it doesn't exist.
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   * @return (Bool) has the opening of the file been successful or not.
   */
bool APRFile::open(std::string file_name_,std::string read_write_append){


    if(read_write_append == "WRITE"){

        fileStructure.init(file_name_, APRWriter::FileStructure::Operation::WRITE);

    } else if(read_write_append == "READ") {

        fileStructure.init(file_name_, APRWriter::FileStructure::Operation::READ);

    } else if(read_write_append == "READWRITE") {

        fileStructure.init(file_name_, APRWriter::FileStructure::Operation::WRITE_APPEND);

    } else {
        std::cerr << "Files should either be opened as READ or WRITE, or READWRITE" << std::endl;
    }

    file_name = file_name_;

    return true;

}

/**
   * Close the HDF5 file structures
   */
void APRFile::close(){
    fileStructure.close();
}


/**
   * Write the APR to file (note this does not include particles)
   * @param APR to be written
   * @param t the time point to be written (default will be to append to the end of the file, starting with 0)
   */
void APRFile::write_apr(APR &apr,uint64_t t,std::string channel_name){

    APRTimer timer_f(false);

    current_t = t;

    if(fileStructure.isOpened()){
        //std::cout << "file is open" << std::endl;
    }

    fileStructure.create_time_point(t,with_tree_flag,channel_name);

    hid_t meta_location = fileStructure.objectId;

    //global property
    APRWriter::writeAttr(AprTypes::TimeStepType, meta_location, &t);

    // ------------- write metadata -------------------------
    //per time step
    APRWriter::writeString(AprTypes::NameType,meta_location, (apr.name.size() == 0) ? "no_name" : apr.name);
    APRWriter::writeString(AprTypes::GitType, meta_location, ConfigAPR::APR_GIT_HASH);

    APRWriter::write_apr_info(meta_location,apr.aprInfo);

    APRWriter::write_apr_parameters(meta_location,apr.parameters);

    timer.start_timer("access_data");

    APRWriter::write_random_access(meta_location,fileStructure.objectId, apr.apr_access,blosc_comp_type_access, blosc_comp_level_access,blosc_shuffle_access);

    timer.stop_timer();



    if(with_tree_flag){

        apr.init_tree(); //incase it hasn't been initialized.

        APRWriter::write_random_access(fileStructure.objectIdTree,fileStructure.objectIdTree, apr.tree_access,blosc_comp_type_access, blosc_comp_level_access,blosc_shuffle_access);

    }

}

/**
   * Write the APR to file and append it as the next time point (note this does not include particles)
   * @param APR to be written
   */
void APRFile::write_apr_append(APR &apr){
    write_apr(apr,current_t + 1);
}

/**
   * Write particles to file, they will be associated with a given time point, and either as APR particles of APRTree particles
   * @param particles_name string name of the particles dataset to be written (is then to be used for reading, each time point can have a similmarly named dataset)
   * @paramt particles particle dataset to be written (contiguos block of memory)
   * @param t (uint64_t) the time point to be written. (DEFAULT: t = 0)
   * @param apr_or_tree (Default = true (APR), false = APR Tree)
   */
template<typename DataType>
void APRFile::write_particles(APR &apr,const std::string particles_name,ParticleData<DataType>& particles,uint64_t t,bool apr_or_tree,std::string channel_name){

    fileStructure.open_time_point(t,with_tree_flag,channel_name);

    APRTimer timer;

    hid_t part_location;

    if(apr_or_tree){
        part_location = fileStructure.objectId;
    } else {
        part_location = fileStructure.objectIdTree;
    }

    // ------------- write data ----------------------------

    timer.start_timer("intensities");
    if (aprCompress.get_compression_type() > 0){
        aprCompress.compress(apr,particles);
    }

    int compress_type_num = aprCompress.get_compression_type();
    APRWriter::writeAttr(AprTypes::CompressionType, part_location, &compress_type_num);
    float quantization_factor = aprCompress.get_quantization_factor();
    APRWriter::writeAttr(AprTypes::QuantizationFactorType, part_location, &quantization_factor);

    hid_t type = APRWriter::Hdf5Type<DataType>::type();
    APRWriter::writeData({type, particles_name.c_str()}, part_location, particles.data, blosc_comp_type_parts, blosc_comp_level_parts, blosc_shuffle_parts);
    timer.stop_timer();

}

/**
   * Read the an APR from file (note this does not include particles)
   * @param APR to be read to
   * @param t the time point to be written (default will be to append to the end of the file, starting with 0)
   */
void APRFile::read_apr(APR &apr,uint64_t t,std::string channel_name){

    APRTimer timer_f(false);

    if(fileStructure.isOpened()){
        //std::cout << "file is open" << std::endl;
    }

    bool tree_exists = fileStructure.open_time_point(t,with_tree_flag,channel_name);

    hid_t meta_data = fileStructure.groupId;

    //check if old or new file, for location of the properties. (The metadata moved to the time point.)
    bool old = attribute_exists(fileStructure.objectId,AprTypes::MaxLevelType.typeName);

    if(old) {
        meta_data = fileStructure.objectId;
    }

    if (!fileStructure.isOpened()) return;

    // ------------- read metadata --------------------------
    char string_out[100] = {0};
    hid_t attr_id = H5Aopen(meta_data,"name",H5P_DEFAULT);
    hid_t atype = H5Aget_type(attr_id);
    hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);
    H5Aread(attr_id, atype_mem, string_out) ;
    H5Aclose(attr_id);
    apr.name= string_out;

    //read in access information
    APRWriter::read_access_info(meta_data,apr.aprInfo);
    apr.linearAccess.genInfo = &apr.aprInfo;
    apr.apr_access.genInfo = &apr.aprInfo;

    //read in pipeline parameters
    APRWriter::read_apr_parameters(meta_data,apr.parameters);

    APRWriter::read_random_access(meta_data,fileStructure.objectId, apr.apr_access);


    if(with_tree_flag) {

        if(!tree_exists){
            //initializing it from the dataset.
            std::cout << "Initializing tree from file" << std::endl;
            apr.init_tree();
        } else {

            timer.start_timer("build tree - map");

            apr.treeInfo.init_tree(apr.org_dims(0),apr.org_dims(1),apr.org_dims(2));
            apr.tree_access.genInfo = &apr.treeInfo;

            APRWriter::readAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree,
                                &apr.treeInfo.total_number_particles);

            APRWriter::read_random_tree_access(meta_data,fileStructure.objectIdTree, apr.tree_access,apr.apr_access);

        }

    }


};

/**
   * Read particles from file, they will be associated with a given time point, and either as APR particles of APRTree particles (see get_particles_names for saved particle datasets)
   * @param particles_name string name of the particles dataset to be read (is then to be used for reading, each time point can have a similmarly named dataset)
   * @paramt particles particle dataset to be read (contiguos block of memory)
   * @param t (uint64_t) the time point to be read from. (DEFAULT: t = 0)
   * @param apr_or_tree (Default = True (APR), false = APR Tree)
   */
template<typename DataType>
void APRFile::read_particles(APR apr,std::string particles_name,ParticleData<DataType>& particles,uint64_t t,bool apr_or_tree,std::string channel_name){

    if(fileStructure.isOpened()){
        //std::cout << "file is open" << std::endl;
    }

    fileStructure.open_time_point(t,with_tree_flag,channel_name);

    //uint64_t max_read_level = apr.apr_access.level_max()-max_level_delta;
    //uint64_t max_read_level_tree = std::min(apr.apr_access.level_max()-1,max_read_level);
    //uint64_t prev_read_level = 0;

    uint64_t parts_start = 0;
    uint64_t parts_end = apr.total_number_particles(); //apr.apr_access.global_index_by_level_end[max_read_level] + 1;

    if(!apr_or_tree){
        parts_end = apr.total_number_tree_particles();
    }

    //check if old or new file, for location of the properties. (The metadata moved to the time point.)
    hid_t part_location;

    if(apr_or_tree){
        part_location = fileStructure.objectId;
    } else {
        part_location = fileStructure.objectIdTree;
    }

    //backwards support
    bool new_parts = attribute_exists(part_location,AprTypes::CompressionType.typeName);

    hid_t meta_data = part_location;

    if(!new_parts) {
        meta_data = fileStructure.groupId;
    }


    int compress_type;
    APRWriter::readAttr(AprTypes::CompressionType, meta_data, &compress_type);
    float quantization_factor;
    APRWriter::readAttr(AprTypes::QuantizationFactorType, meta_data, &quantization_factor);


    timer.start_timer("Read intensities");
    // ------------- read data ------------------------------
    particles.data.resize(parts_end);
    if (particles.data.size() > 0) {
        APRWriter::readData(particles_name.c_str(), part_location, particles.data.data() + parts_start,parts_start,parts_end);
    }

    timer.stop_timer();

    timer.start_timer("decompress");
    // ------------ decompress if needed ---------------------
    if (compress_type > 0) {
        aprCompress.set_compression_type(compress_type);
        aprCompress.set_quantization_factor(quantization_factor);
        aprCompress.decompress(apr, particles,parts_start);
    }
    timer.stop_timer();

};



//get helpers

/**
   * Number of time steps saved in the file
   *   @param Channel name (default = t)
    * @return Number of time steps.
   */
uint64_t APRFile::get_number_time_steps(std::string channel_name){

    const int max_name_size = 1024;

    ssize_t len;
    hsize_t nobj;
    herr_t err;
    int otype;

    hid_t obj_name = fileStructure.groupId;

    char group_name[max_name_size];
    char memb_name[max_name_size];

    len = H5Iget_name(obj_name, group_name, max_name_size  );

    err = H5Gget_num_objs(obj_name, &nobj);

    uint64_t counter_t = 0;

    for (int i = 0; i < nobj; i++) {

        len = H5Gget_objname_by_idx(obj_name, (hsize_t) i,
                                    memb_name, (size_t) max_name_size);

        otype = H5Gget_objtype_by_idx(obj_name, (size_t) i);

        std::string w = memb_name;

        if(otype == H5G_GROUP){

            int ch_l = channel_name.size();
            std::string subs = w.substr(0,ch_l);

            if(subs == channel_name){
                counter_t++;
            }
        }

    }
    return counter_t;

};

/**
   * Number of time steps saved in the file
    * @return Channel names
   */
std::vector<std::string> APRFile::get_channel_names(){
    const int max_name_size = 1024;

    ssize_t len;
    hsize_t nobj;
    herr_t err;
    int otype;

    hid_t obj_name = fileStructure.groupId;

    char group_name[max_name_size];
    char memb_name[max_name_size];

    len = H5Iget_name(obj_name, group_name, max_name_size  );

    err = H5Gget_num_objs(obj_name, &nobj);

    std::vector<std::string> channel_names;

    for (int i = 0; i < nobj; i++) {

        len = H5Gget_objname_by_idx(obj_name, (hsize_t) i,
                                    memb_name, (size_t) max_name_size);

        otype = H5Gget_objtype_by_idx(obj_name, (size_t) i);

        if(otype == H5G_GROUP){
            channel_names.push_back(memb_name);
        }

    }
    return channel_names;

};


/**
   * Gets the names of particles datasets saved for a particular file for a time step, and either for APR, or APR Tree.
   * @param t the time step the particle datasets have been saved to.
   * @param apr_or_tree Is it an APR or APR Tree dataset. (Defualt = true (APR), flase = (APR Tree))
   * @return vector of strings of the names of the datasets (can be then used with read_particles).
   */
std::vector<std::string> APRFile::get_particles_names(uint64_t t,bool apr_or_tree,std::string channel_name){

    fileStructure.open_time_point(t,with_tree_flag,channel_name);

    const int max_name_size = 1024;

    ssize_t len;
    hsize_t nobj;
    herr_t err;
    int otype;

    std::vector<std::string> access_names = {"map_level","map_global_index","map_number_gaps","map_x","map_y_begin","map_y_end","map_z"};

    hid_t obj_name;

    if(apr_or_tree){
        obj_name = fileStructure.objectId;
    } else{
        obj_name = fileStructure.objectIdTree;
    }

    char group_name[max_name_size];
    char memb_name[max_name_size];

    len = H5Iget_name(obj_name, group_name, max_name_size  );

    err = H5Gget_num_objs(obj_name, &nobj);

    std::vector<std::string> dataset_names;

    for (int i = 0; i < nobj; i++) {

        len = H5Gget_objname_by_idx(obj_name, (hsize_t)i,
                                    memb_name, (size_t)max_name_size );

        otype =  H5Gget_objtype_by_idx(obj_name, (size_t)i );

        if(otype == H5G_DATASET){
            bool access = false;

            for (int j = 0; j < access_names.size(); ++j) {
                if(memb_name == access_names[j]){
                    access = true;
                }
            }
            if(!access) {
                dataset_names.push_back(memb_name);
            }
        }
    }

    return dataset_names;

};



#endif //LIBAPR_APRFILE_HPP
