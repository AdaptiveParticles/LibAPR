//
// Created by cheesema on 2019-05-20.
//

#ifndef LIBAPR_APRFILE_HPP
#define LIBAPR_APRFILE_HPP

#include "hdf5functions_blosc.h"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/access/RandomAccess.hpp"
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
    void write_apr(APR& apr,uint64_t t = 0,std::string channel_name = "t");
    void write_apr_append(APR& apr);
    template<typename DataType>
    void write_particles(std::string particles_name,ParticleData<DataType>& particles,bool apr_or_tree = true,uint64_t t = 0,std::string channel_name = "t");

    // read
    void read_apr(APR& apr,uint64_t t = 0,std::string channel_name = "t");
    template<typename DataType>
    void read_particles(APR& apr,std::string particles_name,ParticleData<DataType>& particles,bool apr_or_tree = true,uint64_t t = 0,std::string channel_name = "t");

    //set helpers
    bool get_read_write_tree(){
        return with_tree_flag;
    }

    //set helperslts
    /**
   * Set whether the internal APR Tree internal access should also be written and read.
   * @param write_with_tree_flag_ indicate whether the APRTree should be written and read. (True = save both APR and APR Tree)
   */
    void set_read_write_tree(bool with_tree_flag_){
        with_tree_flag = with_tree_flag_;
    }

    void set_write_linear_flag(bool flag_){
        write_linear = flag_;
    }

    std::vector<std::string> get_channel_names();

    //get helpers
    uint64_t get_number_time_steps(std::string channel_name = "t");
    std::vector<std::string> get_particles_names(bool apr_or_tree = true,uint64_t t=0,std::string channel_name = "t");

    APRTimer timer;

    float current_file_size_MB(){
        return (fileStructure.getFileSize())/(1000000.0);
    }

    float current_file_size_GB(){
        return (fileStructure.getFileSize())/(1000000000.0);
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

    /*
     *  Advanced IO capability
     *
     */

    //sets an offset as to the maximum level read.
    void set_max_level_read_delta(int max_level_delta_){

        if(max_level_delta_ < 0){
            std::cerr << "Max level delta must be positive" << std::endl;
        }

        max_level_delta = max_level_delta_;
    }

    APRWriter::FileStructure* get_fileStructure(){
        return &fileStructure;
    }

private:

    //Basic Properties.
    uint64_t current_t=0;
    bool with_tree_flag = true;
    std::string file_name = "noname";
    APRWriter::FileStructure fileStructure;

    bool write_linear = true;
    bool write_linear_tree = true; //this is not exposed. Just leaving this here if it could be useful

    //HDF5 - BLOSC parameters
    unsigned int blosc_comp_type_parts = BLOSC_ZSTD;
    unsigned int blosc_comp_level_parts = 4;
    unsigned int blosc_shuffle_parts=1;

    unsigned int blosc_comp_type_access = BLOSC_ZSTD;
    unsigned int blosc_comp_level_access = 4;
    unsigned int blosc_shuffle_access=1;

    //Advanced Parameters.
    int max_level_delta = 0;

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
    } else {
        std::cerr << "File is not open!" << std::endl;
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

    timer.start_timer("write_apr_access_data");

    if(write_linear){

        apr.init_linear();
        APRWriter::write_linear_access(meta_location, fileStructure.objectId, apr.linearAccess, blosc_comp_type_access,
                            blosc_comp_level_access, blosc_shuffle_access);

    } else {

        apr.initialize_random_access(); //check that it is initialized.

        APRWriter::write_random_access(meta_location, fileStructure.objectId, apr.apr_access, blosc_comp_type_access,
                                       blosc_comp_level_access, blosc_shuffle_access);
    }

    timer.stop_timer();


    if(with_tree_flag){

        if(write_linear_tree) {

            timer.start_timer("init_tree");
            apr.init_tree_linear();
            timer.stop_timer();

            timer.start_timer("write_tree_access_data");
            APRWriter::writeAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree,
                                 &apr.treeInfo.total_number_particles);

            APRWriter::write_linear_access(fileStructure.objectIdTree, fileStructure.objectIdTree, apr.linearAccessTree, blosc_comp_type_access,
                                           blosc_comp_level_access, blosc_shuffle_access);
            timer.stop_timer();
        } else {

            apr.init_tree_random(); //incase it hasn't been initialized.

            APRWriter::writeAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree,
                                 &apr.treeInfo.total_number_particles);

            APRWriter::write_random_access(fileStructure.objectIdTree, fileStructure.objectIdTree, apr.tree_access,
                                           blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

        }

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
void APRFile::write_particles(const std::string particles_name,ParticleData<DataType>& particles,bool apr_or_tree,uint64_t t,std::string channel_name){

    if(fileStructure.isOpened()){
    } else {
        std::cerr << "File is not open!" << std::endl;
    }

    fileStructure.open_time_point(t,with_tree_flag,channel_name);

    hid_t part_location;

    if(apr_or_tree){
        part_location = fileStructure.objectId;
    } else {
        part_location = fileStructure.objectIdTree;
    }

    // This is for backwards compatability indicating that these particles are written in contiguous fasion in lzxy order
    int lzxy_order = 1;
    APRWriter::writeAttr({H5T_NATIVE_INT,"lzxy_order"}, part_location, &lzxy_order);

    // ------------- write data ----------------------------

    timer.start_timer("write intensities");

    if (particles.compressor.get_compression_type() > 0){
        particles.compressor.compress(particles.data);
    }

    int compress_type_num = particles.compressor.get_compression_type();
    APRWriter::writeAttr(AprTypes::CompressionType, part_location, &compress_type_num);
    float quantization_factor = particles.compressor.get_quantization_factor();
    APRWriter::writeAttr(AprTypes::QuantizationFactorType, part_location, &quantization_factor);
    float compress_background = particles.compressor.get_background();
    APRWriter::writeAttr(AprTypes::CompressBackgroundType, part_location, &compress_background);

    hid_t type = APRWriter::Hdf5Type<DataType>::type();

    // Using this version allows for extension including sequential write for the compress. #TODO.
    Hdf5DataSet partsData;
    partsData.init(part_location,particles_name.c_str());
    partsData.create(type,particles.size());
    partsData.open();
    partsData.write(particles.data.data(),0,particles.size());
    partsData.close();


    timer.stop_timer();

}

/**
   * Read the an APR from file (note this does not include particles)
   * @param APR to be read to
   * @param t the time point to be written (default will be to append to the end of the file, starting with 0)
   */
void APRFile::read_apr(APR &apr,uint64_t t,std::string channel_name){


    if(fileStructure.isOpened()){
    } else {
        std::cerr << "File is not open!" << std::endl; //#TODO: should check if it is readable, and other functions if writeable ect.
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

    std::string data_n = fileStructure.subGroup1 + "/map_level";
    bool stored_random = data_exists(fileStructure.fileId,data_n.c_str());

    timer.start_timer("read_apr_access");

    if(!stored_random) {

        //make this an automatic check to see what the file is.
        APRWriter::read_linear_access( fileStructure.objectId, apr.linearAccess,max_level_delta);
        apr.apr_initialized = true;
    } else {

        APRWriter::read_random_access(meta_data, fileStructure.objectId, apr.apr_access);
        apr.apr_initialized_random = true;
    }

    timer.stop_timer();

    timer.start_timer("read_tree_access");

    if(with_tree_flag) {

        data_n = fileStructure.subGroupTree1 + "/map_level";
        bool stored_random_tree = data_exists(fileStructure.fileId,data_n.c_str());

        data_n = fileStructure.subGroupTree1 + "/y_vec";
        bool stored_linear_tree = data_exists(fileStructure.fileId,data_n.c_str());;

        tree_exists = true;
        if(!stored_random_tree && !stored_linear_tree){
            tree_exists = false;
        }

        if(!tree_exists){
            //initializing it from the dataset.
            std::cout << "Initializing tree from file" << std::endl;
            apr.init_tree_linear();
        } else {

            APRWriter::readAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree,
                                &apr.treeInfo.total_number_particles);

            apr.treeInfo.init_tree(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2));
            apr.tree_access.genInfo = &apr.treeInfo;
            apr.linearAccessTree.genInfo = &apr.treeInfo;

            if(!stored_linear_tree) {
                //Older data structures are saved.
                APRWriter::read_random_tree_access(fileStructure.objectIdTree, fileStructure.objectIdTree,
                                                   apr.tree_access, apr.apr_access);
                apr.tree_initialized_random = true;

            } else {
                apr.tree_initialized = true;
                int max_level_delta_tree=0;

                if(max_level_delta > 0){
                    max_level_delta_tree = max_level_delta - 1;
                }
                APRWriter::read_linear_access( fileStructure.objectIdTree, apr.linearAccessTree,max_level_delta_tree);
            }

        }

    }

    timer.stop_timer();


};

/**
   * Read particles from file, they will be associated with a given time point, and either as APR particles of APRTree particles (see get_particles_names for saved particle datasets)
   * @param apr requires a valid APR, this is for more general functionality including partial reading by level.
   * @param particles_name string name of the particles dataset to be read (is then to be used for reading, each time point can have a similmarly named dataset)
   * @paramt particles particle dataset to be read (contiguos block of memory)
   * @param t (uint64_t) the time point to be read from. (DEFAULT: t = 0)
   * @param apr_or_tree (Default = True (APR), false = APR Tree)
   */
template<typename DataType>
void APRFile::read_particles(APR &apr,std::string particles_name,ParticleData<DataType>& particles,bool apr_or_tree,uint64_t t,std::string channel_name){

    if(fileStructure.isOpened()){
    } else {
        std::cerr << "File is not open!" << std::endl;
    }

    fileStructure.open_time_point(t,with_tree_flag,channel_name);

    int max_read_level;
    uint64_t parts_start = 0;
    uint64_t parts_end = 0;

    if(apr_or_tree){
        max_read_level = std::max((int)apr.level_min(),(int)(apr.level_max() - max_level_delta));
        parts_start = 0;
        auto it = apr.iterator();
        parts_end = it.total_number_particles(max_read_level);

    } else {
        auto tree_it = apr.tree_iterator();
        max_read_level = std::min((int)tree_it.level_max(),(int)(apr.level_max() - max_level_delta));
        max_read_level = std::max((int)tree_it.level_min(),max_read_level);
        parts_start = 0;
        auto it_tree = apr.tree_iterator();
        parts_end = it_tree.total_number_particles(max_read_level);
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

    /*
     *  Particle Compression Options
     */

    int compress_type;
    APRWriter::readAttr(AprTypes::CompressionType, meta_data, &compress_type);
    float quantization_factor;
    APRWriter::readAttr(AprTypes::QuantizationFactorType, meta_data, &quantization_factor);
    float compress_background;

    //backwards support
    bool background_exists = attribute_exists(part_location,AprTypes::CompressBackgroundType.typeName);

    if((compress_type > 0) && (background_exists)){
        APRWriter::readAttr(AprTypes::CompressBackgroundType, meta_data, &compress_background);
    } else {
        compress_background = apr.parameters.background_intensity_estimate - apr.parameters.noise_sd_estimate;
    }

    timer.start_timer("Read intensities");



    // ------------- read data ------------------------------
    particles.data.resize(parts_end - parts_start);
    if (particles.data.size() > 0) {
        APRWriter::readData(particles_name.c_str(), part_location, particles.data.data() + parts_start,parts_start,parts_end);
    }

    timer.stop_timer();

    timer.start_timer("decompress");
    // ------------ decompress if needed ---------------------
    if (compress_type > 0) {
        particles.compressor.set_compression_type(compress_type);
        particles.compressor.set_quantization_factor(quantization_factor);
        particles.compressor.set_background(compress_background);
        particles.compressor.decompress(particles.data,parts_start);
    }

    // ------------ re-ordering for backwards compatability if needed ------- //
    // only files generated with older versions of the library will need this.
    bool new_particles = attribute_exists(part_location,"lzxy_order");

    if(!new_particles){
        APRWriter::re_order_parts(apr,particles);
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
std::vector<std::string> APRFile::get_particles_names(bool apr_or_tree,uint64_t t,std::string channel_name){

    fileStructure.open_time_point(t,with_tree_flag,channel_name);

    const int max_name_size = 1024;

    ssize_t len;
    hsize_t nobj;
    herr_t err;
    int otype;

    std::vector<std::string> access_names = {"map_level","map_global_index","map_number_gaps","map_x","map_y_begin","map_y_end","map_z","y_vec","xz_end_vec"};

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
