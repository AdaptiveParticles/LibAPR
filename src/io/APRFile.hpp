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
    bool write_apr(APR& apr,uint64_t t = 0,std::string channel_name = "t", bool write_tree = true);
    bool write_apr_append(APR& apr);
    template<typename DataType>
    bool write_particles(std::string particles_name,ParticleData<DataType>& particles,bool apr_or_tree = true,uint64_t t = 0,std::string channel_name = "t");

    // read
    bool read_apr(APR& apr,uint64_t t = 0,std::string channel_name = "t");

    bool read_metadata(GenInfo& aprInfo, uint64_t t = 0, std::string channel_name = "t");

    bool read_metadata_tree(GenInfo& treeInfo, uint64_t t = 0, std::string channel_name = "t");

    template<typename DataType>
    bool read_particles(APR& apr,std::string particles_name,ParticleData<DataType>& particles,bool apr_or_tree = true,uint64_t t = 0,std::string channel_name = "t");

    template<typename DataType>
    bool read_particles(APR& apr,ParticleData<DataType>& particles,bool apr_or_tree = true,uint64_t t = 0,std::string channel_name = "t");

    std::string get_particle_type(std::string particles_name, bool apr_or_tree=true, uint64_t t=0, std::string channel_name="t");

    template<typename DataType>
    bool read_particles(std::string particles_name, ParticleData<DataType>& particles, bool apr_or_tree=true, uint64_t t=0, std::string channel_name="t");


    [[deprecated("this now controlled via arguments to affected methods")]]
    bool get_read_write_tree(){
        return with_tree_flag;
    }

    [[deprecated("this now controlled via arguments to affected methods")]]
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
    bool with_tree_flag = true; //usage is deprecated, keeping for backward compatibility
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

    hid_t open_dataset_location(std::string& particles_name, bool apr_or_tree, uint64_t t, std::string& channel_name);


};

/**
   * Open the file, creates it if it doesn't exist.
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   * @return (Bool) has the opening of the file been successful or not.
   */
bool APRFile::open(std::string file_name_,std::string read_write_append){

    if(read_write_append == "WRITE"){
        return fileStructure.init(file_name_, APRWriter::FileStructure::Operation::WRITE);
    } else if(read_write_append == "READ") {
        return fileStructure.init(file_name_, APRWriter::FileStructure::Operation::READ);
    } else if(read_write_append == "READWRITE") {
        return fileStructure.init(file_name_, APRWriter::FileStructure::Operation::WRITE_APPEND);
    } else {
        std::cerr << "Files should either be opened as READ or WRITE, or READWRITE" << std::endl;
        return false;
    }
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
bool APRFile::write_apr(APR &apr, uint64_t t, std::string channel_name, bool write_tree){

    current_t = t;

    if(!fileStructure.isOpened()){
        std::cerr << "File is not open!" << std::endl;
        return false;
    }

    timer.start_timer("setup");

    fileStructure.create_time_point(t, write_tree, channel_name);

    hid_t meta_location = fileStructure.objectId;

    //global property
    APRWriter::writeAttr(AprTypes::TimeStepType, meta_location, &t);

    // ------------- write metadata -------------------------
    //per time step
    APRWriter::writeString(AprTypes::NameType,meta_location, (apr.name.size() == 0) ? "no_name" : apr.name);
    APRWriter::writeString(AprTypes::GitType, meta_location, ConfigAPR::APR_GIT_HASH);

    APRWriter::write_apr_info(meta_location,apr.aprInfo);

    APRWriter::write_apr_parameters(meta_location,apr.parameters);

    timer.stop_timer();

    if(write_linear){
        timer.start_timer("init_access");
        apr.initialize_linear();
        timer.stop_timer();

        timer.start_timer("write_apr_access_data");

        APRWriter::write_linear_access(meta_location, fileStructure.objectId, apr.linearAccess, blosc_comp_type_access,
                            blosc_comp_level_access, blosc_shuffle_access);
        timer.stop_timer();

    } else {
        timer.start_timer("write_apr_access_data");
        apr.initialize_random(); //check that it is initialized.

        APRWriter::write_random_access(meta_location, fileStructure.objectId, apr.apr_access, blosc_comp_type_access,
                                       blosc_comp_level_access, blosc_shuffle_access);
        timer.stop_timer();
    }


    if(write_tree){

        if(write_linear_tree) {

            timer.start_timer("init_tree");
            apr.initialize_tree_linear();
            timer.stop_timer();

            timer.start_timer("write_tree_access_data");
            APRWriter::writeAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree,
                                 &apr.treeInfo.total_number_particles);

            APRWriter::write_linear_access(fileStructure.objectIdTree, fileStructure.objectIdTree, apr.linearAccessTree, blosc_comp_type_access,
                                           blosc_comp_level_access, blosc_shuffle_access);
            timer.stop_timer();
        } else {

            apr.initialize_tree_random(); //incase it hasn't been initialized.

            APRWriter::writeAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree,
                                 &apr.treeInfo.total_number_particles);

            APRWriter::write_random_access(fileStructure.objectIdTree, fileStructure.objectIdTree, apr.tree_access,
                                           blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

        }

    }

    return true;


}

/**
   * Write the APR to file and append it as the next time point (note this does not include particles)
   * @param APR to be written
   */
bool APRFile::write_apr_append(APR &apr){
    return write_apr(apr,current_t + 1);
}

/**
   * Write particles to file, they will be associated with a given time point, and either as APR particles of APRTree particles
   * @param particles_name string name of the particles dataset to be written (is then to be used for reading, each time point can have a similmarly named dataset)
   * @paramt particles particle dataset to be written (contiguos block of memory)
   * @param t (uint64_t) the time point to be written. (DEFAULT: t = 0)
   * @param apr_or_tree (Default = true (APR), false = APR Tree)
   */
template<typename DataType>
bool APRFile::write_particles(const std::string particles_name,ParticleData<DataType>& particles,bool apr_or_tree,uint64_t t,std::string channel_name){


    if(fileStructure.isOpened()){
    } else {
        std::cerr << "File is not open!" << std::endl;
    }

    // opens dataset if it exists, otherwise it is created
    fileStructure.create_time_point(t, !apr_or_tree, channel_name);

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
    if(!data_exists(part_location,particles_name.c_str())) {
        partsData.create(type, particles.size());
        partsData.open();
    } else {
        // if dataset already exists, allow overwriting if the size and datatype match the input particles
        partsData.open();

        std::vector<uint64_t> new_dims = {particles.size()};
        auto old_dims = partsData.get_dimensions();

        if(new_dims != old_dims) {
            std::cerr << "Failed to overwrite existing particle dataset '" << particles_name << "': input size does not match." << std::endl;
            partsData.close();
            return false;
        }

        if(!H5Tequal(type, partsData.get_type())) {
            std::cerr << "Failed to overwrite existing particle dataset '" << particles_name << "': input datatype does not match." << std::endl;
            partsData.close();
            return false;
        }
    }
    partsData.write(particles.data.data(),0,particles.size());
    partsData.close();


    timer.stop_timer();

    return true;

}

/**
   * Read the an APR from file (note this does not include particles)
   * @param APR to be read to
   * @param t the time point to be written (default will be to append to the end of the file, starting with 0)
   */
bool APRFile::read_apr(APR &apr,uint64_t t,std::string channel_name){


    if(!fileStructure.isOpened()){
        std::cerr << "File is not open!" << std::endl; //#TODO: should check if it is readable, and other functions if writeable ect.
        return false;
    }

    if(!fileStructure.open_time_point(t, false, channel_name)) {
        std::cerr << "Error reading APR file: could not open time point t=" << t << " in channel '" << channel_name << "'" << std::endl;
        return false;
    }

    hid_t meta_data = fileStructure.groupId;

    //check if old or new file, for location of the properties. (The metadata moved to the time point.)
    if(attribute_exists(fileStructure.objectId,AprTypes::MaxLevelType.typeName)) {
        meta_data = fileStructure.objectId;
    }

    std::string data_n = fileStructure.subGroup1 + "/map_level";
    bool stored_random = data_exists(fileStructure.fileId,data_n.c_str());

    data_n = fileStructure.subGroup1 + "/y_vec";
    bool stored_linear = data_exists(fileStructure.fileId,data_n.c_str());

    if(!stored_linear && !stored_random){
        std::cerr << "Error reading APR file: data does not exist for channel '" <<
                      channel_name << "' and time point t=" << t << std::endl;
        return false;
    }

    // ------------- read metadata --------------------------
    char string_out[100] = {0};
    hid_t attr_id = H5Aopen(meta_data,"name",H5P_DEFAULT);
    hid_t atype = H5Aget_type(attr_id);
    hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_DEFAULT);
    H5Aread(attr_id, atype_mem, string_out) ;
    H5Aclose(attr_id);
    apr.name = string_out;

    //read in access information
    APRWriter::read_access_info(meta_data,apr.aprInfo);
    apr.linearAccess.genInfo = &apr.aprInfo;
    apr.apr_access.genInfo = &apr.aprInfo;

    //read in pipeline parameters
    APRWriter::read_apr_parameters(meta_data,apr.parameters);

    timer.start_timer("read_apr_access");

    if(stored_linear) {
        //TODO: make this an automatic check to see what the file is.
        APRWriter::read_linear_access( fileStructure.objectId, apr.linearAccess,max_level_delta);
        apr.apr_initialized = true;
    } else {
        APRWriter::read_random_access(meta_data, fileStructure.objectId, apr.apr_access);
        apr.apr_initialized_random = true;
    }

    timer.stop_timer();

    timer.start_timer("read_tree_access");

    data_n = fileStructure.subGroupTree1 + "/map_level";
    bool stored_random_tree = data_exists(fileStructure.fileId,data_n.c_str());

    data_n = fileStructure.subGroupTree1 + "/y_vec";
    bool stored_linear_tree = data_exists(fileStructure.fileId,data_n.c_str());;

    bool tree_exists = stored_linear_tree || stored_random_tree;

    if(tree_exists){

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

    timer.stop_timer();

    return true;
}


bool APRFile::read_metadata(GenInfo& aprInfo, uint64_t t, std::string channel_name){

    if(!fileStructure.isOpened()){
        std::cerr << "File is not open!" << std::endl;
        return false;
    }

    if(!fileStructure.open_time_point(t, false, channel_name)) {
        std::cerr << "Error reading APR file: could not open time point t=" << t << " in channel '" << channel_name << "'" << std::endl;
        return false;
    }

    APRWriter::read_access_info(fileStructure.objectId, aprInfo);
    return true;
}


bool APRFile::read_metadata_tree(GenInfo& treeInfo, uint64_t t, std::string channel_name) {
    if(!fileStructure.isOpened()){
        std::cerr << "File is not open!" << std::endl;
        return false;
    }

    if(!fileStructure.open_time_point(t, true,channel_name)) {
        std::cerr << "Error reading APR file: could not open time point t=" << t << " in channel '" << channel_name << "'" << std::endl;
        return false;
    }

    APRWriter::read_dims(fileStructure.objectId, treeInfo);
    APRWriter::read_access_info_tree(fileStructure.objectIdTree, treeInfo);
    treeInfo.init_tree(treeInfo.org_dims[0], treeInfo.org_dims[1], treeInfo.org_dims[2]);
    return true;
}



/**
   * Read particles from file, they will be associated with a given time point, and either as APR particles of APRTree particles (see get_particles_names for saved particle datasets)
   * @param apr requires a valid APR, this is for more general functionality including partial reading by level.
   * @param particles_name string name of the particles dataset to be read (is then to be used for reading, each time point can have a similmarly named dataset)
   * @paramt particles particle dataset to be read (contiguos block of memory)
   * @param t (uint64_t) the time point to be read from. (DEFAULT: t = 0)
   * @param apr_or_tree (Default = True (APR), false = APR Tree)
   */
template<typename DataType>
bool APRFile::read_particles(APR &apr,
                             std::string particles_name,
                             ParticleData<DataType>& particles,
                             bool apr_or_tree,
                             uint64_t t,
                             std::string channel_name){

    hid_t part_location = open_dataset_location(particles_name, apr_or_tree, t, channel_name);

    if(part_location == 0){
        return false;
    }

    int max_read_level;
    uint64_t parts_start = 0;
    uint64_t parts_end = 0;


     // Check that the APR is initialized.
    if(!apr.is_initialized()){
        std::cerr << "Error reading particles: input APR is not initialized" << std::endl;
        return false;
    }

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

    hid_t meta_data = part_location;

    // for backwards compatibility
    if(!attribute_exists(part_location,AprTypes::CompressionType.typeName)) {
        meta_data = fileStructure.groupId;
    }

    Hdf5DataSet dataset;
    dataset.init(part_location,particles_name.c_str());
    dataset.open();

    // Check the size of the dataset
    std::vector<uint64_t> dims = dataset.get_dimensions();
    uint64_t number_particles_in_dataset = dims[0];

    if(number_particles_in_dataset < parts_end){
        std::cerr << "Error reading particles: dataset is not correct size" << std::endl;
        dataset.close();
        return false;
    }

    // Check the datatype
    hid_t parts_type = APRWriter::Hdf5Type<DataType>::type();
    hid_t dtype = dataset.get_type();

    if(!H5Tequal(parts_type, dtype)) {
        std::cerr << "Error reading particles: datatype does not match input particles" << std::endl;
        dataset.close();
        return false;
    }

    timer.start_timer("Read intensities");

    // ------------- read data ------------------------------
    particles.data.resize(parts_end - parts_start);
    if (particles.data.size() > 0) {
        dataset.read(particles.data.data() + parts_start,parts_start,parts_end);
    }

    timer.stop_timer();
    dataset.close();
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

    return true;

}

template<typename DataType>
bool APRFile::read_particles(APR& apr,ParticleData<DataType>& particles,bool apr_or_tree,uint64_t t ,std::string channel_name){

    std::vector<std::string> file_names = this->get_particles_names(apr_or_tree, t, channel_name);
    std::string particles_name = file_names[0]; //by default it takes the first set of particles.

    bool read =  this->read_particles(apr,particles_name,particles,apr_or_tree,t,channel_name);

    if(read){
        std::cout << "Default Reading Particles Named: " << particles_name << std::endl;
    }

    return read;
}

hid_t APRFile::open_dataset_location(std::string& particles_name, bool apr_or_tree, uint64_t t, std::string& channel_name) {

    if (!fileStructure.isOpened()) {
        std::cerr << "File is not open!" << std::endl;
        return 0;
    }

    if (!fileStructure.open_time_point(t, !apr_or_tree, channel_name)) {
        std::cerr << "Error reading APR file: could not open time point t=" << t << " in channel '" << channel_name
                  << "'" << std::endl;
        return 0;
    }

    hid_t part_location;

    if (apr_or_tree) {
        part_location = fileStructure.objectId;
    } else {
        part_location = fileStructure.objectIdTree;
    }

    // Check that the dataset exists
    std::string data_n = particles_name;
    if (!data_exists(part_location, data_n.c_str())) {
        std::cerr << "Error reading APR file: particle dataset '" << particles_name << "' doesn't exist" << std::endl;
        return 0;
    }

    return part_location;
}

std::string APRFile::get_particle_type(std::string particles_name, bool apr_or_tree, uint64_t t, std::string channel_name) {

    hid_t part_location = open_dataset_location(particles_name, apr_or_tree,t,channel_name);

    if(part_location == 0){
        return "";
    }

    // Check the datatype
    Hdf5DataSet dataset;
    dataset.init(part_location,particles_name.c_str());
    dataset.open();
    hid_t dtype = dataset.get_type();

    if( H5Tequal(dtype, APRWriter::Hdf5Type<uint16_t>::type()) ) { dataset.close(); return "uint16"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<float>::type()) ) { dataset.close(); return "float"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<uint8_t>::type()) ) { dataset.close(); return "uint8"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<uint32_t>::type()) ) { dataset.close(); return "uint32"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<uint64_t>::type()) ) { dataset.close(); return "uint64"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<double>::type()) ) { dataset.close(); return "double"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<int8_t>::type()) ) { dataset.close(); return "int8"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<int16_t>::type()) ) { dataset.close(); return "int16"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<int32_t>::type()) ) { dataset.close(); return "int32"; }
    if( H5Tequal(dtype, APRWriter::Hdf5Type<int64_t>::type()) ) { dataset.close(); return "int64"; }

    std::cerr << "Error: get_particle_type could not detect the data type (unsupported type)" << std::endl;
    return "";
}


/**
 * Read particle dataset from file without requiring the corresponding APR.
 * @tparam DataType
 * @param particles_name
 * @param particles
 * @param apr_or_tree
 * @param t
 * @param channel_name
 * @return
 */
template<typename DataType>
bool APRFile::read_particles(std::string particles_name, ParticleData<DataType>& particles, bool apr_or_tree, uint64_t t, std::string channel_name){

    if(!fileStructure.isOpened()){
        std::cerr << "File is not open!" << std::endl;
        return false;
    }

    if(!fileStructure.open_time_point(t, !apr_or_tree, channel_name)) {
        std::cerr << "Error reading APR file: could not open time point t=" << t << " in channel '" << channel_name << "'" << std::endl;
        return false;
    }

    //check if old or new file, for location of the properties. (The metadata moved to the time point.)
    hid_t part_location;
    if(apr_or_tree){
        part_location = fileStructure.objectId;
    } else {
        part_location = fileStructure.objectIdTree;
    }

    // Check that the dataset exists
    std::string data_n = particles_name;
    if(!data_exists(part_location,data_n.c_str())) {
        std::cerr << "Error reading APR file: particle dataset '" << particles_name << "' doesn't exist" << std::endl;
        return false;
    }

    hid_t meta_data = part_location;
    Hdf5DataSet dataset;
    dataset.init(part_location,particles_name.c_str());
    dataset.open();

    // Get the size of the dataset
    std::vector<uint64_t> dims = dataset.get_dimensions();
    uint64_t number_particles_in_dataset = dims[0];

    // Check the datatype
    hid_t parts_type = APRWriter::Hdf5Type<DataType>::type();
    hid_t dtype = dataset.get_type();
    if(!H5Tequal(parts_type, dtype)) {
        std::cerr << "Error reading particles: datatype does not match input particles" << std::endl;
        dataset.close();
        return false;
    }

    particles.init(number_particles_in_dataset);

    timer.start_timer("Read intensities");
    // ------------- read data ------------------------------
    if (particles.data.size() > 0) {
        dataset.read(particles.begin(), 0, number_particles_in_dataset);
    }
    timer.stop_timer();
    dataset.close();
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
    }
    timer.start_timer("decompress");
    // ------------ decompress if needed ---------------------
    if (compress_type > 0) {
        particles.compressor.set_compression_type(compress_type);
        particles.compressor.set_quantization_factor(quantization_factor);
        particles.compressor.set_background(compress_background);
        particles.compressor.decompress(particles.data);
    }
    timer.stop_timer();
    return true;
}


//get helpers

/**
   * Number of time steps saved in the file
   *   @param Channel name (default = t)
    * @return Number of time steps.
   */
uint64_t APRFile::get_number_time_steps(std::string channel_name){

    const int max_name_size = 1024;

    hsize_t nobj;
    int otype;

    hid_t obj_name = fileStructure.groupId;

    char group_name[max_name_size];
    char memb_name[max_name_size];

    H5Iget_name(obj_name, group_name, max_name_size  );

    H5Gget_num_objs(obj_name, &nobj);

    uint64_t counter_t = 0;

    for (int i = 0; i < (int) nobj; i++) {

        H5Gget_objname_by_idx(obj_name, (hsize_t) i,
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

}

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

    for (int i = 0; i < (int) nobj; i++) {

        len = H5Gget_objname_by_idx(obj_name, (hsize_t) i,
                                    memb_name, (size_t) max_name_size);

        otype = H5Gget_objtype_by_idx(obj_name, (size_t) i);

        if(otype == H5G_GROUP){
            channel_names.push_back(memb_name);
        }

    }

    (void) len;
    (void) err;

    return channel_names;

}


/**
   * Gets the names of particles datasets saved for a particular file for a time step, and either for APR, or APR Tree.
   * @param t the time step the particle datasets have been saved to.
   * @param apr_or_tree Is it an APR or APR Tree dataset. (Defualt = true (APR), flase = (APR Tree))
   * @return vector of strings of the names of the datasets (can be then used with read_particles).
   */
std::vector<std::string> APRFile::get_particles_names(bool apr_or_tree,uint64_t t,std::string channel_name){

    fileStructure.open_time_point(t, !apr_or_tree, channel_name);

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

    for (int i = 0; i < (int) nobj; i++) {

        len = H5Gget_objname_by_idx(obj_name, (hsize_t)i,
                                    memb_name, (size_t)max_name_size );

        otype =  H5Gget_objtype_by_idx(obj_name, (size_t)i );

        if(otype == H5G_DATASET){
            bool access = false;

            for (int j = 0; j < (int) access_names.size(); ++j) {
                if(memb_name == access_names[j]){
                    access = true;
                }
            }
            if(!access) {
                dataset_names.push_back(memb_name);
            }
        }
    }

    (void) len;
    (void) err;

    return dataset_names;

}



#endif //LIBAPR_APRFILE_HPP
