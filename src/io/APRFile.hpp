//
// Created by cheesema on 2019-05-20.
//

#ifndef LIBAPR_APRFILE_HPP
#define LIBAPR_APRFILE_HPP

#include "hdf5functions_blosc.h"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRAccess.hpp"
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

    //get helpers
    uint64_t get_number_time_steps();
    std::vector<std::string> get_particles_names(uint64_t t,bool apr_or_tree = true);

    //APRTimer timer(false);

    // #TODO get_set methods for Compress + BLSOC parameters + min/max
    // #TODO get_file_info method returning a struct with different file info.
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
    APRCompress aprCompress;

    //Maximum and minimum levels to read.
    unsigned int maximum_level_read = 0; //have a set method either by level or by delta
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

    APRTimer timer(true);
    APRTimer timer_f(true);

    current_t = t;

    if(fileStructure.isOpened()){
        std::cout << "file is open" << std::endl;
    }

    fileStructure.create_time_point(t,with_tree_flag,channel_name);

    hid_t meta_location = fileStructure.objectId;

    //global property
    APRWriter::writeAttr(AprTypes::TimeStepType, meta_location, &t);

    // ------------- write metadata -------------------------
    //per time step

    APRWriter::writeAttr(AprTypes::NumberOfXType, meta_location, &apr.apr_access.org_dims[1]);
    APRWriter::writeAttr(AprTypes::NumberOfYType, meta_location, &apr.apr_access.org_dims[0]);
    APRWriter::writeAttr(AprTypes::NumberOfZType, meta_location, &apr.apr_access.org_dims[2]);
    APRWriter::writeAttr(AprTypes::TotalNumberOfGapsType, meta_location, &apr.apr_access.total_number_gaps);
    APRWriter::writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, meta_location, &apr.apr_access.total_number_non_empty_rows);

    APRWriter::writeString(AprTypes::NameType,meta_location, (apr.name.size() == 0) ? "no_name" : apr.name);
    APRWriter::writeString(AprTypes::GitType, meta_location, ConfigAPR::APR_GIT_HASH);
    APRWriter::writeAttr(AprTypes::TotalNumberOfParticlesType, meta_location, &apr.apr_access.total_number_particles);
    APRWriter::writeAttr(AprTypes::MaxLevelType, meta_location, &apr.apr_access.l_max);
    APRWriter::writeAttr(AprTypes::MinLevelType, meta_location, &apr.apr_access.l_min);


    APRWriter::writeAttr(AprTypes::LambdaType, meta_location, &apr.parameters.lambda);
    APRWriter::writeAttr(AprTypes::SigmaThType, meta_location, &apr.parameters.sigma_th);
    APRWriter::writeAttr(AprTypes::SigmaThMaxType, meta_location, &apr.parameters.sigma_th_max);
    APRWriter::writeAttr(AprTypes::IthType, meta_location, &apr.parameters.Ip_th);
    APRWriter::writeAttr(AprTypes::DxType, meta_location, &apr.parameters.dx);
    APRWriter::writeAttr(AprTypes::DyType, meta_location, &apr.parameters.dy);
    APRWriter::writeAttr(AprTypes::DzType, meta_location, &apr.parameters.dz);
    APRWriter::writeAttr(AprTypes::PsfXType, meta_location, &apr.parameters.psfx);
    APRWriter::writeAttr(AprTypes::PsfYType, meta_location, &apr.parameters.psfy);
    APRWriter::writeAttr(AprTypes::PsfZType, meta_location, &apr.parameters.psfz);
    APRWriter::writeAttr(AprTypes::RelativeErrorType, meta_location, &apr.parameters.rel_error);
    APRWriter::writeAttr(AprTypes::NoiseSdEstimateType, meta_location, &apr.parameters.noise_sd_estimate);
    APRWriter::writeAttr(AprTypes::BackgroundIntensityEstimateType, meta_location,
                         &apr.parameters.background_intensity_estimate);

    timer.start_timer("access_data");
    MapStorageData map_data;
    apr.apr_access.flatten_structure( map_data);

    std::vector<uint16_t> index_delta;
    index_delta.resize(map_data.global_index.size());
    std::adjacent_difference(map_data.global_index.begin(),map_data.global_index.end(),index_delta.begin());
    APRWriter::writeData(AprTypes::MapGlobalIndexType, fileStructure.objectId, index_delta, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

    APRWriter::writeData(AprTypes::MapYendType, fileStructure.objectId, map_data.y_end, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
    APRWriter::writeData(AprTypes::MapYbeginType, fileStructure.objectId, map_data.y_begin, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
    APRWriter::writeData(AprTypes::MapNumberGapsType, fileStructure.objectId, map_data.number_gaps, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
    APRWriter::writeData(AprTypes::MapLevelType, fileStructure.objectId, map_data.level, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
    APRWriter::writeData(AprTypes::MapXType, fileStructure.objectId, map_data.x, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
    APRWriter::writeData(AprTypes::MapZType, fileStructure.objectId, map_data.z, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

    timer.stop_timer();

    for (size_t i = apr.level_min(); i < apr.level_max() ; ++i) {
        int x_num = (int) apr.apr_access.x_num[i];
        APRWriter::writeAttr(AprTypes::NumberOfLevelXType, i, meta_location, &x_num);
        int y_num = (int) apr.apr_access.y_num[i];
        APRWriter::writeAttr(AprTypes::NumberOfLevelYType, i, meta_location, &y_num);
        int z_num = (int) apr.apr_access.z_num[i];
        APRWriter::writeAttr(AprTypes::NumberOfLevelZType, i, meta_location, &z_num);
    }

    if(with_tree_flag){

        apr.init_tree(); //incase it hasn't been initialized.


        APRWriter::writeAttr(AprTypes::TotalNumberOfGapsType, fileStructure.objectIdTree, &apr.tree_access.total_number_gaps);
        APRWriter::writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, fileStructure.objectIdTree, &apr.tree_access.total_number_non_empty_rows);
        APRWriter::writeAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree, &apr.tree_access.total_number_particles);

        MapStorageData map_data_tree;
        apr.tree_access.flatten_structure( map_data_tree);

        std::vector<uint16_t> index_delta;
        index_delta.resize(map_data_tree.global_index.size());
        std::adjacent_difference(map_data_tree.global_index.begin(),map_data_tree.global_index.end(),index_delta.begin());
        APRWriter::writeData(AprTypes::MapGlobalIndexType, fileStructure.objectIdTree, index_delta, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

        APRWriter::writeData(AprTypes::MapYendType, fileStructure.objectIdTree, map_data_tree.y_end, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapYbeginType, fileStructure.objectIdTree, map_data_tree.y_begin, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapNumberGapsType, fileStructure.objectIdTree, map_data_tree.number_gaps, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapLevelType, fileStructure.objectIdTree, map_data_tree.level, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapXType, fileStructure.objectIdTree, map_data_tree.x, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapZType, fileStructure.objectIdTree, map_data_tree.z, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);


    }


}

/**
   * Write the APR to file and append it as the next time point (note this does not include particles)
   * @param APR to be written
   */
void APRFile::write_apr_append(APR &apr){
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

    APRTimer timer(true);
    APRTimer timer_f(true);

    if(fileStructure.isOpened()){
        std::cout << "file is open" << std::endl;
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

    // check if the APR structure has already been read in, the case when a partial load has been done, now only the particles need to be read
    uint64_t old_particles = apr.apr_access.total_number_particles;
    uint64_t old_gaps = apr.apr_access.total_number_gaps;

    //read in access information
    APRWriter::read_access_info(meta_data,apr.apr_access);

    //read in pipeline parameters
    APRWriter::read_apr_parameters(meta_data,apr.parameters);

    // ------------- map handling ----------------------------

    timer.start_timer("map loading data");

    auto map_data = std::make_shared<MapStorageData>();

    map_data->global_index.resize(apr.apr_access.total_number_non_empty_rows);

    timer_f.start_timer("index");
    std::vector<int16_t> index_delta(apr.apr_access.total_number_non_empty_rows);
    APRWriter::readData(AprTypes::MapGlobalIndexType, fileStructure.objectId, index_delta.data());
    std::vector<uint64_t> index_delta_big(apr.apr_access.total_number_non_empty_rows);
    std::copy(index_delta.begin(), index_delta.end(), index_delta_big.begin());
    std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data->global_index.begin());

    timer_f.stop_timer();

    timer_f.start_timer("y_b_e");
    map_data->y_end.resize(apr.apr_access.total_number_gaps);
    APRWriter::readData(AprTypes::MapYendType, fileStructure.objectId, map_data->y_end.data());
    map_data->y_begin.resize(apr.apr_access.total_number_gaps);
    APRWriter::readData(AprTypes::MapYbeginType, fileStructure.objectId, map_data->y_begin.data());

    timer_f.stop_timer();

    timer_f.start_timer("zxl");
    map_data->number_gaps.resize(apr.apr_access.total_number_non_empty_rows);
    APRWriter::readData(AprTypes::MapNumberGapsType, fileStructure.objectId, map_data->number_gaps.data());
    map_data->level.resize(apr.apr_access.total_number_non_empty_rows);
    APRWriter::readData(AprTypes::MapLevelType, fileStructure.objectId, map_data->level.data());
    map_data->x.resize(apr.apr_access.total_number_non_empty_rows);
    APRWriter::readData(AprTypes::MapXType, fileStructure.objectId, map_data->x.data());
    map_data->z.resize(apr.apr_access.total_number_non_empty_rows);
    APRWriter::readData(AprTypes::MapZType, fileStructure.objectId, map_data->z.data());
    timer_f.stop_timer();

    timer.stop_timer();

    timer.start_timer("map building");

    apr.apr_access.rebuild_map(*map_data);

    timer.stop_timer();

    if(with_tree_flag) {

        if(!tree_exists){
            //initializing it from the dataset.
            std::cout << "Initializing tree from file" << std::endl;
            apr.init_tree();
        } else {

            timer.start_timer("build tree - map");

            apr.tree_access.l_max = apr.level_max() - 1;
            apr.tree_access.l_min = apr.level_min() - 1;

            apr.tree_access.x_num.resize(apr.tree_access.level_max() + 1);
            apr.tree_access.z_num.resize(apr.tree_access.level_max() + 1);
            apr.tree_access.y_num.resize(apr.tree_access.level_max() + 1);

            for (int i = apr.tree_access.level_min(); i <= apr.tree_access.level_max(); ++i) {
                apr.tree_access.x_num[i] = apr.spatial_index_x_max(i);
                apr.tree_access.y_num[i] = apr.spatial_index_y_max(i);
                apr.tree_access.z_num[i] = apr.spatial_index_z_max(i);
            }

            apr.tree_access.x_num[apr.level_min() - 1] = ceil(apr.spatial_index_x_max(apr.level_min()) / 2.0f);
            apr.tree_access.y_num[apr.level_min() - 1] = ceil(apr.spatial_index_y_max(apr.level_min()) / 2.0f);
            apr.tree_access.z_num[apr.level_min() - 1] = ceil(apr.spatial_index_z_max(apr.level_min()) / 2.0f);

            APRWriter::readAttr(AprTypes::TotalNumberOfParticlesType, fileStructure.objectIdTree,
                                &apr.tree_access.total_number_particles);
            APRWriter::readAttr(AprTypes::TotalNumberOfGapsType, fileStructure.objectIdTree,
                                &apr.tree_access.total_number_gaps);
            APRWriter::readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, fileStructure.objectIdTree,
                                &apr.tree_access.total_number_non_empty_rows);

            auto map_data_tree = std::make_shared<MapStorageData>();

            map_data_tree->global_index.resize(apr.tree_access.total_number_non_empty_rows);

            std::vector<int16_t> index_delta(apr.tree_access.total_number_non_empty_rows);
            APRWriter::readData(AprTypes::MapGlobalIndexType, fileStructure.objectIdTree, index_delta.data());
            std::vector<uint64_t> index_delta_big(apr.tree_access.total_number_non_empty_rows);
            std::copy(index_delta.begin(), index_delta.end(), index_delta_big.begin());
            std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data_tree->global_index.begin());

            map_data_tree->y_end.resize(apr.tree_access.total_number_gaps);
            APRWriter::readData(AprTypes::MapYendType, fileStructure.objectIdTree, map_data_tree->y_end.data());
            map_data_tree->y_begin.resize(apr.tree_access.total_number_gaps);
            APRWriter::readData(AprTypes::MapYbeginType, fileStructure.objectIdTree, map_data_tree->y_begin.data());

            map_data_tree->number_gaps.resize(apr.tree_access.total_number_non_empty_rows);
            APRWriter::readData(AprTypes::MapNumberGapsType, fileStructure.objectIdTree,
                                map_data_tree->number_gaps.data());
            map_data_tree->level.resize(apr.tree_access.total_number_non_empty_rows);
            APRWriter::readData(AprTypes::MapLevelType, fileStructure.objectIdTree, map_data_tree->level.data());
            map_data_tree->x.resize(apr.tree_access.total_number_non_empty_rows);
            APRWriter::readData(AprTypes::MapXType, fileStructure.objectIdTree, map_data_tree->x.data());
            map_data_tree->z.resize(apr.tree_access.total_number_non_empty_rows);
            APRWriter::readData(AprTypes::MapZType, fileStructure.objectIdTree, map_data_tree->z.data());

            apr.tree_access.rebuild_map_tree(*map_data_tree, apr.apr_access);

            timer.stop_timer();
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
        std::cout << "file is open" << std::endl;
    }

    fileStructure.open_time_point(t,with_tree_flag,channel_name);

    uint64_t max_read_level = apr.apr_access.level_max()-max_level_delta;
    //uint64_t max_read_level_tree = std::min(apr.apr_access.level_max()-1,max_read_level);
    uint64_t prev_read_level = 0;

    APRTimer timer;

    uint64_t parts_start = 0;
    uint64_t parts_end = apr.apr_access.global_index_by_level_end[max_read_level] + 1;

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

    prev_read_level = 0;

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

    //std::cout << "Data rate intensities: " << (apr.particles_intensities.data.size()*2)/(timer.timings.back()*1000000.0f) << " MB/s" << std::endl;

    timer.start_timer("decompress");
    // ------------ decompress if needed ---------------------
    if (compress_type > 0) {
        aprCompress.set_compression_type(compress_type);
        aprCompress.set_quantization_factor(quantization_factor);
        aprCompress.decompress(apr, particles,parts_start);
    }
    timer.stop_timer();

//    if(!read_structure) {
//        uint64_t current_parts_size = apr.total_number_particles();
//
//        for (int j = apr.level_min(); j <apr.level_max(); ++j) {
//            if((apr.apr_access.global_index_by_level_end[j] + 1)==current_parts_size){
//                prev_read_level = j;
//            }
//        }
//    }

//    if(prev_read_level > 0){
//        parts_start = apr.apr_access.global_index_by_level_end[prev_read_level] + 1;
//    }

    //apr.apr_access.level_max = max_read_level;

};



//get helpers

/**
   * Number of time steps saved in the file
    * @return Number of time steps.
   */
uint64_t APRFile::get_number_time_steps(){

};

/**
   * Gets the names of particles datasets saved for a particular file for a time step, and either for APR, or APR Tree.
   * @param t the time step the particle datasets have been saved to.
   * @param apr_or_tree Is it an APR or APR Tree dataset. (Defualt = true (APR), flase = (APR Tree))
   * @return vector of strings of the names of the datasets (can be then used with read_particles).
   */
std::vector<std::string> APRFile::get_particles_names(uint64_t t,bool apr_or_tree){

};



#endif //LIBAPR_APRFILE_HPP
