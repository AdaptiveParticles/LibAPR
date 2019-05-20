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

class APRFile {

public:

    APRFile(){

    }

    // setup
    bool open(std::string file_name,std::string read_write = "WRITE");
    void close();

    // write
    void write_apr(APR &apr,uint64_t t = 0);
    void write_apr_append(APR &apr);
    template<typename DataType>
    void write_particles(std::string& particles_name,ParticleData<DataType>& particles,uint64_t t = 0,bool apr_or_tree = true);

    // read
    void read_apr(APR &apr,uint64_t t = 0);
    template<typename DataType>
    void read_particles(std::string& particles_name,ParticleData<DataType>& particles,uint64_t t = 0,bool apr_or_tree = true);

    //set helpers
    void set_read_write_tree(bool write_with_tree_flag_);

    //get helpers
    uint64_t get_number_time_steps();
    std::vector<std::string> get_particles_names(uint64_t t,bool apr_or_tree = true);


    // #TODO get_set methods for Compress + BLSOC parameters + min/max
    // #TODO get_file_info method returning a struct with different file info.
private:

    //Basic Properties.
    uint64_t current_t=0;
    bool write_with_tree_flag = true;
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

};


/**
   * Open the file, creates it if it doesn't exist.
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   * @return (Bool) has the opening of the file been successful or not.
   */
bool APRFile::open(std::string file_name,std::string read_write){

    if(read_write == "WRITE"){

        fileStructure.init(file_name, APRWriter::FileStructure::Operation::WRITE_WITH_TREE);
        if (!fileStructure.isOpened()) return 0;

    } else if(read_write == "READ"){
        fileStructure.init(file_name, APRWriter::FileStructure::Operation::READ_WITH_TREE);
        if (!fileStructure.isOpened()) return 0;
    } else {
        std::cerr << "Files should either be opened as READ or WRITE" << std::endl;
    }

    return true;

}


/**
   * Close the HDF5 file structures
   */
void APRFile::close(){
}


/**
   * Write the APR to file (note this does not include particles)
   * @param APR to be written
   * @param t the time point to be written (default will be to append to the end of the file, starting with 0)
   */
void APRFile::write_apr(APR &apr,uint64_t t){
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
void APRFile::write_particles(std::string& particles_name,ParticleData<DataType>& particles,uint64_t t,bool apr_or_tree){

}

/**
   * Read the an APR from file (note this does not include particles)
   * @param APR to be read to
   * @param t the time point to be written (default will be to append to the end of the file, starting with 0)
   */
void APRFile::read_apr(APR &apr,uint64_t t){

};

/**
   * Read particles from file, they will be associated with a given time point, and either as APR particles of APRTree particles (see get_particles_names for saved particle datasets)
   * @param particles_name string name of the particles dataset to be read (is then to be used for reading, each time point can have a similmarly named dataset)
   * @paramt particles particle dataset to be read (contiguos block of memory)
   * @param t (uint64_t) the time point to be read from. (DEFAULT: t = 0)
   * @param apr_or_tree (Default = True (APR), false = APR Tree)
   */
template<typename DataType>
void APRFile::read_particles(std::string& particles_name,ParticleData<DataType>& particles,uint64_t t,bool apr_or_tree){

};

/**
   * Set whether the internal APR Tree internal access should also be written and read.
   * @param write_with_tree_flag_ indicate whether the APRTree should be written and read. (True = save both APR and APR Tree)
   */
void APRFile::set_read_write_tree(bool write_with_tree_flag_){

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
