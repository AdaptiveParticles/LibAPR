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
    void open(std::string file_name,std::string read_write = "WRITE");
    void close();

    // write
    void write_apr(APR &apr,uint64_t t = 0);
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

private:

    //Basic Properties.
    uint64_t current_t=0;
    bool write_with_tree_flag = true;
    std::string name = "noname";
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
   * Open the file
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   */
void APRFile::open(std::string file_name,std::string read_write){

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
   * Write particles to file, they will be associated with a given time point, and either as APR particles of APRTree particles
   * @param particles_name string name of the particles dataset to be written (is then to be used for reading, each time point can have a similmarly named dataset)
   * @paramt particles particle dataset to be written (contiguos block of memory)
   * @param t (uint64_t) the time point to be read from. (DEFAULT: t = 0)
   * @param apr_or_tree
   */
template<typename DataType>
void APRFile::write_particles(std::string& particles_name,ParticleData<DataType>& particles,uint64_t t,bool apr_or_tree){

}


/**
   * Open the file
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   */
void read_apr(APR &apr,uint64_t t = 0);

/**
   * Open the file
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   */
template<typename DataType>
void read_particles(std::string& particles_name,ParticleData<DataType>& particles,uint64_t t = 0,bool apr_or_tree = true);

/**
   * Open the file
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   */
void set_read_write_tree(bool write_with_tree_flag_);

//get helpers

/**
   * Open the file
    * @return created object by value
   */
uint64_t get_number_time_steps();

/**
   * Open the file
   * @param file_name Absolute path to the file to be created.
   * @param read_write Either "WRITE" or "READ"
   * @return created object by value
   */
std::vector<std::string> get_particles_names(uint64_t t,bool apr_or_tree = true);



#endif //LIBAPR_APRFILE_HPP
