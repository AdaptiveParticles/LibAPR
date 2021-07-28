///////////////////////////////////////////
//
//  ImageGen 2016
//
//  Bevan Cheeseman 2016
//
//  Header file with some specific hdf5 functions
//
//////////////////////////////////////////

#ifndef PARTPLAY_HDF5FUNCTIONS_BLOSC_H
#define PARTPLAY_HDF5FUNCTIONS_BLOSC_H

extern "C" {
	#include "blosc_filter.h"
}
#include "hdf5.h"
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>


//extern "C"
#ifdef WIN_COMPILE
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API
#endif

LIBRARY_API void hdf5_register_blosc();
LIBRARY_API hid_t hdf5_create_file_blosc(std::string file_name);
LIBRARY_API void hdf5_load_data_blosc(hid_t obj_id, void* buff, const char* data_name);
LIBRARY_API void hdf5_load_data_blosc(hid_t obj_id, hid_t dataType, void* buff, const char* data_name);
LIBRARY_API void hdf5_write_attribute_blosc(hid_t obj_id,hid_t type_id,const char* attr_name,hsize_t rank,hsize_t* dims, const void * const data );
LIBRARY_API void hdf5_write_data_blosc(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data ,unsigned int comp_type,unsigned int comp_level,unsigned int shuffle);
LIBRARY_API void hdf5_write_data_standard(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data );
LIBRARY_API void write_main_paraview_xdmf_xml(const std::string &aDestinationDir,const std::string &aHdf5FileName, const std::string &aParaviewFileName, uint64_t aNumOfParticles);
LIBRARY_API void hdf5_load_data_blosc_partial(hid_t obj_id, void* buff, const char* data_name,uint64_t number_of_elements_read,uint64_t number_of_elements_total);
LIBRARY_API void write_main_paraview_xdmf_xml_time(const std::string &aDestinationDir,const std::string &aHdf5FileName, const std::string &aParaviewFileName, std::vector<uint64_t> aNumOfParticles);

LIBRARY_API void hdf5_write_data_blosc_create(hid_t obj_id, hid_t type_id, const char *ds_name, hsize_t rank, hsize_t *dims, void *data ,unsigned int comp_type,unsigned int comp_level,unsigned int shuffle);
LIBRARY_API uint64_t hdf5_write_data_blosc_append(hid_t obj_id, hid_t type_id, const char *ds_name, void *data,hsize_t* num_2_add);
LIBRARY_API void hdf5_write_data_blosc_partial(hid_t obj_id, void* buff, const char* data_name,uint64_t elements_start,uint64_t elements_end);

LIBRARY_API void hdf5_create_dataset_blosc(hid_t obj_id, hid_t type_id, const char *ds_name, hsize_t rank, hsize_t *dims,unsigned int comp_type,unsigned int comp_level,unsigned int shuffle);

LIBRARY_API bool attribute_exists(hid_t obj_id,const char* attr_name);

LIBRARY_API bool group_exists(hid_t fileId,const char * attr_name);

LIBRARY_API bool data_exists(hid_t fileId,const char * attr_name);


#endif