///////////////////////////////////////////
//
//  ImageGen 2016
//
//  Bevan Cheeseman 2016
//
//  Header file with some specific hdf5 functions
//
//////////////////////////////////////////

#ifndef PARTPLAY_HDF5FUNCTIONS_H
#define PARTPLAY_HDF5FUNCTIONS_H

#include "hdf5.h"
#include <string>

void hdf5_create_file(std::string file_name);
void hdf5_load_data(hid_t obj_id,hid_t data_type,void* buff, const char* data_name);
void hdf5_write_attribute(hid_t obj_id,hid_t type_id,const char* attr_name,hsize_t rank,hsize_t* dims, void* data );
void hdf5_write_data(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data );
void hdf5_write_string(hid_t obj_id,const char* attr_name,std::string output_str);

template <typename T>
hid_t get_type(T data_type){
    //
    //  Return the type id
    //
    //
    
    int num_byts = sizeof(data_type);
    
    
    if (num_byts == 0) {
        return H5T_NATIVE_INT;
    } else if(num_byts == 2){
        return H5T_NATIVE_UINT16;
        
    } else if(num_byts == 1){
        return H5T_NATIVE_UINT8;
    } else {
        
        return H5T_NATIVE_FLOAT;
    }
    
    
}

#endif