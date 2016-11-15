///////////////////////////////////////////
//
//  ImageGen 2016
//
//  Bevan Cheeseman 2016
//
//  Header file with some specific hdf5 functions
//
//////////////////////////////////////////

#include "hdf5functions_blosc.h"

void register_bosc(){
    
    char *version, *date;
    int r, i;
    
    /* Register the filter with the library */
    r = register_blosc(&version, &date);
    printf("Blosc version info: %s (%s)\n", version, date);
    
    
}

void hdf5_load_data_blosc(hid_t obj_id,hid_t data_type,void* buff, const char* data_name){
    
    //declare var
    hid_t data_id;
    
    //stuff required to pull the data in
    data_id =  H5Dopen2(obj_id , data_name ,H5P_DEFAULT);
    
    
    H5Dread( data_id, data_type, H5S_ALL, H5S_ALL,H5P_DEFAULT, buff );
    H5Dclose(data_id);
};

void hdf5_write_data_blosc(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data ){
    //writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
    
    unsigned int cd_values[7];
    //Declare the required hdf5 shiz
    hid_t space_id,dset_id,plist_id;
    hsize_t cdims[rank]; //chunking dims
    
    //compression parameters
    int deflate_level = 9;
    
    //int szip_options_mask = H5_SZIP_NN_OPTION_MASK;
    //int szip_pixels_per_block = 8;
    
    //dataspace id
    space_id = H5Screate_simple(rank, dims, NULL);
    plist_id  = H5Pcreate(H5P_DATASET_CREATE);
    
    /* Dataset must be chunked for compression */
    //cdims[0] = 20; //Could try playing with these for compression performance
    //cdims[1] = 20;
    
    int max_size = 200000;
    
    if (rank == 1) {
        if (dims[0] < max_size){
            cdims[0] = dims[0];
        }else {
            cdims[0] = max_size;
        }
    }
    else {
        cdims[0] = 100;
        cdims[1] = 100;
    }
    
    H5Pset_chunk(plist_id, rank, cdims);
    
    /////SET COMPRESSION TYPE /////
    
    /* But you can also taylor Blosc parameters to your needs */
    /* 0 to 3 (inclusive) param slots are reserved. */
    cd_values[4] = 2;       /* compression level */
    cd_values[5] = 1;       /* 0: shuffle not active, 1: shuffle active */
    cd_values[6] = BLOSC_ZSTD; /* the actual compressor to use */
    
    /* Set the filter with 7 params */
    H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 7, cd_values);
    
    //create write and close
    dset_id = H5Dcreate2(obj_id,ds_name,type_id,space_id,H5P_DEFAULT,plist_id,H5P_DEFAULT);
    
    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
    
    H5Dclose(dset_id);
    
    
};

void hdf5_write_attribute_blosc(hid_t obj_id,hid_t type_id,const char* attr_name,hsize_t rank,hsize_t* dims, void* data ){
    //writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
    
    //Declare the required hdf5 shiz
    hid_t space_id,dset_id,attr_id;
    hsize_t cdims[rank]; //chunking dims
    
    space_id = H5Screate_simple(rank, dims, NULL);
    //plist_id  = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    
    attr_id = H5Acreate2( obj_id, attr_name, type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Awrite(attr_id, type_id, data );
    H5Aclose(attr_id);
    
};
void hdf5_write_string_blosc(hid_t obj_id,const char* attr_name,std::string output_str){
    //
    //  Writes string information as an attribute
    //
    //
    
    hid_t       aid, atype, attr;
    herr_t      status;
    
    aid = H5Screate(H5S_SCALAR);
    
    atype = H5Tcopy (H5T_C_S1);
    
    if (output_str.size() > 0){
        
        status = H5Tset_size (atype, output_str.size());
        
        attr = H5Acreate2(obj_id, attr_name, atype, aid, H5P_DEFAULT,H5P_DEFAULT);
        
        status = H5Awrite (attr, atype,output_str.c_str());
    }
    
}





void hdf5_create_file_blosc(std::string file_name){
    //creates the hdf5 file before you can then write to it
    
    hid_t fid,pr_groupid; //file id
    
    //fid = H5F.create(name,'H5F_ACC_EXCL', 'H5P_DEFAULT', 'H5P_DEFAULT'); %create the file (throws error if it already exists)
    fid = H5Fcreate(file_name.c_str(),H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); //this writes over the current file
    
    //close shiz
    H5Fclose(fid);
    
};
