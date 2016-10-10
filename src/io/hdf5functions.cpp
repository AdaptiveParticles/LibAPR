///////////////////////////////////////////
//
//  ImageGen 2016
//
//  Bevan Cheeseman 2016
//
//  Header file with some specific hdf5 functions
//
//////////////////////////////////////////

#include "hdf5functions.h"

void hdf5_load_data(hid_t obj_id,hid_t data_type,void* buff, const char* data_name){

    //declare var
    hid_t data_id;

    //stuff required to pull the data in
    data_id =  H5Dopen2(obj_id , data_name ,H5P_DEFAULT);


    H5Dread( data_id, data_type, H5S_ALL, H5S_ALL,H5P_DEFAULT, buff );
    H5Dclose(data_id);
};

void hdf5_write_data(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data ){
    //writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type

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

    //DEFLATE ENCODING (GZIP)
    H5Pset_deflate (plist_id, deflate_level);

    // Uncomment these lines to set SZIP Compression
    // H5Pset_szip (plist_id, szip_options_mask, szip_pixels_per_block);
    //
    //////////////////////////////

    //create write and close
    dset_id = H5Dcreate2(obj_id,ds_name,type_id,space_id,H5P_DEFAULT,plist_id,H5P_DEFAULT);

    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);

    H5Dclose(dset_id);


};

void hdf5_write_attribute(hid_t obj_id,hid_t type_id,const char* attr_name,hsize_t rank,hsize_t* dims, void* data ){
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
void hdf5_write_string(hid_t obj_id,const char* attr_name,std::string output_str){
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





void hdf5_create_file(std::string file_name){
    //creates the hdf5 file before you can then write to it

    hid_t fid,pr_groupid; //file id

    //fid = H5F.create(name,'H5F_ACC_EXCL', 'H5P_DEFAULT', 'H5P_DEFAULT'); %create the file (throws error if it already exists)
    fid = H5Fcreate(file_name.c_str(),H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); //this writes over the current file

    //close shiz
    H5Fclose(fid);

};
