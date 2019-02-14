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


/**
 * Register the 'blosc' filter with the HDF5 library
 */
void hdf5_register_blosc(){
    register_blosc(nullptr, nullptr);
}

/**
 * reads data from hdf5
 */
void hdf5_load_data_blosc(hid_t obj_id, hid_t dataType, void* buff, const char* data_name) {
    hid_t data_id =  H5Dopen2(obj_id, data_name ,H5P_DEFAULT);
    H5Dread(data_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff);
    H5Dclose(data_id);
}

/**
 * reads data from hdf5 (data type auto-detection)
 */
void hdf5_load_data_blosc(hid_t obj_id, void* buff, const char* data_name) {
    hid_t data_id =  H5Dopen2(obj_id, data_name ,H5P_DEFAULT);
    hid_t dataType = H5Dget_type(data_id);
    H5Dread(data_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff);
    H5Tclose(dataType);
    H5Dclose(data_id);
}


/**
 * reads only a set number of elements
 */
void hdf5_load_data_blosc_partial(hid_t obj_id, void* buff, const char* data_name,uint64_t elements_start,uint64_t elements_end) {
    hid_t data_id =  H5Dopen2(obj_id, data_name ,H5P_DEFAULT);

    hid_t memspace_id=0;
    hid_t dataspace_id=0;

    hid_t dataType = H5Dget_type(data_id);

    hsize_t dims = elements_end - elements_start;

    memspace_id = H5Screate_simple (1, &dims, NULL);

    hsize_t offset = elements_start;
    hsize_t count = dims;
    hsize_t stride = 1;
    hsize_t block = 1;

    dataspace_id = H5Dget_space (data_id);
    H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, &offset,
                                  &stride, &count, &block);

    H5Dread(data_id, dataType, memspace_id, dataspace_id, H5P_DEFAULT, buff);

    H5Sclose (memspace_id);
    H5Sclose (dataspace_id);

    H5Tclose(dataType);
    H5Dclose(data_id);
}



/**
 * writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
 */
void hdf5_write_data_blosc(hid_t obj_id, hid_t type_id, const char *ds_name, hsize_t rank, hsize_t *dims, void *data ,unsigned int comp_type,unsigned int comp_level,unsigned int shuffle) {
    hid_t plist_id  = H5Pcreate(H5P_DATASET_CREATE);

    // Dataset must be chunked for compression
    const uint64_t max_size = 100000;
    hsize_t cdims = (dims[0] < max_size) ? dims[0] : max_size;
    rank = 1;
    H5Pset_chunk(plist_id, rank, &cdims);

    /////SET COMPRESSION TYPE /////
    // But you can also taylor Blosc parameters to your needs
    // 0 to 3 (inclusive) param slots are reserved.
    const int numOfParams = 7;
    unsigned int cd_values[numOfParams];
    cd_values[4] = comp_level; // compression level
    cd_values[5] = shuffle;    // 0: shuffle not active, 1: shuffle active
    cd_values[6] = comp_type;  // the actual compressor to use
    H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, numOfParams, cd_values);

    //create write and close
    hid_t space_id = H5Screate_simple(rank, dims, NULL);
    hid_t dset_id = H5Dcreate2(obj_id, ds_name, type_id, space_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
    H5Dclose(dset_id);

    H5Pclose(plist_id);
}



void
hdf5_write_data_blosc_create(hid_t obj_id, hid_t type_id, const char *ds_name, hsize_t rank, hsize_t *dims, void *data,
                             unsigned int comp_type, unsigned int comp_level, unsigned int shuffle) {
    hid_t plist_id  = H5Pcreate(H5P_DATASET_CREATE);

    // Dataset must be chunked for compression
    const uint64_t max_size = 100000;
    hsize_t cdims = (dims[0] < max_size) ? dims[0] : max_size;
    cdims = max_size;
    rank = 1;
    H5Pset_chunk(plist_id, rank, &cdims);

    /////SET COMPRESSION TYPE /////
    // But you can also taylor Blosc parameters to your needs
    // 0 to 3 (inclusive) param slots are reserved.
    const int numOfParams = 7;
    unsigned int cd_values[numOfParams];
    cd_values[4] = comp_level; // compression level
    cd_values[5] = shuffle;    // 0: shuffle not active, 1: shuffle active
    cd_values[6] = comp_type;  // the actual compressor to use
    H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, numOfParams, cd_values);

    hsize_t max_dims = H5S_UNLIMITED;

    //create write and close
    hid_t space_id = H5Screate_simple(rank, dims, &max_dims);
    hid_t dset_id = H5Dcreate2(obj_id, ds_name, type_id, space_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);

    H5Sclose (space_id);
    H5Dclose(dset_id);

    H5Pclose(plist_id);
}




uint64_t hdf5_write_data_blosc_append(hid_t obj_id, hid_t type_id, const char *ds_name, void *data, hsize_t* num_2_add) {
    //
    //  Appends data to the end of an existing dataset
    //

    hid_t dset_id =  H5Dopen2(obj_id, ds_name ,H5P_DEFAULT);


    hid_t dspace = H5Dget_space(dset_id);

    hsize_t current_dim;
    H5Sget_simple_extent_dims(dspace, &current_dim, NULL);

    hsize_t new_dim = current_dim+ num_2_add[0];

    H5Dset_extent(dset_id, &new_dim);

    hid_t memspace_id=0;
    hid_t dataspace_id=0;

    hid_t dataType = H5Dget_type(dset_id);

    //extends the current file
    H5Dset_extent(dset_id,&new_dim);
    memspace_id = H5Screate_simple (1, num_2_add, NULL);

    hsize_t offset = current_dim;
    hsize_t count = num_2_add[0];
    hsize_t stride = 1;
    hsize_t block = 1;

    dataspace_id = H5Dget_space (dset_id);
    H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, &offset,
                         NULL, &count, NULL);
    if(new_dim > 0) {

        //H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, buffer);
        H5Dwrite(dset_id, type_id, memspace_id, dataspace_id, H5P_DEFAULT, data);

    }

    H5Sclose (memspace_id);
    H5Sclose (dataspace_id);
    H5Sclose (dspace);

    H5Tclose(dataType);
    H5Dclose(dset_id);

    return new_dim;

//    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
//    H5Dclose(dset_id);

}


/**
 * writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type without using blosc
 */
void hdf5_write_data_standard(hid_t obj_id, hid_t type_id, const char *ds_name, hsize_t rank, hsize_t *dims, void *data) {
    hid_t plist_id  = H5Pcreate(H5P_DATASET_CREATE);

    // Dataset must be chunked for compression
    const uint64_t max_size = 100000;
    hsize_t cdims = (dims[0] < max_size) ? dims[0] : max_size;
    rank = 1;
    H5Pset_chunk(plist_id, rank, &cdims);

    //compression parameters
    int deflate_level = 9;

    /////SET COMPRESSION TYPE /////

    //DEFLATE ENCODING (GZIP)
    H5Pset_deflate (plist_id, deflate_level);

    //create write and close
    hid_t space_id = H5Screate_simple(rank, dims, NULL);
    hid_t dset_id = H5Dcreate2(obj_id, ds_name, type_id, space_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
    H5Dclose(dset_id);

    H5Pclose(plist_id);

}


/**
 * writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
 */

bool attribute_exists(hid_t obj_id,const char* attr_name){
    /* Save old error handler */
    herr_t (*old_func)(void*);
    void *old_client_data;
    H5Eget_auto1(&old_func, &old_client_data);

    hid_t attr_id;

    /* Turn off error handling */
    H5Eset_auto1(NULL, NULL);
    attr_id = H5Aopen_name(obj_id,attr_name);

    /* Restore previous error handler */
    H5Eset_auto1(old_func, old_client_data);

    bool exists = (bool)(attr_id > 0);

    //if the attribute is not being used, we need to close it.
    if(exists){
        H5Aclose(attr_id);
    }

    return exists;
}

void hdf5_write_attribute_blosc(hid_t obj_id,hid_t type_id,const char* attr_name,hsize_t rank,hsize_t* dims, const void * const data ){
    hid_t exists;

    herr_t (*old_func)(void*);
    void *old_client_data;
    H5Eget_auto1(&old_func, &old_client_data);

    /* Turn off error handling */
    H5Eset_auto1(NULL, NULL);
    exists = H5Aopen_name(obj_id,attr_name);

    /* Restore previous error handler */
    H5Eset_auto1(old_func, old_client_data);
    //does it already exist
    //if it already exists then over-write

    if(exists<0) {
        hid_t attr_id;
        //doesn't exist
        hid_t space_id = H5Screate_simple(rank, dims, NULL);
        attr_id = H5Acreate2(obj_id, attr_name, type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr_id, type_id, data);
        H5Aclose(attr_id);
        H5Sclose(space_id);
    } else {
        //
        H5Awrite(exists, type_id, data);
        H5Aclose(exists);
    }


}

/**
 * creates the hdf5 file before you can then write to it
 */
hid_t hdf5_create_file_blosc(std::string file_name){

    hid_t fcpl_id=H5Pcreate(H5P_FILE_CREATE);
    //H5Pset_file_space(fcpl_id,H5F_FILE_SPACE_ALL_PERSIST,(hsize_t)0);

    //return H5Fcreate(file_name.c_str(),H5F_ACC_TRUNC, fcpl_id, H5P_DEFAULT); //this writes over the current file
    return H5Fcreate(file_name.c_str(),H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); //this writes over the current file
}

void write_main_paraview_xdmf_xml(const std::string &aDestinationDir,const std::string &aHdf5FileName, const std::string &aParaviewFileName, uint64_t aNumOfParticles){
    std::ofstream myfile(aDestinationDir + aParaviewFileName + "_paraview.xmf");
    myfile << "<?xml version=\"1.0\" ?>\n";
    myfile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    myfile << "<Xdmf Version=\"2.0\" xmlns:xi=\"[http://www.w3.org/2001/XInclude]\">\n";
    myfile <<  " <Domain>\n";
    myfile <<  "   <Grid Name=\"parts\" GridType=\"Uniform\">\n";
    myfile <<  "     <Topology TopologyType=\"Polyvertex\" Dimensions=\"" << aNumOfParticles << "\"/>\n";
    myfile <<  "     <Geometry GeometryType=\"X_Y_Z\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< aNumOfParticles <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << aHdf5FileName << ":/ParticleRepr/t/x\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< aNumOfParticles <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << aHdf5FileName << ":/ParticleRepr/t/y\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< aNumOfParticles <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << aHdf5FileName << ":/ParticleRepr/t/z\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "     </Geometry>\n";
    myfile <<  "     <Attribute Name=\"particle property\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< aNumOfParticles <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << aHdf5FileName << ":/ParticleRepr/t/particle property\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "     <Attribute Name=\"level\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< aNumOfParticles <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
    myfile <<  "        " << aHdf5FileName << ":/ParticleRepr/t/level\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
//    myfile <<  "     <Attribute Name=\"type\" AttributeType=\"Scalar\" Center=\"Node\">\n";
//    myfile <<  "       <DataItem Dimensions=\""<< aNumOfParticles <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
//    myfile <<  "        " << aHdf5FileName << ":/ParticleRepr/t/type\n";
//    myfile <<  "       </DataItem>\n";
//    myfile <<  "    </Attribute>\n";
    myfile <<  "   </Grid>\n";
    myfile <<  " </Domain>\n";
    myfile <<  "</Xdmf>\n";
    myfile.close();
}
void write_main_paraview_xdmf_xml_time(const std::string &aDestinationDir,const std::string &aHdf5FileName, const std::string &aParaviewFileName, std::vector<uint64_t> aNumOfParticles){
    std::ofstream myfile(aDestinationDir + aParaviewFileName + "_paraview.xmf");
    myfile << "<?xml version=\"1.0\" ?>\n";
    myfile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    myfile << "<Xdmf Version=\"2.0\" xmlns:xi=\"[http://www.w3.org/2001/XInclude]\">\n";
    myfile <<  " <Domain>\n";
    myfile << " <Grid Name=\"partsTime\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

    for (int time_step = 0; time_step < aNumOfParticles.size(); ++time_step) {

        std::string t_string;
        if(time_step==0){
            t_string = "t";
        } else{
            t_string = "t" +std::to_string(time_step);
        }

        myfile << "   <Grid Name=\"parts\" GridType=\"Uniform\">\n";
        myfile << "     <Time Value=\"" << time_step << "\" />\n";
        myfile << "     <Topology TopologyType=\"Polyvertex\" Dimensions=\"" << aNumOfParticles[time_step] << "\"/>\n";
        myfile << "     <Geometry GeometryType=\"X_Y_Z\">\n";
        myfile << "       <DataItem Dimensions=\"" << aNumOfParticles[time_step] << "\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
        myfile << "        " << aHdf5FileName << ":/ParticleRepr/" << t_string <<"/x\n";
        myfile << "       </DataItem>\n";
        myfile << "       <DataItem Dimensions=\"" << aNumOfParticles[time_step]
               << "\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
        myfile << "        " << aHdf5FileName << ":/ParticleRepr/" << t_string <<"/y\n";
        myfile << "       </DataItem>\n";
        myfile << "       <DataItem Dimensions=\"" << aNumOfParticles[time_step]
               << "\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
        myfile << "        " << aHdf5FileName << ":/ParticleRepr/" << t_string <<"/z\n";
        myfile << "       </DataItem>\n";
        myfile << "     </Geometry>\n";
        myfile << "     <Attribute Name=\"particle property\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        myfile << "       <DataItem Dimensions=\"" << aNumOfParticles[time_step]
               << "\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
        myfile << "        " << aHdf5FileName << ":/ParticleRepr/" << t_string <<"/particle property\n";
        myfile << "       </DataItem>\n";
        myfile << "    </Attribute>\n";
        myfile << "     <Attribute Name=\"level\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        myfile << "       <DataItem Dimensions=\"" << aNumOfParticles[time_step]
               << "\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
        myfile << "        " << aHdf5FileName << ":/ParticleRepr/" << t_string <<"/level\n";
        myfile << "       </DataItem>\n";
        myfile << "    </Attribute>\n";
        myfile << "   </Grid>\n";
    }


    myfile <<  "   </Grid>\n";
    myfile <<  " </Domain>\n";
    myfile <<  "</Xdmf>\n";
    myfile.close();
}


