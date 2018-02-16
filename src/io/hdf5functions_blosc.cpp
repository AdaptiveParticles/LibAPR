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
void register_blosc(){
    register_blosc(nullptr, nullptr);
}

/**
 * reads data from hdf5
 */
void hdf5_load_data_blosc(hid_t obj_id, hid_t data_type, void* buff, const char* data_name) {
    hid_t data_id =  H5Dopen2(obj_id, data_name ,H5P_DEFAULT);
    H5Dread(data_id, data_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff);
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

/**
 * writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
 */
void hdf5_write_attribute_blosc(hid_t obj_id,hid_t type_id,const char* attr_name,hsize_t rank,hsize_t* dims, const void * const data ){
    hid_t space_id = H5Screate_simple(rank, dims, NULL);
    hid_t attr_id = H5Acreate2( obj_id, attr_name, type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, type_id, data);
    H5Aclose(attr_id);
    H5Sclose(space_id);
}

/**
 * Writes string information as an attribute
 */
void hdf5_write_string_blosc(hid_t obj_id, const char *attr_name, const std::string &output_str) {
    if (output_str.size() > 0){
        hid_t aid = H5Screate(H5S_SCALAR);
        hid_t atype = H5Tcopy (H5T_C_S1);
        H5Tset_size(atype, output_str.size());
        hid_t attr = H5Acreate2(obj_id, attr_name, atype, aid, H5P_DEFAULT,H5P_DEFAULT);
        H5Awrite (attr, atype,output_str.c_str());
        H5Tclose(atype);
        H5Aclose(attr);
        H5Sclose(aid);
    }
}

/**
 * creates the hdf5 file before you can then write to it
 */
hid_t hdf5_create_file_blosc(std::string file_name){
    return H5Fcreate(file_name.c_str(),H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); //this writes over the current file
}

void write_main_paraview_xdmf_xml(std::string save_loc,std::string file_name,uint64_t num_parts){
    const std::string hdf5_file_name = file_name + ".h5";
    std::ofstream myfile(save_loc + file_name + ".xmf");
    myfile << "<?xml version=\"1.0\" ?>\n";
    myfile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    myfile << "<Xdmf Version=\"2.0\" xmlns:xi=\"[http://www.w3.org/2001/XInclude]\">\n";
    myfile <<  " <Domain>\n";
    myfile <<  "   <Grid Name=\"parts\" GridType=\"Uniform\">\n";
    myfile <<  "     <Topology TopologyType=\"Polyvertex\" Dimensions=\"" << num_parts << "\"/>\n";
    myfile <<  "     <Geometry GeometryType=\"X_Y_Z\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/x\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/y\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/z\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "     </Geometry>\n";
    myfile <<  "     <Attribute Name=\"particle property\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/particle property\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "     <Attribute Name=\"level\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/level\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "     <Attribute Name=\"type\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/type\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "   </Grid>\n";
    myfile <<  " </Domain>\n";
    myfile <<  "</Xdmf>\n";
    myfile.close();
}
