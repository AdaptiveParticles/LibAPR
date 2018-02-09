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
#include <array>
#include <string>
#include <memory>

// TODO: This is not supported for UWP, find a replacement
#ifdef _WINDOWS
#define popen _popen
#define pclose _pclose
#endif

std::string exec_blosc(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}

void register_bosc(){
    /* Register the filter with the library */
    char *version, *date;
    register_blosc(&version, &date);
    //printf("Blosc version info: %s (%s)\n", version, date);
}

void hdf5_load_data_blosc(hid_t obj_id,hid_t data_type,void* buff, const char* data_name){
    
    //declare var
    hid_t data_id;
    
    //stuff required to pull the data in
    data_id =  H5Dopen2(obj_id , data_name ,H5P_DEFAULT);
    hid_t datatype  = H5Dget_type(data_id);
    
    H5Dread( data_id, datatype, H5S_ALL, H5S_ALL,H5P_DEFAULT, buff );
    H5Dclose(data_id);
};

void hdf5_write_data_blosc(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data ){
    //writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
    
    unsigned int cd_values[7];
    //Declare the required hdf5 shiz
    hid_t space_id,dset_id,plist_id;
    hsize_t *cdims = new hsize_t[rank]; //chunking dims
    
    //compression parameters

    
    //int szip_options_mask = H5_SZIP_NN_OPTION_MASK;
    //int szip_pixels_per_block = 8;
    
    //dataspace id
    space_id = H5Screate_simple(rank, dims, NULL);
    plist_id  = H5Pcreate(H5P_DATASET_CREATE);
    
    /* Dataset must be chunked for compression */
    //cdims[0] = 20; //Could try playing with these for compression performance
    //cdims[1] = 20;
    
    int max_size = 100000;
    
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
    cd_values[4] = 6;       /* compression level */
    cd_values[5] = 2;       /* 0: shuffle not active, 1: shuffle active */
    cd_values[6] = BLOSC_ZSTD; /* the actual compressor to use */

    /* Set the filter with 7 params */
    H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 7, cd_values);
    
    //create write and close
    dset_id = H5Dcreate2(obj_id,ds_name,type_id,space_id,H5P_DEFAULT,plist_id,H5P_DEFAULT);
    
    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
    
    H5Dclose(dset_id);
    
    
};
void hdf5_write_data_blosc(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data ,unsigned int comp_type,unsigned int comp_level,unsigned int shuffle){
    //writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
    
    unsigned int cd_values[7];
    //Declare the required hdf5 shiz
    hid_t space_id,dset_id,plist_id;
    //hsize_t *cdims = new hsize_t[rank]; //chunking dims
    hsize_t cdims[1]; //chunking dims
    
    //compression parameters
    
    
    //int szip_options_mask = H5_SZIP_NN_OPTION_MASK;
    //int szip_pixels_per_block = 8;
    rank = 1;
    //dataspace id
    space_id = H5Screate_simple(rank, dims, NULL);
    plist_id  = H5Pcreate(H5P_DATASET_CREATE);
    
    /* Dataset must be chunked for compression */
    //cdims[0] = 20; //Could try playing with these for compression performance
    //cdims[1] = 20;
    
    int max_size = 100000;
    

    if (dims[0] < max_size){
        cdims[0] = dims[0];
    }else {
        cdims[0] = max_size;
    }


    
    H5Pset_chunk(plist_id, rank, cdims);
    
    /////SET COMPRESSION TYPE /////
    
    /* But you can also taylor Blosc parameters to your needs */
    /* 0 to 3 (inclusive) param slots are reserved. */
    cd_values[4] = comp_level;       /* compression level */
    cd_values[5] = shuffle;       /* 0: shuffle not active, 1: shuffle active */
    cd_values[6] = comp_type; /* the actual compressor to use */
    
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
    hid_t space_id, attr_id;
    //hsize_t *cdims = new hsize_t[rank]; //chunking dims
    
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
    
    hid_t fid; //file id
    
    //fid = H5F.create(name,'H5F_ACC_EXCL', 'H5P_DEFAULT', 'H5P_DEFAULT'); %create the file (throws error if it already exists)
    fid = H5Fcreate(file_name.c_str(),H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); //this writes over the current file
    
    //close shiz
    H5Fclose(fid);
    
};
void write_main_paraview_xdmf_xml(std::string save_loc,std::string file_name,uint64_t num_parts){
    //
    //
    //
    //

    std::string hdf5_file_name = file_name + ".h5";
    std::string xdmf_file_name = save_loc + file_name + ".xmf";


    std::ofstream myfile;
    myfile.open (xdmf_file_name);

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
void write_paraview_xdmf_xml_extra(std::string save_loc,std::string file_name,int num_parts,std::vector<std::string> extra_data_names,std::vector<std::string> extra_data_types){
    //
    //
    //  Also writes out extra datafields
    //


    std::string hdf5_file_name = file_name + ".h5";
    std::string xdmf_file_name = save_loc + file_name + ".xmf";


    std::ofstream myfile;
    myfile.open (xdmf_file_name);

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
    myfile <<  "     <Attribute Name=\"Ip\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/Ip\n";
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

    for (int i = 0; i < extra_data_names.size(); i++) {
        //adds the additional datasets

        std::string num_type;
        std::string prec;


        if (extra_data_types[i]=="uint8_t") {
            num_type = "UInt";
            prec = "1";
        } else if (extra_data_types[i]=="bool"){
            std::cout << "Bool type can't be stored with xdmf change datatype" << std::endl;
        } else if (extra_data_types[i]=="uint16_t"){
            num_type = "UInt";
            prec = "2";
        } else if (extra_data_types[i]=="int16_t"){
            num_type = "Int";
            prec = "2";

        } else if (extra_data_types[i]=="int"){
            num_type = "Int";
            prec = "4";
        }  else if (extra_data_types[i]=="int8_t"){
            num_type = "Int";
            prec = "1";
        }else if (extra_data_types[i]=="float"){
            num_type = "Float";
            prec = "4";
        } else {
            std::cout << "Unknown Type in extra printout encournterd" << std::endl;

        }

        myfile <<  "     <Attribute Name=\""<< extra_data_names[i] <<"\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"" << num_type << "\" Precision=\"" << prec <<"\" Format=\"HDF\">\n";
        myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/" <<  extra_data_names[i] << "\n";
        myfile <<  "       </DataItem>\n";
        myfile <<  "    </Attribute>\n";

    }

    myfile <<  "   </Grid>\n";
    myfile <<  " </Domain>\n";
    myfile <<  "</Xdmf>\n";

    myfile.close();

}