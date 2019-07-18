//
// Created by cheesema on 24/01/17.
//

#ifndef PARTPLAY_ANALYSISDATA_HPP
#define PARTPLAY_ANALYSISDATA_HPP

#include "DataManager.hpp"
#include "misc/APRTimer.hpp"
#include "io/hdf5functions_blosc.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <ctime>
#include <fstream>

std::string exec(const char* cmd);

inline void hdf5_write_data_blosc2(hid_t obj_id,hid_t type_id,const char* ds_name,hsize_t rank,hsize_t* dims, void* data,unsigned int comp_type,unsigned int comp_level,unsigned int shuffle ){
    //writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type

    //Declare the required hdf5 shiz
    hid_t space_id,dset_id,plist_id;
    hsize_t *cdims = new hsize_t[rank]; //chunking dims

    //compression parameters
    int deflate_level = 9;

    //dataspace id
    space_id = H5Screate_simple(rank, dims, NULL);
    plist_id  = H5Pcreate(H5P_DATASET_CREATE);

    unsigned int max_size = 200000;

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

    H5Pclose(plist_id);

    H5Sclose(space_id);
}

struct AnalysisFile {
    enum class Operation {READ, WRITE};
    hid_t fileId = -1;

    AnalysisFile(const std::string &aFileName, const Operation aOp) {
        switch(aOp) {
            case Operation::READ:
                fileId = H5Fopen(aFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                if (fileId == -1) {
                    std::cerr << "Could not open file [" << aFileName << "]" << std::endl;
                    return;
                }
                break;
            case Operation::WRITE:
                fileId = hdf5_create_file_blosc(aFileName);
                if (fileId == -1) {
                    std::cerr << "Could not create file [" << aFileName << "]" << std::endl;
                    return;
                }
                break;
        }
    }
    ~AnalysisFile() {
        if (fileId != -1) H5Fclose(fileId);
    }

    /**
     * Is File opened?
     */
    bool isOpened() const { return fileId != -1; }

    hsize_t getFileSize() const {
        hsize_t size;
        H5Fget_filesize(fileId, &size);
        return size;
    }
};

class AnalysisData: public Data_manager {
    //
    //  Bevan Cheeseman 2016
    //
    //  General information and data to be outputted for analysis and comparison
    //
    //


    protected:


    public:

    std::string name; //used for labelling the file
    std::string description; //used for understanding what was being done

    std::string file_name;


    AnalysisData(){

        // current date/time based on current system
        time_t now = time(0);

        // convert now to string form
        std::string dt = ctime(&now);

        create_string_dataset("Date", 0);
        get_data_ref<std::string>("Date")->data.push_back(dt);
        part_data_list["Date"].print_flag = true;

        time_t timer;
        struct tm y2k = {0};
        double seconds;

        y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
        y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

        time(&timer);  /* get current time; same as: timer = time(NULL)  */

        seconds = difftime(timer,mktime(&y2k));


        file_name = name + std::to_string((uint64_t)seconds);

        get_git_version();
    }

    void init(std::string name_,std::string description_,int argc, char **argv)
    {

        name = name_;
        description = description_;

        // current date/time based on current system
        time_t now = time(0);

        // convert now to string form
        std::string dt = ctime(&now);

        create_string_dataset("Date", 0);
        get_data_ref<std::string>("Date")->data.push_back(dt);
        part_data_list["Date"].print_flag = true;

        create_string_dataset("Name", 0);
        get_data_ref<std::string>("Name")->data.push_back(name);
        part_data_list["Name"].print_flag = true;

        create_string_dataset("Description", 0);
        get_data_ref<std::string>("Description")->data.push_back(description);
        part_data_list["Description"].print_flag = true;


        time_t timer;
        struct tm y2k = {0};
        double seconds;

        y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
        y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

        time(&timer);  /* get current time; same as: timer = time(NULL)  */

        seconds = difftime(timer,mktime(&y2k));


        file_name = name + std::to_string((uint64_t)seconds);

        //get the current git version
        get_git_version();

        std::vector<std::string> arguments(argv + 1, argv + argc);

        //add the command line arguments
        for (unsigned int i = 0; i < arguments.size(); ++i) {
            create_string_dataset("arg_" + std::to_string(i),0);
            get_data_ref<std::string>("arg_" + std::to_string(i))->data.push_back(arguments[i]);
            part_data_list["arg_" + std::to_string(i)].print_flag = true;
        }


    };

    void add_float_data(std::string name,float value){

        //first check if its first run and the variables need to be set up
        Part_data<float> *check_ref = get_data_ref<float>(name);

        if (check_ref == nullptr) {
            // First experiment need to set up the variables

            //pipeline parameters
            create_float_dataset(name, 0);
        }

        get_data_ref<float>(name)->data.push_back(value);
        part_data_list[name].print_flag = true;

    }

    void add_string_data(std::string name, std::string value){

        //first check if its first run and the variables need to be set up
        Part_data<float> *check_ref = get_data_ref<float>(name);

        if (check_ref == nullptr) {
            // First experiment need to set up the variables

            //pipeline parameters
            create_string_dataset(name, 0);
        }

        get_data_ref<std::string>(name)->data.push_back(value);
        part_data_list[name].print_flag = true;

    }

    void add_timer(APRTimer& timer){

        //set up timing variables

        for (int i = 0; i < (int) timer.timings.size(); i++) {
            add_float_data(timer.timing_names[i],timer.timings[i]);
        }

    }

    void add_timer(APRTimer& timer,float number_particles,float num_reps){

        //set up timing variables

        for (int i = 0; i < (int) timer.timings.size(); i++) {
            add_float_data(timer.timing_names[i] + "_total",timer.timings[i]/(num_reps));
            add_float_data(timer.timing_names[i] + "_pp",timer.timings[i]/(number_particles*num_reps));
        }

    }


    void add_apr_info(APR& apr){

        add_float_data("num_parts",apr.total_number_particles());
        add_float_data("num_tree_parts",apr.total_number_tree_particles());
        add_float_data("CR",apr.computational_ratio());
        add_float_data("num_pixels",apr.org_dims(0)*apr.org_dims(1)*apr.org_dims(2));
        add_float_data("org_image_MB",(apr.org_dims(0)*apr.org_dims(1)*apr.org_dims(2)*2)/(1000000000.0f));
    }

    //writes the results to hdf5
    void write_analysis_data_hdf5();

    void get_git_version();

};

inline void hdf5_write_string_blosc(hid_t obj_id,const char* attr_name,std::string output_str){
    //
    //  Writes string information as an attribute
    //
    //

    hid_t       aid, atype, attr;

    aid = H5Screate(H5S_SCALAR);

    atype = H5Tcopy (H5T_C_S1);

    if (output_str.size() > 0){

        H5Tset_size (atype, output_str.size());

        attr = H5Acreate2(obj_id, attr_name, atype, aid, H5P_DEFAULT,H5P_DEFAULT);

        H5Awrite (attr, atype,output_str.c_str());
    }

}

///**
// * writes data to the hdf5 file or group identified by obj_id of hdf5 datatype data_type
// */
//inline void hdf5_write_data_blosc(hid_t obj_id, hid_t type_id, const char *ds_name, hsize_t rank, hsize_t *dims, void *data ,unsigned int comp_type,unsigned int comp_level,unsigned int shuffle) {
//    hid_t plist_id  = H5Pcreate(H5P_DATASET_CREATE);
//
//    // Dataset must be chunked for compression
//    const uint64_t max_size = 100000;
//    hsize_t cdims = (dims[0] < max_size) ? dims[0] : max_size;
//    rank = 1;
//    H5Pset_chunk(plist_id, rank, &cdims);
//
//    /////SET COMPRESSION TYPE /////
//    // But you can also taylor Blosc parameters to your needs
//    // 0 to 3 (inclusive) param slots are reserved.
//    const int numOfParams = 7;
//    unsigned int cd_values[numOfParams];
//    cd_values[4] = comp_level; // compression level
//    cd_values[5] = shuffle;    // 0: shuffle not active, 1: shuffle active
//    cd_values[6] = comp_type;  // the actual compressor to use
//    H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, numOfParams, cd_values);
//
//    //create write and close
//    hid_t space_id = H5Screate_simple(rank, dims, NULL);
//    hid_t dset_id = H5Dcreate2(obj_id, ds_name, type_id, space_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
//    H5Dwrite(dset_id,type_id,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
//    H5Dclose(dset_id);
//
//    H5Pclose(plist_id);
//}

inline void write_part_data_to_hdf5(Data_manager& p_rep,hid_t obj_id,std::vector<std::string>& extra_data_type,std::vector<std::string>& extra_data_name,int flag_type,unsigned int req_size){
    //
    //  Bevan Cheeseman 2016
    //
    //  This function writes part_data data types to hdf5 files, according to them satisfying the flag and data length requirements
    //
    //  flag_type: allows to set local part_data variables for printing; or deciding which set to prinrt;
    //
    //  req_size: allows you to determine if you want only a specific type of data printed out; which is requried if you want a valid xdmf file
    //
    //


    hsize_t dims;
    hsize_t rank = 1;

    for(auto const &part_data_info : p_rep.part_data_list) {
        if (part_data_info.second.print_flag == flag_type) {
            // add the name

            if (part_data_info.second.data_type=="uint8_t") {

                Part_data<uint8_t>* data_pointer = p_rep.get_data_ref<uint8_t>(part_data_info.first);
                dims = data_pointer->data.size();
                //check if it is particle data of cell data
                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {

                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);



                    hdf5_write_data_blosc2(obj_id,H5T_NATIVE_UINT8,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data(), BLOSC_ZSTD, 1, 2);
                }

            } else if (part_data_info.second.data_type=="bool"){
                std::cout << "Bool type can't be stored with xdmf change datatype" << std::endl;
            } else if (part_data_info.second.data_type=="uint16_t"){

                Part_data<uint16_t>* data_pointer = p_rep.get_data_ref<uint16_t>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data_blosc2(obj_id,H5T_NATIVE_UINT16,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data(), BLOSC_ZSTD, 1, 2 );
                }

            }else if (part_data_info.second.data_type=="int16_t"){

                Part_data<int16_t>* data_pointer = p_rep.get_data_ref<int16_t>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data_blosc2(obj_id,H5T_NATIVE_INT16,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data(), BLOSC_ZSTD, 1, 2 );
                }

            }  else if (part_data_info.second.data_type=="int"){

                Part_data<int>* data_pointer = p_rep.get_data_ref<int>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data_blosc2(obj_id,H5T_NATIVE_INT,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data(), BLOSC_ZSTD, 1, 2 );

                }
            } else if (part_data_info.second.data_type=="float"){
                Part_data<float>* data_pointer = p_rep.get_data_ref<float>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data_blosc2(obj_id,H5T_NATIVE_FLOAT,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data(), BLOSC_ZSTD, 1, 2 );
                }

            } else if (part_data_info.second.data_type=="int8_t") {

                Part_data<int8_t>* data_pointer = p_rep.get_data_ref<int8_t>(part_data_info.first);
                dims = data_pointer->data.size();
                //check if it is particle data of cell data
                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {

                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);


                    hdf5_write_data_blosc2(obj_id,H5T_NATIVE_INT8,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data(), BLOSC_ZSTD, 1, 2  );
                }

            } else if (part_data_info.second.data_type=="string") {
                //ONLY WRITES TEH FIRST STRING!!##!$

                Part_data<std::string>* data_pointer = p_rep.get_data_ref<std::string>(part_data_info.first);
                dims = 1;
                //check if it is particle data of cell data
                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {

                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_string_blosc(obj_id, part_data_info.first.c_str(), data_pointer->data[0]);

                }

            }else {
                std::cout << "Unknown Type in extra printout encournterd" << std::endl;

            }
        }
    }

}


inline void AnalysisData::write_analysis_data_hdf5() {
    std::string save_loc = "";
    std::string hdf5_file_name = save_loc + file_name + ".h5";

    AnalysisFile f(hdf5_file_name, AnalysisFile::Operation::WRITE);

    //create the main group
    hid_t pr_groupid = H5Gcreate2(f.fileId, "Analysis_data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5G_info_t info;
    H5Gget_info(pr_groupid, &info);

    std::vector<std::string> extra_data_type;
    std::vector<std::string> extra_data_name;

    int req_size = 0;
    int flag_type = 1;
    write_part_data_to_hdf5(*this, pr_groupid, extra_data_type, extra_data_name, flag_type, req_size);

    //close shiz
    H5Gclose(pr_groupid);

    std::cout << "Data Analysis File Writing Complete to [" << hdf5_file_name << "] file" << std::endl;
    clearAll();
}
/*
static long long GetFileSize(std::string filename)
{
    std::ifstream mySource;
    mySource.open(filename, std::ios_base::binary);
    mySource.seekg(0,std::ios_base::end);
    long long size = mySource.tellg();
    mySource.close();
    return size;
}
*/
#ifdef _MSC_VER
#define popen _popen
#define pclose _pclose
#endif


inline std::string exec(const char* cmd) {
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

inline void AnalysisData::get_git_version(){
    //
    //  Get the git hash
    //
    //

    std::string git_hash = exec("git rev-parse HEAD");

    create_string_dataset("git_hash",0);
    get_data_ref<std::string>("git_hash")->data.push_back(git_hash);
    part_data_list["git_hash"].print_flag = true;

}

#endif //PARTPLAY_ANALYSISDATA_HPP
