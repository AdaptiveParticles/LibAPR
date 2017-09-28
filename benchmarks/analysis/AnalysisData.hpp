//
// Created by cheesema on 24/01/17.
//

#ifndef PARTPLAY_ANALYSISDATA_HPP
#define PARTPLAY_ANALYSISDATA_HPP

#include "../../src/data_structures/structure_parts.h"
#include "../../src/io/parameters.h"
#include "../../src/io/hdf5functions.h"
#include "../../src/io/write_parts.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <ctime>

std::string exec(const char* cmd);


class AnalysisData: public Data_manager{
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

    bool quality_metrics_gt;
    bool quality_metrics_input;
    bool file_size;
    bool information_content;
    bool segmentation_mesh;
    bool segmentation_parts;
    bool filters_parts;
    bool filters_mesh;
    bool debug;
    bool segmentation_eval;
    bool filters_eval;
    bool quality_true_int;
    bool check_scale;
    bool comp_perfect;

    AnalysisData(){

        // current date/time based on current system
        time_t now = time(0);

        // convert now to string form
        std::string dt = ctime(&now);

        create_string_dataset("Date", 0);
        get_data_ref<std::string>("Date")->data.push_back(dt);
        part_data_list["Date"].print_flag = true;

        init_proc_parameter_data();

        quality_metrics_gt = false;
        quality_metrics_input = false;
        information_content = false;
        file_size = false;
        segmentation_parts = false;
        filters_parts = false;
        segmentation_mesh = false;
        filters_mesh = false;
        debug = false;
        segmentation_eval = false;
        filters_eval = false;

        quality_true_int = false;
        check_scale = false;
        comp_perfect = false;

        time_t timer;
        struct tm y2k = {0};
        double seconds;

        y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
        y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

        time(&timer);  /* get current time; same as: timer = time(NULL)  */

        seconds = difftime(timer,mktime(&y2k));


        file_name = name + std::to_string((uint64)seconds);

        get_git_version();
    }

    AnalysisData(std::string name,std::string description,int argc, char **argv): Data_manager(),name(name),description(description)
    {

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


        init_proc_parameter_data();

        quality_metrics_gt = false;
        quality_metrics_input = false;
        information_content = false;
        file_size = false;

        segmentation_parts = false;
        filters_parts = false;
        filters_mesh = false;
        segmentation_mesh = false;
        segmentation_eval = false;
        filters_eval = false;
        quality_true_int = false;
        check_scale = false;

        comp_perfect = false;

        debug=false;

        time_t timer;
        struct tm y2k = {0};
        double seconds;

        y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
        y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

        time(&timer);  /* get current time; same as: timer = time(NULL)  */

        seconds = difftime(timer,mktime(&y2k));


        file_name = name + std::to_string((uint64)seconds);

        //get the current git version
        get_git_version();

        std::vector<std::string> arguments(argv + 1, argv + argc);

        //add the command line arguments
        for (int i = 0; i < arguments.size(); ++i) {
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

    void add_timer(Part_timer& timer){

        //set up timing variables

        for (int i = 0; i < timer.timings.size(); i++) {
            add_float_data(timer.timing_names[i],timer.timings[i]);
        }

    }

    void init_proc_parameter_data(){
        //
        //  This method initializes part_data sets for the standard processing pipeline parameters
        //
        //

        //pipeline parameters
        create_int8_dataset("k_method",0);
        create_int8_dataset("grad_method",0);
        create_int8_dataset("var_method",0);
        create_int8_dataset("padd_flag",0);

        create_float_dataset("lambda",0);
        create_float_dataset("tol",0);
        create_float_dataset("var_scale",0);
        create_float_dataset("mean_scale",0);

        //data paths
        create_string_dataset("image_path",0);
        create_string_dataset("output_path",0);
        create_string_dataset("data_path",0);
        create_string_dataset("utest_path",0);

        //img parameters
        create_float_dataset("dy",0);
        create_float_dataset("dx",0);
        create_float_dataset("dz",0);

        create_float_dataset("psfy",0);
        create_float_dataset("psfx",0);
        create_float_dataset("psfz",0);

        create_float_dataset("ydim",0);
        create_float_dataset("xdim",0);
        create_float_dataset("zdim",0);

        create_float_dataset("noise_sigma",0);
        create_float_dataset("background",0);

    }

    void push_proc_par(Proc_par par){
        //
        //
        //

        //pipeline parameters
        get_data_ref<int8_t>("k_method")->data.push_back(par.k_method);
        part_data_list["k_method"].print_flag = true;
        get_data_ref<int8_t>("grad_method")->data.push_back(par.grad_method);
        part_data_list["grad_method"].print_flag = true;
        get_data_ref<int8_t>("var_method")->data.push_back(par.var_method);
        part_data_list["var_method"].print_flag = true;
        get_data_ref<int8_t>("padd_flag")->data.push_back(par.padd_flag);
        part_data_list["padd_flag"].print_flag = true;


        get_data_ref<float>("lambda")->data.push_back(par.lambda);
        part_data_list["lambda"].print_flag = true;

        get_data_ref<float>("tol")->data.push_back(par.tol);
        part_data_list["tol"].print_flag = true;

        get_data_ref<float>("var_scale")->data.push_back(par.var_scale);
        part_data_list["var_scale"].print_flag = true;

        get_data_ref<float>("mean_scale")->data.push_back(par.mean_scale);
        part_data_list["mean_scale"].print_flag = true;

        //data paths

        //strings are currently written as an attribute, and therefore only the first will be written

        get_data_ref<std::string>("image_path")->data.push_back(par.image_path);
        part_data_list["image_path"].print_flag = true;

        get_data_ref<std::string>("output_path")->data.push_back(par.output_path);
        part_data_list["output_path"].print_flag = true;

        get_data_ref<std::string>("data_path")->data.push_back(par.data_path);
        part_data_list["data_path"].print_flag = true;

        get_data_ref<std::string>("utest_path")->data.push_back(par.utest_path);
        part_data_list["utest_path"].print_flag = true;


        //img parameters
        get_data_ref<float>("dx")->data.push_back(par.dx);
        part_data_list["dx"].print_flag = true;

        get_data_ref<float>("dy")->data.push_back(par.dy);
        part_data_list["dy"].print_flag = true;

        get_data_ref<float>("dz")->data.push_back(par.dz);
        part_data_list["dz"].print_flag = true;

        get_data_ref<float>("psfy")->data.push_back(par.psfy);
        part_data_list["psfy"].print_flag = true;

        get_data_ref<float>("psfx")->data.push_back(par.psfx);
        part_data_list["psfx"].print_flag = true;

        get_data_ref<float>("psfz")->data.push_back(par.psfz);
        part_data_list["psfz"].print_flag = true;

        get_data_ref<float>("ydim")->data.push_back(par.ydim);
        part_data_list["ydim"].print_flag = true;

        get_data_ref<float>("xdim")->data.push_back(par.xdim);
        part_data_list["xdim"].print_flag = true;

        get_data_ref<float>("zdim")->data.push_back(par.zdim);
        part_data_list["zdim"].print_flag = true;

        get_data_ref<float>("noise_sigma")->data.push_back(par.noise_sigma);
        part_data_list["noise_sigma"].print_flag = true;


        get_data_ref<float>("background")->data.push_back(par.background);
        part_data_list["background"].print_flag = true;


    }

    //writes the results to hdf5
    void write_analysis_data_hdf5();

    void get_git_version();

};
void AnalysisData::write_analysis_data_hdf5(){

    std::string save_loc = get_path("ANALYSIS_DATA_PATH");

    std::string hdf5_file_name = save_loc + file_name + ".h5";

    hdf5_create_file(hdf5_file_name);

    //hdf5 inits
    hid_t fid, pr_groupid;
    H5G_info_t info;

    hsize_t dims;

    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;

    //create the main group
    pr_groupid = H5Gcreate2(fid,"Analysis_data",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );

    //////////////////////////////////////////////////////////////////
    //
    //  Write analysis data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////

    std::vector<std::string> extra_data_type;
    std::vector<std::string> extra_data_name;

    int req_size = 0;
    int flag_type = 1;

    write_part_data_to_hdf5(*this,pr_groupid,extra_data_type,extra_data_name,flag_type,req_size);

    //close shiz
    H5Gclose(pr_groupid);
    H5Fclose(fid);

    std::cout << "Data Analysis File Writing Complete" << std::endl;


}


long long GetFileSize(std::string filename);

long long GetFileSize(std::string filename)
{
    std::ifstream mySource;
    mySource.open(filename, std::ios_base::binary);
    mySource.seekg(0,std::ios_base::end);
    long long size = mySource.tellg();
    mySource.close();
    return size;
}


std::string exec(const char* cmd) {
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

void AnalysisData::get_git_version(){
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
