
#include "parameters.h"

#include <cmath>
#include <iostream>
#include <fstream>

std::string get_path(std::string PATH_ENV){
    //
    //
    //  Gets library paths from environment variables (if run through make files, these are done through make file or command line), if done with a IDE, you need to set them up locally
    //
    //

    char const* tmp = getenv( PATH_ENV.c_str() );

    if ( tmp == NULL ) {
        std::cout << "***WARNING**** Please set the " << PATH_ENV << " environment variable" << std::endl;
        std::string path;
        return path;
    } else {
        std::string path( tmp );
        return path;
    }


}
void get_test_paths(std::string& image_path,std::string& utest_path,std::string& output_path){
    //
    //  Gets the path stored in set_paths.txt
    //
    //  Bevan Cheeseman 2015
    //
    //  File shoudl be ordered, witht he utest path in the first line then image path in second
    //

    //get the paths; need to be set a environment variables

    image_path = get_path("PARTGEN_IMAGE_PATH");
    utest_path = get_path("PARTGEN_UTEST_PATH");
    output_path = get_path("PARTGEN_OUTPUT_PATH");



}
bool get_image_stats(Proc_par& pars,std::string output_path,std::string stats_name){
    //
    //  Gets the image parameters for the specific file to be run
    //
    //  Bevan Cheeseman 2016
    //
    //
    std::cout << output_path + stats_name << std::endl;

    bool correct_file = true;

    //open the files
    std::ifstream path_file;
    path_file.open (output_path + stats_name);

    std::string out_line;

    std::size_t found;

    //get the paths

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("name: ");

    if (found!=std::string::npos){

        pars.name = out_line.substr(found+6);
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("dx: ");

    if (found!=std::string::npos){

        pars.dx = stof(out_line.substr(found+4));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("dy: ");

    if (found!=std::string::npos){

        pars.dy = stof(out_line.substr(found+4));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("dz: ");

    if (found!=std::string::npos){

        pars.dz = stof(out_line.substr(found+4));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("xdim: ");

    if (found!=std::string::npos){

        pars.xdim = stof(out_line.substr(found+6));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("ydim: ");

    if (found!=std::string::npos){

        pars.ydim = stof(out_line.substr(found + 6));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("zdim: ");

    if (found!=std::string::npos){

        pars.zdim = stof(out_line.substr(found + 6));
    } else {


    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("psfx: ");

    if (found!=std::string::npos){

        pars.psfx = stof(out_line.substr(found + 6));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("psfy: ");

    if (found!=std::string::npos){

        pars.psfy = stof(out_line.substr(found+6));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("psfz: ");

    if (found!=std::string::npos){

        pars.psfz = stof(out_line.substr(found+6));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("noise_sigma: ");

    if (found!=std::string::npos){

        pars.noise_sigma = stof(out_line.substr(found+13));
    } else {



    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("min_signal: ");

    if (found!=std::string::npos){

        pars.var_th = stof(out_line.substr(found+12));
    } else {

        std::cout << "Setting file incomplete" << std::endl;

        correct_file = false;

    }
    
    //file name (relative path)
    std::getline(path_file,out_line);
    
    found = out_line.find("rel_error: ");
    
    if (found!=std::string::npos){
        
        pars.rel_error = stof(out_line.substr(found+11));
    } else {
        correct_file = false;
    }
    
    //file name (relative path)
    std::getline(path_file,out_line);
    
    float lambda_or=0;
    
    found = out_line.find("lambda: ");
    
    if (found!=std::string::npos){
        
        lambda_or = stof(out_line.substr(found+8));
    } else {
        correct_file = false;
    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("I_th: ");

    if (found!=std::string::npos){

        pars.I_th = stof(out_line.substr(found+6));
    } else {

    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("var_th_max: ");

    if (found!=std::string::npos){

        pars.var_th_max = stof(out_line.substr(found+12));
    } else {
        pars.var_th_max = 0;
    }

    //file name (relative path)
    std::getline(path_file,out_line);

    found = out_line.find("aniso: ");

    if (found!=std::string::npos){

        pars.aniso = stof(out_line.substr(found+7));
    } else {
        pars.aniso = 1.0;
    }

    pars.tol = 0.0005f;

    pars.var_scale = 2;

    float k_diff = -3.0f;

    //set lambda
    //float lambda = expf((-1.0f/0.6161f) * logf((pars.var_th/pars.noise_sigma) *
                 //  powf(2.0f,k_diff + log2f(pars.rel_error))/0.12531f));
    pars.lambda = expf((-1.0f/0.6161f) * logf((pars.var_th/pars.noise_sigma) *
                                                    powf(2.0f,k_diff + log2f(.05))/0.12531f));
    //float lambda = expf((-1.0f/0.6161f) * logf((pars.var_th/pars.noise_sigma)));

    float lambda_min = 0.1f;
    float lambda_max = 5000;

    pars.lambda = std::max(lambda_min,pars.lambda);
    pars.lambda = std::min(pars.lambda,lambda_max);
    
    if(lambda_or >0){
        pars.lambda = lambda_or;
    }
    
    //std::cout << "Lamda: " << lambda << std::endl;
    std::cout << "Lamda: " << pars.lambda << std::endl;
    std::cout << "I_th: " << pars.I_th << std::endl;
    std::cout << "Rel Error: " << pars.rel_error << std::endl;
    
    //float max_var_th = 1.2f * pars.noise_sigma * expf(-0.5138f * logf(pars.lambda)) *
      //                 (0.1821f * logf(pars.lambda)+ 1.522f);

    if(pars.var_th_max == 0){
        pars.var_th_max = 0.75*pars.var_th;

    }

    std::cout << "Var Th: " << pars.var_th << std::endl;
    std::cout << "Var Th Max: " << pars.var_th_max << std::endl;
    std::cout << "Z factor: " << pars.z_factor << std::endl;


    return correct_file;

//    if (max_var_th > .25f*pars.var_th){
//        float desired_th = 0.1f*pars.var_th;
//        pars.lambda = std::max((float)exp((-1.0/0.5138) * log(desired_th/pars.noise_sigma)),pars.lambda);
//        pars.var_th_max = .25f*pars.var_th;
//    } else {
//        pars.var_th_max = max_var_th;
//    }

}
void get_image_parameters(Proc_par& pars,std::string output_path,std::string image_name){
    //
    //  Gets the image parameters for the specific file to be run
    //
    //  Bevan Cheeseman 2015
    //
    //
    std::cout << output_path + image_name + "_par.txt" << std::endl;


    //open the files
    std::ifstream path_file;
    path_file.open (output_path + image_name + "_par.txt");

    std::string out_line;

    //get the paths

    //file name (relative path)
    std::getline(path_file,out_line);
    pars.name = out_line;

    //window mean length
    std::getline(path_file,out_line);
    pars.window_mean = stoi(out_line);

    //window smooth length
    std::getline(path_file,out_line);
    pars.window_smooth =  stoi(out_line);

    //window smooth length
    std::getline(path_file,out_line);
    pars.window_var =  stoi(out_line);

    //number points in gradient stencil
    std::getline(path_file,out_line);
    pars.num_p_grad =  stoi(out_line);

    //gradient rescaling
    std::getline(path_file,out_line);
    pars.grad_h = stof(out_line);

    //intensity threshold
    std::getline(path_file,out_line);
    pars.I_th =  stof(out_line);

    //parameter scaling
    std::getline(path_file,out_line);
    pars.E0 = stof(out_line);

    //z scaling anisotropy
    std::getline(path_file,out_line);
    pars.z_factor = stof(out_line);

    //minimum variance threshold
    std::getline(path_file,out_line);
    pars.var_th = stof(out_line);

    //float type indicator
    std::getline(path_file,out_line);
    pars.type_float = stoi(out_line);

    //max_I scaling indicator
    std::getline(path_file,out_line);
    pars.max_I = stof(out_line);

    //max_I scaling indicator
    std::getline(path_file,out_line);
    pars.min_I = stof(out_line);


}
void setup_output_folder(Proc_par& pars){
    //
    //  Bevan Cheeseman 2015
    //
    //  Set up folder for data output
    //
    //

    pars.data_path = pars.output_path  + pars.name;

    std::string system_call = "mkdir " + pars.data_path;

    std::cout << pars.output_path << std::endl;

    if(system(system_call.c_str()) == -1){
        std::cout << "System call failed!" << std::endl;
    }

    pars.data_path = pars.output_path  + pars.name + "/";

}
void get_file_names(std::vector<std::string>& file_names,std::string file_list_path){
    //
    //
    //creates a list of files from
    //
    //


    ////open the files
    std::ifstream path_file;
    path_file.open (file_list_path);

    std::string str;

    while (std::getline(path_file, str)) {
        file_names.push_back(str);
    }



}
