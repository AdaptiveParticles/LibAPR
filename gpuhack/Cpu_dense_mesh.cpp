//
// Created by cheesema on 05.03.18.
//
//
// Created by cheesema on 28.02.18.
//

//
// Created by cheesema on 28.02.18.
//

//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(


)";


#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"

#include "algorithm/APRConverter.hpp"
#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include <numerics/APRNumerics.hpp>
#include <numerics/APRComputeHelper.hpp>

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

template<typename U,typename V>
float pixels_linear_neighbour_access(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats);



int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);


    float time = pixels_linear_neighbour_access<uint16_t,uint16_t>(apr.spatial_index_y_max(apr.level_max()),apr.spatial_index_x_max(apr.level_max()),apr.spatial_index_z_max(apr.level_max()),1);

    std::cout << "Mesh neighbour sum " << time << std::endl;


}


bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
        std::cerr << usage << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;

}





template<typename U,typename V>
float pixels_linear_neighbour_access(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    MeshData<U> input_data;
    MeshData<V> output_data;
    input_data.init((int)y_num,(int)x_num,(int)z_num,23);
    output_data.init((int)y_num,(int)x_num,(int)z_num,0);

    APRTimer timer;
    timer.verbose_flag = false;
    timer.start_timer("full pixel neighbour access");

    int j = 0;
    int k = 0;
    int i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    //float neigh_sum = 0;

    for(int r = 0;r < num_repeats;r++){

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n)
#endif
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                for(k = 0;k < y_num;k++){
                    U neigh_sum = 0;
                    U counter = 0;

                    for(int  d  = 0;d < 6;d++){

                        i_n = i + dir_x[d];
                        k_n = k + dir_y[d];
                        j_n = j + dir_z[d];

                        //check boundary conditions
                        if((i_n >=0) & (i_n < x_num) ){
                            if((j_n >=0) & (j_n < z_num) ){
                                if((k_n >=0) & (k_n < y_num) ){
                                    neigh_sum += input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                                    counter++;
                                }
                            }
                        }
                    }

                    output_data.mesh[j*x_num*y_num + i*y_num + k] = neigh_sum/(counter*1.0);

                }
            }
        }

    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds/num_repeats;

    std::cout << "Pixel Linear Neigh: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    std::cout << "per 1000000 pixel took: " << (time)/((1.0*x_num*y_num*z_num)/1000000.0) << std::endl;



    return (time)/((1.0*x_num*y_num*z_num)/1000000.0);

}
