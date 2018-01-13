//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Examples of simple iteration an access to Particle Cell, and particle information. (See Example_neigh, for neighbor access)
///
/// Usage:
///
/// (using output of Example_get_apr)
///
/// Example_apr_iterate -i input_image_tiff -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <benchmarks/development/old_numerics/filter_numerics.hpp>

#include "benchmarks/development/Example_newstructures.h"

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
        std::cerr << "Usage: \"Example_apr_iterate -i input_apr_file -d directory\"" << std::endl;
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



int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APR_timer timer;

    timer.verbose_flag = false;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    apr.parameters.input_dir = options.directory;

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    apr.write_apr_paraview(options.directory,name,apr.particles_int);

    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information

    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_cell_keys;

    uint64_t y_coord;

    std::vector<std::vector<uint16_t>> y_gaps;

    y_gaps.resize(apr.depth_max());

    ExtraPartCellData<uint64_t> gaps;
    gaps.initialize_structure_parts_empty(apr.particles_int);

    ExtraPartCellData<uint64_t> gaps_end;
    gaps_end.initialize_structure_parts_empty(apr.particles_int);


    ExtraPartCellData<uint64_t> index;
    index.initialize_structure_parts_empty(apr.particles_int);

    uint64_t count_gaps=0;
    uint64_t count_parts = 0;

    for(uint64_t i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = apr.pc_data.x_num[i];
        const unsigned int z_num_ = apr.pc_data.z_num[i];

//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            apr.pc_data.pc_key_set_z(curr_key,z_);
            apr.pc_data.pc_key_set_depth(curr_key,i);

            for(x_ = 0;x_ < x_num_;x_++){

                apr.pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = apr.pc_data.data[i][offset_pc_data].size();

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    float part_int= 0;

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = apr.pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        y_coord++;
                        count_parts++;

                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node

                        if(j_>0){
                            gaps_end.data[i][offset_pc_data].push_back(y_coord);
                        }

                        y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                        if(j_num > 1) {
                            if(j_ < (j_num - 1)) {
                                count_gaps++;
                                gaps.data[i][offset_pc_data].push_back(y_coord+1);
                                index.data[i][offset_pc_data].push_back(j_+1);
                            }

                        }


                    }

                }

            }

        }
    }

    std::cout << count_gaps << std::endl;
    std::cout << count_parts << std::endl;

}

