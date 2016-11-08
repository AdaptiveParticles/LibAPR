#include <algorithm>
#include <iostream>

#include "get_apr.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"

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

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep){
    
    cmdLineOptions result;
    
    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile [-t] [-s example_name -d stats_directory] [-o outputfile]\"" << std::endl;
        exit(1);
    }
    
    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    
    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }
    
    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.stats_directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-s"))
    {
        result.stats = std::string(get_command_option(argv, argv + argc, "-s"));
        get_image_stats(part_rep.pars, result.stats_directory, result.stats);
        result.stats_file = true;
    }
    if(command_option_exists(argv, argv + argc, "-l"))
    {
        part_rep.pars.lambda = (float)std::atof(get_command_option(argv, argv + argc, "-l"));
        if(part_rep.pars.lambda == 0.0){
            std::cerr << "Lambda can't be zero" << std::endl;
            exit(3);
        }
    }
    if(command_option_exists(argv, argv + argc, "-t"))
    {
        part_rep.timer.verbose_flag = true;
    }
    
    return result;
    
}


int main(int argc, char **argv) {
    
    Part_rep part_rep;
    
    // INPUT PARSING
    
    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);
    
    // COMPUTATIONS
    
    
    //output
    std::string save_loc = options.output;
    std::string file_name = options.stats;
    
    
    part_rep.timer.start_timer("writing output");
    
    read_apr_pc_struct(pc_read,save_loc + file_name + "_pcstruct_part.h5");
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t j_;
    uint64_t curr_key = 0;
    PartCellNeigh<uint64_t> neigh_keys;
    
    bool pass_test = true;
    
    //
    //
    //  Get neighbour loop
    //
    //
    
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num = pc_struct.pc_data.x_num[i];
        const unsigned int z_num = pc_struct.pc_data.z_num[i];
        
        
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key,neigh_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            pc_key_set_z(curr_key,z_);
            pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    node_val = data[i][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        //get the index gap node
                        
                        pc_key_set_j(curr_key,j_);
                        
                        get_neighs_all(curr_key,node_val,neigh_keys);
                        
                        
                    } else {
                        
                    }
                    
                }
                
            }
            
        }
    }
    
    std::cout << "Finished Neigh Cell test" << std::endl;
    
    
    part_rep.timer.stop_timer();
    
}


