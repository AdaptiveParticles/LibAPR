#include <algorithm>
#include <iostream>

#include "compute_cells.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellBase.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
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
        std::cerr << "Usage: \"pipeline -i inputfile -d directory [-t] [-o outputfile]\"" << std::endl;
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
    PartCellStructure<uint16_t,uint64_t> pc_struct;
    
    //output
    std::string file_name = options.directory + options.input;
    
    read_apr_pc_struct(pc_struct,file_name);
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t j_;
    uint64_t curr_key = 0;
    PartCellNeigh<uint64_t> neigh_keys;
    
    //
    //
    //  Get neighbour loop
    //
    //
    
    part_rep.timer.start_timer("Loop over cells and get neighbours");
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,curr_key) firstprivate(neigh_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val&1)){
                        //get the index gap node
                        
                        pc_struct.pc_data.pc_key_set_j(curr_key,j_);
                        
                        pc_struct.pc_data.get_neighs_all(curr_key,node_val,neigh_keys);
                        
                        
                    } else {
                        
                    }
                    
                }
                
            }
            
        }
    }
    
    std::cout << "Finished Neigh Cell test" << std::endl;
    
    
    part_rep.timer.stop_timer();
    
}


