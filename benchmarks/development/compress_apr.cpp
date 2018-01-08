#include <algorithm>
#include <iostream>

#include "compress_apr.h"
#include "src/data_structures/Mesh/meshclass.h"
#include "src/io/readimage.h"

#include "benchmarks/development/old_algorithm/gradient.hpp"
#include "benchmarks/development/old_structures/particle_map.hpp"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "benchmarks/development/old_algorithm/level.hpp"
#include "src/io/writeimage.h"
#include "src/io/write_parts.h"
#include "src/io/partcell_io.h"
#include "src/numerics/apr_compression.hpp"
#include "test/utils.h"

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
    
    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;
    
    // Filename
    std::string file_name = options.directory + options.input;
    
    part_rep.timer.start_timer("read_pc_struct");
    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);
    
    part_rep.timer.stop_timer();
    
    part_rep.timer.start_timer("write_wavelet");
    
    write_apr_wavelet<float,int8_t>(pc_struct,options.directory,"wavelet_test",3);
    
    part_rep.timer.stop_timer();
    
    part_rep.timer.start_timer("write");
    
    write_apr_pc_struct(pc_struct,options.directory,"standard");
    
    part_rep.timer.stop_timer();
    
    part_rep.timer.start_timer("read_wavelet");
    
    file_name = options.directory + "wavelet_test_pcstruct_part.h5";
    
    // APR data structure
    PartCellStructure<float,uint64_t> wavelet_struct;
    
    read_apr_wavelet<float,int8_t>(wavelet_struct,file_name);
    
    part_rep.timer.stop_timer();
    
    //write_apr_pc_struct(wavelet_struct,options.directory,"standard");
    
    write_apr_full_format(wavelet_struct,options.directory,options.output);
    
    // comapre
    
    //compare_two_structures_test(pc_struct,wavelet_struct);
}


