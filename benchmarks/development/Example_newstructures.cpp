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


#define _NO_NEIGHBOUR ((uint16_t)3)
#define _LEVEL_SAME ((uint16_t)1)
#define _LEVEL_DECREASE ((uint16_t)0)
#define _LEVEL_INCREASE ((uint16_t)2)

#define _EMPTY ((uint16_t)0)
#define _SEED ((uint16_t)1)
#define _BOUNDARY ((uint16_t)2)
#define _FILLER ((uint16_t)3)

#define YP_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 1
#define YP_LEVEL_SHIFT (uint16_t)  2

#define YM_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 3
#define YM_LEVEL_SHIFT (uint16_t) 3

#define XP_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 5
#define XP_LEVEL_SHIFT 5

#define XM_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 7
#define XM_LEVEL_SHIFT 7

#define ZP_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 9
#define ZP_LEVEL_SHIFT 9

#define ZM_LEVEL_MASK ((((uint16_t)1) << 2) - 1) << 11
#define ZM_LEVEL_SHIFT 11

#define PC_TYPE_MASK ((((uint16_t)1) << 2) - 1) << 13
#define PC_TYPE_SHIFT 13




//xp is x + 1 neigh
//#define XP_DEPTH_MASK ((((uint64_t)1) << 2) - 1) << 4
//#define XP_DEPTH_SHIFT 4
//#define XP_INDEX_MASK ((((uint64_t)1) << 13) - 1) << 6
//#define XP_INDEX_SHIFT 6



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

    std::vector<uint16_t> neighbours;

    apr.get_part_numbers();

    neighbours.resize(apr.num_parts_total);

    ExtraPartCellData<uint64_t> gaps;
    gaps.initialize_structure_parts_empty(apr.particles_int);

    ExtraPartCellData<uint64_t> gaps_end;
    gaps_end.initialize_structure_parts_empty(apr.particles_int);

    ExtraPartCellData<uint64_t> index;
    index.initialize_structure_parts_empty(apr.particles_int);

    ExtraPartCellData<uint64_t> iterator;
    iterator.initialize_structure_parts_empty(apr.particles_int);

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

                if(j_num > 1){
                    iterator.data[i][offset_pc_data].push_back(0);
                }

                uint64_t prev = 0;

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    float part_int= 0;

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = apr.pc_data.data[i][offset_pc_data][j_];



                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        y_coord++;
                        count_parts++;

                        uint16_t status = (node_val_pc & STATUS_MASK) >> STATUS_SHIFT;
                        uint16_t type = (node_val_pc & TYPE_MASK) >> TYPE_SHIFT;

                        uint16_t xp_j= (node_val_pc & XP_INDEX_MASK) >> XP_INDEX_SHIFT;
                        uint16_t xp_dep = (node_val_pc & XP_DEPTH_MASK) >> XP_DEPTH_SHIFT;

                        uint16_t zp_j = (node_val_pc & ZP_INDEX_MASK) >> ZP_INDEX_SHIFT;
                        uint16_t zp_dep = (node_val_pc & ZP_DEPTH_MASK) >> ZP_DEPTH_SHIFT;

                        uint16_t m_j = (node_val_pc & XM_INDEX_MASK) >> XM_INDEX_SHIFT;
                        uint16_t xm_dep = (node_val_pc & XM_DEPTH_MASK) >> XM_DEPTH_SHIFT;

                        uint16_t zm_j = (node_val_pc & ZM_INDEX_MASK) >> ZM_INDEX_SHIFT;
                        uint16_t zm_dep = (node_val_pc & ZM_DEPTH_MASK) >> ZM_DEPTH_SHIFT;

//
//                        pc_data.data[i][offset_pc_data][curr_index-1] = TYPE_GAP;
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= (((uint64_t)y_) << NEXT_COORD_SHIFT);
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= ( prev_coord << PREV_COORD_SHIFT);
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YP_DEPTH_SHIFT);
//                        pc_data.data[i][offset_pc_data][curr_index-1] |= (NO_NEIGHBOUR << YM_DEPTH_SHIFT);

                        neighbours[count_parts-1] |= (xp_dep << XP_LEVEL_SHIFT);
                        neighbours[count_parts-1] |= (xm_dep << XM_LEVEL_SHIFT);
                        neighbours[count_parts-1] |= (zp_dep << ZP_LEVEL_SHIFT);
                        neighbours[count_parts-1] |= (zm_dep << ZM_LEVEL_SHIFT);
                        neighbours[count_parts-1] |= (status << PC_TYPE_SHIFT);

                        if(prev == 0){
                            //add a y same flag

                            neighbours[count_parts-1] |= (_LEVEL_SAME << YM_LEVEL_SHIFT);

                            neighbours[count_parts-2] |= (_LEVEL_SAME << YP_LEVEL_SHIFT);

                        }

                        //NEED TO SET YP





                        prev = 0;



                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node

                        prev = 1;

                        uint64_t type = (node_val_pc & TYPE_MASK) >> TYPE_SHIFT;

                        uint64_t yp_j = (node_val_pc & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
                        uint64_t yp_dep = (node_val_pc & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;

                        uint64_t ym_j = (node_val_pc & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
                        uint64_t ym_dep = (node_val_pc & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;

                        uint64_t next_y = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;

                        uint64_t prev_y = (node_val_pc & PREV_COORD_MASK) >> PREV_COORD_SHIFT;

                        if((j_ == 0) & (j_num > 1)){
                            //first node (do forward) (YM)
                            neighbours[count_parts] |= (ym_dep << YM_LEVEL_SHIFT);

                        } else if (j_ == (j_num-1) & (j_num > 1)){
                            //last node (do behind) (YP)
                            neighbours[count_parts-1] |= (yp_dep << YP_LEVEL_SHIFT);

                        } else if (j_num > 1){
                            // front (YM) and behind (YP)

                            neighbours[count_parts] |= (ym_dep << YM_LEVEL_SHIFT);
                            neighbours[count_parts-1] |= (yp_dep << YP_LEVEL_SHIFT);

                        }


                        if(j_>0){
                            gaps_end.data[i][offset_pc_data].push_back(y_coord);
                        }

                        y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                        if(j_num > 1) {
                            if(j_ < (j_num - 1)) {
                                count_gaps++;
                                gaps.data[i][offset_pc_data].push_back(y_coord+1);
                                index.data[i][offset_pc_data].push_back(count_parts);
                            }

                        }


                    }

                }

            }

        }
    }

    std::cout << count_gaps << std::endl;
    std::cout << count_parts << std::endl;

    std::vector<uint16_t> pint;
    pint.reserve(count_parts);

    std::vector<uint16_t> px;
    px.reserve(count_parts);

    std::vector<uint16_t> py;
    py.reserve(count_parts);

    std::vector<uint16_t> pz;
    pz.reserve(count_parts);

    timer.verbose_flag = true;

    timer.start_timer("iterate old");

    for (apr.begin();apr.end()!=0 ;apr.it_forward()) {
        pint.push_back(apr(apr.particles_int));
        px.push_back(apr.x());
        py.push_back(apr.y());
        pz.push_back(apr.z());
    }

    timer.stop_timer();

    std::vector<uint16_t> pint2;
    pint2.reserve(count_parts);

    std::vector<uint16_t> px2;
    px2.reserve(count_parts);

    std::vector<uint16_t> py2;
    py2.reserve(count_parts);

    std::vector<uint16_t> pz2;
    pz2.reserve(count_parts);


    timer.start_timer("iterate new");

    uint64_t counter_new = -1;

    for(uint64_t i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max;i++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = apr.pc_data.x_num[i];
        const unsigned int z_num_ = apr.pc_data.z_num[i];

//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                const size_t offset_pc_data = x_num_*z_ + x_;

                if(iterator.data[i][offset_pc_data].size() > 0){

                    for (int j = 0; j < gaps.data[i][offset_pc_data].size(); ++j) {

                        uint64_t curr_index = index.data[i][offset_pc_data][j];

                        uint64_t begin = gaps.data[i][offset_pc_data][j];
                        uint64_t end = gaps_end.data[i][offset_pc_data][j];

                        curr_index--;

                        for (int y = gaps.data[i][offset_pc_data][j];
                             y <= gaps_end.data[i][offset_pc_data][j]; y++) {

                            curr_index++;
                            counter_new++;

                            pint2.push_back(pint[curr_index]);
                            px2.push_back(x_);
                            pz2.push_back(z_);
                            py2.push_back(y);

                        }

                    }

                }
            }
        }
    }

    timer.stop_timer();

    std::cout << counter_new << std::endl;

    /////////////////
    ///
    /// Checking everything is okay here..
    ///
    ///////////////////

    for (int k = 0; k < count_parts; ++k) {

        if(pint[k] != pint2[k]){
            std::cout << "broke" << std::endl;
        }

        if(py[k] != py2[k]){
            std::cout << "broke" << std::endl;
        }

        if(px[k] != px2[k]){
            std::cout << "broke" << std::endl;
        }

        if(pz[k] != pz2[k]){
            std::cout << "broke" << std::endl;
        }

    }





//    apr.write_particles_only(options.directory,name+"gaps",gaps);
//    apr.write_particles_only(options.directory,name+"gaps_end",gaps_end);
//    apr.write_particles_only(options.directory,name+"index",index);
//    apr.write_particles_only(options.directory,name+"iterator",iterator);


}
