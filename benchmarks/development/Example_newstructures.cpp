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
#define YP_LEVEL_SHIFT (uint16_t)  1

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

struct PartCell {
    uint16_t x,y,z,level,type;
    uint64_t pc_offset,global_index;
};

struct YGap {
    uint16_t y_begin;
    uint16_t y_end;
    uint64_t global_index_begin;
};

struct YGap_map {
    uint16_t y_end;
    uint64_t global_index_begin;
};

struct ParticleCellGapMap{
    std::map<uint16_t,YGap_map> map;
    std::map<uint16_t,YGap_map>::iterator current_iterator;
    //uint16_t last_value = 0;
};


struct GapIteratorMap {

    //

};


struct GapIterator {
    YGap current_gap;
    uint16_t y_min;
    uint16_t y_max;
    uint16_t current_gap_index;
    uint16_t gap_num;
};

bool get_neighbour_coordinate(const PartCell& input,PartCell& neigh,const unsigned int& face,const uint16_t& level_delta,const uint16_t& index){
    //
    //

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    constexpr uint8_t children_index_offsets[4][2] = {{0,0},{0,1},{1,0},{1,1}};

    unsigned int dir;

    switch (level_delta){
        case _LEVEL_SAME:
            //Same Level Particle Cell
            neigh.x = input.x + dir_x[face];
            neigh.y = input.y + dir_y[face];
            neigh.z = input.z + dir_z[face];
            neigh.level = input.level;

            return true;
        case _LEVEL_DECREASE:
            //Larger Particle Cell (Lower Level)
            neigh.level = input.level - 1;
            neigh.x = input.x/2;
            neigh.y = input.y/2;
            neigh.z = input.z/2;

            return true;
        case _LEVEL_INCREASE:
            //Higher Level Particle Cell (Smaller/Higher Resolution), there is a maximum of 4 (conditional on boundary conditions)
            neigh.level = input.level + 1;
            neigh.x = (input.x + dir_x[face])*2 + (dir_x[face]<0);
            neigh.y = (input.y + dir_y[face])*2 + (dir_y[face]<0);
            neigh.z = (input.z + dir_z[face])*2 + (dir_z[face]<0);

            dir = (index/2);

            switch (dir){
                case 0:
                    //y+ and y-
                    neigh.x = neigh.x + children_index_offsets[index][0];
                    neigh.z = neigh.z + children_index_offsets[index][1];

                    break;

                case 1:
                    //x+ and x-
                    neigh.y = neigh.y + children_index_offsets[index][0];
                    neigh.z = neigh.z + children_index_offsets[index][1];

                    break;
                case 2:
                    //z+ and z-
                    neigh.y = neigh.y + children_index_offsets[index][0];
                    neigh.x = neigh.x + children_index_offsets[index][1];

                    break;
            }

            return true;
        case _NO_NEIGHBOUR:

            return false;
    }

    return false;

}

uint8_t number_neighbours_in_direction(const uint8_t& level_delta){
    //
    //  Gives the maximum number of neighbours in a direction given the level_delta.
    //

    switch (level_delta){
        case _LEVEL_INCREASE:
            return 4;
        case _NO_NEIGHBOUR:
            return 0;
    }
    return 1;
}

bool find_particle_cell(ExtraPartCellData<ParticleCellGapMap>& gap_map,PartCell& part_cell){

    if(gap_map.data[part_cell.level][part_cell.pc_offset].size() > 0) {

        ParticleCellGapMap* current_pc_map = &gap_map.data[part_cell.level][part_cell.pc_offset][0];

        std::map<uint16_t,YGap_map>::iterator& map_it = (current_pc_map->current_iterator);
        //std::map<uint16_t,YGap_map>::iterator map_it = current_pc_map->map.begin();
        //std::advance (map_it,current_pc_map->last_value);

        if(map_it == current_pc_map->map.end()){
            //check if pointing to a valid key
            //map_it = current_pc_map->map.begin();

            map_it--;
        }

        if ((part_cell.y >= map_it->first) & (part_cell.y <= map_it->second.y_end)) {
            // already pointing to the correct place
            part_cell.global_index = map_it->second.global_index_begin +
                    (part_cell.y - map_it->first);

            //current_pc_map->last_value = std::distance(current_pc_map->map.begin(),map_it);

            return true;
        } else {
            //first try next element
            if(map_it != current_pc_map->map.end()){
                map_it++;
                //check if there
                if(map_it != current_pc_map->map.end()) {
                    if ((part_cell.y >= map_it->first) &
                        (part_cell.y <= map_it->second.y_end)) {
                        // already pointing to the correct place
                        part_cell.global_index = map_it->second.global_index_begin +
                                                 (part_cell.y - map_it->first);

                        //current_pc_map->last_value = std::distance(current_pc_map->map.begin(),map_it);
                        return true;
                    }
                }

            }

            //otherwise search for it (points to first key that is greater than the y value)
            map_it = current_pc_map->map.upper_bound(part_cell.y);

            if(map_it == current_pc_map->map.begin()){
                //less then the first value
                return false;
            } else{
                map_it--;
            }

            if ((part_cell.y >= map_it->first) & (part_cell.y <= map_it->second.y_end)) {
                // already pointing to the correct place
                part_cell.global_index = map_it->second.global_index_begin +
                                         (part_cell.y - map_it->first);

                //current_pc_map->last_value = std::distance(current_pc_map->map.begin(),map_it);
                return true;
            }
        }
    }

    return false;

}



bool find_particle_cell_global_index_forward_iteration(ExtraPartCellData<YGap>& ygaps,ExtraPartCellData<GapIterator>& gap_iterator,PartCell& part_cell){
    //
    //  Finds the global index for a part_cell if it exists
    //
//    if(gap_iterator.data[part_cell.level][part_cell.pc_offset].size() > 0) {
//        //GapIterator *current_it = &gap_iterator.data[part_cell.level][part_cell.pc_offset][0];
//
//        if ((part_cell.y >= gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_begin) & (part_cell.y <= gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_end)) {
//            part_cell.global_index =
//                    gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.global_index_begin + (part_cell.y - gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_begin);
//
//            return true;
//
//        } else {
//
//            if((part_cell.y > gap_iterator.data[part_cell.level][part_cell.pc_offset][0].y_max) | (part_cell.y < gap_iterator.data[part_cell.level][part_cell.pc_offset][0].y_min)){
//                return false;
//            }
//            const uint16_t y_prev =  gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_end;
//
//            if(part_cell.y > y_prev) {
//                //first try next iteration (iterate forward)
//                if (gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index <
//                    (gap_iterator.data[part_cell.level][part_cell.pc_offset][0].gap_num - 1)) {
//
//                    //need to reset if below.
//                    gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index++;
//                    gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap = ygaps.data[part_cell.level][part_cell.pc_offset][gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index];
//
//                    if ((part_cell.y >=
//                         gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_begin) &
//                        (part_cell.y <= gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_end)) {
//                        //is it in the next gap
//
//                        part_cell.global_index =
//                                gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.global_index_begin +
//                                (part_cell.y -
//                                 gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_begin);
//
//                        return true;
//                    } else {
//                        //does it actually exist
//                        if ((part_cell.y > y_prev) & (part_cell.y <
//                                                      gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap.y_begin)) {
//                            //does not exist
//                            return false;
//                        } else {
//                            //must be further along, if possible
//                            if (gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index <
//                                (gap_iterator.data[part_cell.level][part_cell.pc_offset][0].gap_num - 1)) {
//
//                                //need to reset if below.
//                                gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index++;
//                                gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap = ygaps.data[part_cell.level][part_cell.pc_offset][gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index];
//
//                                return find_particle_cell_global_index_forward_iteration(ygaps, gap_iterator,
//                                                                                         part_cell);
//                            } else {
//                                return false;
//                            }
//                        }
//                    }
//
//                } else {
//                    //shouldn't make it here as this should mean outside range
//
//                    return false;
//                }
//            } else {
//                //reset to zero and start again
//                gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index=0;
//                gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap = ygaps.data[part_cell.level][part_cell.pc_offset][gap_iterator.data[part_cell.level][part_cell.pc_offset][0].current_gap_index];
//
//                return find_particle_cell_global_index_forward_iteration(ygaps,gap_iterator,part_cell);
//            }
//
//        }
//
//    } else {
//        return false;
//    }
    return false;

}

void initialize_neigh(ExtraPartCellData<ParticleCellGapMap>& gap_map){

    for(uint64_t i = gap_map.depth_min;i <= gap_map.depth_max;i++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = gap_map.x_num[i];
        const unsigned int z_num_ = gap_map.z_num[i];

//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
        for (int z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure


            for (int x_ = 0; x_ < x_num_; x_++) {
                const size_t offset_pc_data = x_num_*z_ + x_;

                if(gap_map.data[i][offset_pc_data].size() > 0){
                    gap_map.data[i][offset_pc_data][0].current_iterator = gap_map.data[i][offset_pc_data][0].map.begin();
                }
            }
        }
    }



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

    //apr.write_apr_paraview(options.directory,name,apr.particles_int);

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

    ExtraPartCellData<YGap> ygaps;
    ygaps.initialize_structure_parts_empty(apr.particles_int);

//    ExtraPartCellData<uint64_t> gaps_end;
//    gaps_end.initialize_structure_parts_empty(apr.particles_int);
//
//    ExtraPartCellData<uint64_t> index;
//    index.initialize_structure_parts_empty(apr.particles_int);

    ExtraPartCellData<GapIterator> iterator;
    iterator.initialize_structure_parts_empty(apr.particles_int);

    GapIterator gap_it;

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



                uint64_t prev = 0;

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    float part_int= 0;

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = apr.pc_data.data[i][offset_pc_data][j_];

                    YGap gap;

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

                        uint16_t type = (node_val_pc & TYPE_MASK) >> TYPE_SHIFT;

                        uint16_t yp_j = (node_val_pc & YP_INDEX_MASK) >> YP_INDEX_SHIFT;
                        uint16_t yp_dep = (node_val_pc & YP_DEPTH_MASK) >> YP_DEPTH_SHIFT;

                        uint16_t ym_j = (node_val_pc & YM_INDEX_MASK) >> YM_INDEX_SHIFT;
                        uint16_t ym_dep = (node_val_pc & YM_DEPTH_MASK) >> YM_DEPTH_SHIFT;

                        uint16_t next_y = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;

                        uint16_t prev_y = (node_val_pc & PREV_COORD_MASK) >> PREV_COORD_SHIFT;

                        //yp_dep = yp_dep + 2;


                        if((j_ == 0) & (j_num > 1)){
                            //first node (do forward) (YM)
                            neighbours[count_parts] |= (ym_dep << YM_LEVEL_SHIFT);

                        } else if ((j_ == (j_num-1)) & (j_num > 1)){
                            //last node (do behind) (YP)
                            neighbours[count_parts-1] |= (yp_dep << YP_LEVEL_SHIFT);



                        } else if (j_num > 1){
                            // front (YM) and behind (YP)



                            neighbours[count_parts] |= (ym_dep << YM_LEVEL_SHIFT);
                            neighbours[count_parts-1] |= (yp_dep << YP_LEVEL_SHIFT);

                        }


                        if(j_>0){
                            //gaps_end.data[i][offset_pc_data].push_back(y_coord);
                            gap.y_end = y_coord;
                            ygaps.data[i][offset_pc_data].push_back(gap);

                        }

                        y_coord = (node_val_pc & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                        if(j_num > 1) {
                            if(j_ < (j_num - 1)) {
                                count_gaps++;

                                gap.y_begin = y_coord + 1;
                                gap.global_index_begin = count_parts;
                                //gaps.data[i][offset_pc_data].push_back(y_coord+1);
                                //index.data[i][offset_pc_data].push_back(count_parts);
                            }

                        }


                    }


                }

                if(j_num > 1){

                    gap_it.gap_num = ygaps.data[i][offset_pc_data].size();
                    gap_it.y_min = ygaps.data[i][offset_pc_data][0].y_begin;
                    gap_it.current_gap_index = 0;
                    gap_it.current_gap = ygaps.data[i][offset_pc_data][0];
                    gap_it.y_max = ygaps.data[i][offset_pc_data].back().y_end;

                    iterator.data[i][offset_pc_data].push_back(gap_it);
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





    //now how do I check they are right?, compare with the old structure and request the information out.

    std::vector<uint16_t> shift = {YP_LEVEL_SHIFT,YM_LEVEL_SHIFT,XP_LEVEL_SHIFT,XM_LEVEL_SHIFT,ZP_LEVEL_SHIFT,ZM_LEVEL_SHIFT};
    std::vector<uint16_t> mask = {YP_LEVEL_MASK,YM_LEVEL_MASK,XP_LEVEL_MASK,XM_LEVEL_MASK,ZP_LEVEL_MASK,ZM_LEVEL_MASK};

    apr.set_part_numbers_xz();

    APR_iterator<uint16_t> neighbour_iterator(apr);

    uint64_t c = 0;

    std::vector<unsigned int > dir_vec = {0,1,2,3,4,5};

    for (apr.begin();apr.end()!=0 ;apr.it_forward()) {

        uint16_t node = neighbours[c];

        //now we only update the neighbours, and directly access them through a neighbour iterator
        apr.update_all_neighbours();

        uint16_t type = (node & PC_TYPE_MASK) >> PC_TYPE_SHIFT;

        if(type!= apr.type()){
            std::cout << "broke" << std::endl;
        }

        //loop over all the neighbours and set the neighbour iterator to it
        for (int f = 0; f < dir_vec.size(); ++f) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

            unsigned int dir = dir_vec[f];

            int node_depth_dif = (node & mask[dir]) >> shift[dir];
            int depth_dif = _NO_NEIGHBOUR;

            for (int index = 0; index < apr.number_neighbours_in_direction(dir); ++index) {
                // on each face, there can be 0-4 neighbours accessed by index
                if(neighbour_iterator.set_neighbour_iterator(apr, dir, 0)){
                    //will return true if there is a neighbour defined

                    depth_dif =  neighbour_iterator.level() - apr.level() + 1;

                }
            }

            if(node_depth_dif!=depth_dif){
                std::cout << depth_dif << " " << node_depth_dif << std::endl;
            }


            //compare with new neighbour structure;

        }

        c++;
    }


    ExtraPartCellData<ParticleCellGapMap> gap_map;
    gap_map.initialize_structure_parts_empty(apr.particles_int);

    ExtraPartCellData<std::map<uint16_t,YGap_map>::iterator> gap_map_it;
    gap_map_it.initialize_structure_parts_empty(apr.particles_int);

    for(uint64_t i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max;i++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = apr.pc_data.x_num[i];
        const unsigned int z_num_ = apr.pc_data.z_num[i];

//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            for (x_ = 0; x_ < x_num_; x_++) {


                const size_t offset_pc_data = x_num_ * z_ + x_;

                const size_t gap_num = ygaps.data[i][offset_pc_data].size();

                YGap_map ygap;
                YGap old_gap;

                if(gap_num > 0){

                    gap_map.data[i][offset_pc_data].resize(1);
                    gap_map_it.data[i][offset_pc_data].resize(1);

                    for (int j = 0; j < gap_num; ++j) {
                        old_gap = ygaps.data[i][offset_pc_data][j];

                        ygap.global_index_begin = old_gap.global_index_begin;
                        ygap.y_end = old_gap.y_end;

                        gap_map.data[i][offset_pc_data][0].map[old_gap.y_begin] = ygap;
                    }
                    //initialize the iterator
                    gap_map_it.data[i][offset_pc_data][0] = gap_map.data[i][offset_pc_data][0].map.begin();

                }

            }
        }
    }

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

                if(gap_map.data[i][offset_pc_data].size() > 0){

                    for ( const auto &p : gap_map.data[i][offset_pc_data][0].map ) {

                            YGap_map gap = p.second;
                            uint16_t y_begin = p.first;

                            uint64_t curr_index = gap.global_index_begin;

                            curr_index--;

                            for (int y = y_begin;
                                 y <= gap.y_end; y++) {

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


    float num_rep = 1;

    timer.start_timer("APR serial iterator neighbours loop");

    for (int l = 0; l < num_rep; ++l) {

        //Basic serial iteration over all particles
        for (apr.begin(); apr.end() != 0; apr.it_forward()) {

            //now we only update the neighbours, and directly access them through a neighbour iterator
            apr.update_all_neighbours();

            float counter = 0;
            float temp = 0;

            //loop over all the neighbours and set the neighbour iterator to it
            for (int dir = 0; dir < 6; ++dir) {
                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                for (int index = 0; index < apr.number_neighbours_in_direction(dir); ++index) {
                    // on each face, there can be 0-4 neighbours accessed by index
                    if (neighbour_iterator.set_neighbour_iterator(apr, dir, index)) {
                        //will return true if there is a neighbour defined

                        neighbour_iterator(apr.particles_int) = neighbour_iterator.x();
                        //counter++;

                    }
                }
            }

        }

    }

    timer.stop_timer();



    ////////////////////////////
    ///
    /// Prototype neighbour access
    ///
    //////////////////////////

    PartCell input;
    PartCell neigh;

    initialize_neigh(gap_map);

    timer.start_timer("new neighbour loop");

    for (int l = 0; l < num_rep; ++l) {



        for (uint64_t i = apr.pc_data.depth_min; i <= apr.pc_data.depth_max; i++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = apr.pc_data.x_num[i];
            const unsigned int z_num_ = apr.pc_data.z_num[i];

            input.level = i;

//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
            for (z_ = 0; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                input.z = z_;

                for (x_ = 0; x_ < x_num_; x_++) {

                    const size_t offset_pc_data = x_num_ * z_ + x_;

                    input.x = x_;

                    if(gap_map.data[i][offset_pc_data].size() > 0){

                        for ( const auto &p : gap_map.data[i][offset_pc_data][0].map ) {

                            YGap_map gap = p.second;
                            uint16_t y_begin = p.first;

                            uint64_t curr_index = gap.global_index_begin;

                            curr_index--;

                            for (int y = y_begin;
                                 y <= gap.y_end; y++) {

                                curr_index++;

                                input.y = y;

                                uint16_t node = neighbours[curr_index];

                                for (int f = 0; f < dir_vec.size(); ++f) {
                                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                                    unsigned int face = dir_vec[f];

                                    uint16_t level_delta = (node & mask[face]) >> shift[face];

                                    for (int n = 0; n < number_neighbours_in_direction(level_delta); ++n) {
                                        get_neighbour_coordinate(input, neigh, face, level_delta, n);

                                        if (neigh.x < apr.pc_data.x_num[neigh.level]) {
                                            if (neigh.z < apr.pc_data.z_num[neigh.level]) {

                                                neigh.pc_offset =
                                                        apr.pc_data.x_num[neigh.level] * neigh.z + neigh.x;

                                                if (find_particle_cell(gap_map, neigh)) {
                                                    // do something;
                                                    pint[neigh.global_index] = neigh.x;
                                                }
                                            }
                                        }

                                    }

                                }
                            }
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();
//
//
//
//    //////////////////////////////////////////
//    ///
//    ///
//    /// Check the loop
//    ///
//    ///
//    ///////////////////////////////////
//
//    ExtraPartCellData<uint64_t> index_vec(apr);
//
//    uint64_t cp = 0;
//
//    for (apr.begin();apr.end()!=0 ;apr.it_forward()) {
//        apr(index_vec) = cp;
//        cp++;
//    }
//
//
//    Mesh_data<uint64_t> index_image;
//
//    index_image.initialize(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2));
//
//
//    //CHECK THE CHECKING SCHEME FIRST
//
//    //Basic serial iteration over all particles
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//        //now we only update the neighbours, and directly access them through a neighbour iterator
//        apr.update_all_neighbours();
//
//        float counter = 0;
//        float temp = 0;
//
//        //loop over all the neighbours and set the neighbour iterator to it
//        for (int dir = 0; dir < 6; ++dir) {
//            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//            for (int index = 0; index < apr.number_neighbours_in_direction(dir); ++index) {
//                // on each face, there can be 0-4 neighbours accessed by index
//                if(neighbour_iterator.set_neighbour_iterator(apr, dir, index)){
//                    //will return true if there is a neighbour defined
//
//                    uint16_t x_global = neighbour_iterator.x_nearest_pixel();
//                    uint16_t y_global = neighbour_iterator.y_nearest_pixel();
//                    uint16_t z_global = neighbour_iterator.z_nearest_pixel();
//
//                    index_image(y_global,x_global,z_global) = neighbour_iterator(index_vec);
//
//                }
//            }
//        }
//
//    }
//
//    //Basic serial iteration over all particles
//    for (apr.begin(); apr.end() != 0; apr.it_forward()) {
//
//        //now we only update the neighbours, and directly access them through a neighbour iterator
//        apr.update_all_neighbours();
//
//        float counter = 0;
//        float temp = 0;
//
//        //loop over all the neighbours and set the neighbour iterator to it
//        for (int dir = 0; dir < 6; ++dir) {
//            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//            for (int index = 0; index < apr.number_neighbours_in_direction(dir); ++index) {
//                // on each face, there can be 0-4 neighbours accessed by index
//                if(neighbour_iterator.set_neighbour_iterator(apr, dir, index)){
//                    //will return true if there is a neighbour defined
//
//                    uint16_t x_global = neighbour_iterator.x_nearest_pixel();
//                    uint16_t y_global = neighbour_iterator.y_nearest_pixel();
//                    uint16_t z_global = neighbour_iterator.z_nearest_pixel();
//
//                    uint64_t neigh_index = index_image(y_global,x_global,z_global);
//                    uint64_t neigh_truth = neighbour_iterator(index_vec);
//
//                    if(neigh_index != neigh_truth){
//                        std::cout << "test still broke" << std::endl;
//                    }
//                }
//            }
//        }
//
//    }
//
//
//
//
//    for(uint64_t i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max;i++) {
//        //loop over the resolutions of the structure
//        const unsigned int x_num_ = apr.pc_data.x_num[i];
//        const unsigned int z_num_ = apr.pc_data.z_num[i];
//
//        input.level = i;
//
////#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
//        for (z_ = 0; z_ < z_num_; z_++) {
//            //both z and x are explicitly accessed in the structure
//
//            input.z = z_;
//
//            for (x_ = 0; x_ < x_num_; x_++) {
//
//                const size_t offset_pc_data = x_num_*z_ + x_;
//
//                input.x = x_;
//
//                if(iterator.data[i][offset_pc_data].size() > 0){
//
//                    for (int j = 0; j < ygaps.data[i][offset_pc_data].size(); ++j) {
//
//                        YGap gap = ygaps.data[i][offset_pc_data][j];
//
//                        uint64_t curr_index = gap.global_index_begin;
//
//                        curr_index--;
//
//                        for (int y = gap.y_begin;
//                             y <= gap.y_end; y++) {
//
//                            curr_index++;
//
//                            input.y = y;
//
//                            uint16_t node = neighbours[curr_index];
//
//                            for (int f = 0; f < dir_vec.size(); ++f) {
//                                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
//
//                                unsigned int face = dir_vec[f];
//
//                                uint16_t level_delta = (node & mask[face]) >> shift[face];
//
//                                for (int n = 0; n < number_neighbours_in_direction(level_delta); ++n) {
//                                    get_neighbour_coordinate(input,neigh,face,level_delta, n);
//
//                                    if(input.x < apr.spatial_index_x_max(neigh.level)){
//                                        if(input.z < apr.spatial_index_z_max(neigh.level)){
//
//                                            neigh.pc_offset = apr.spatial_index_x_max(neigh.level)*neigh.z + neigh.x;
//
//                                            if(find_particle_cell_global_index_forward_iteration(ygaps, iterator,neigh)){
//                                                // do something;
//
//                                                uint16_t x_global = floor((neigh.x+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                uint16_t y_global = floor((neigh.y+0.5)*pow(2, apr.level_max() - neigh.level));
//                                                uint16_t z_global = floor((neigh.z+0.5)*pow(2, apr.level_max() - neigh.level));
//
//                                                uint64_t neigh_index = index_image(y_global,x_global,z_global);
//                                                uint64_t neigh_index_comp = neigh.global_index;
//
//                                                if(neigh_index != neigh.global_index){
//                                                    std::cout << "broken" << std::endl;
//                                                    std::cout << std::to_string(number_neighbours_in_direction(level_delta)) << std::endl;
//                                                }
//
//                                            }
//
//                                        }
//                                    }
//
//                                }
//
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//
//


//    apr.write_particles_only(options.directory,name+"gaps",gaps);
//    apr.write_particles_only(options.directory,name+"gaps_end",gaps_end);
//    apr.write_particles_only(options.directory,name+"index",index);
//    apr.write_particles_only(options.directory,name+"iterator",iterator);


}

