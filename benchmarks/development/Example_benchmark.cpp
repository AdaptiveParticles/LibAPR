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

#include "benchmarks/development/Example_benchmark.h"

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

template<typename T>
void particle_linear_neigh_access(APR<T>& apr,float num_repeats,AnalysisData& analysis_data);

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

    APRIterator<uint16_t> apr_iterator(apr);

    //initialization of the iteration structures
    APRIterator<uint16_t> apr_parallel_iterator(apr); //this is required for parallel access
    uint64_t part; //declare parallel iteration variable

    int num_repeats = 5;


    PartCellStructure<float,uint64_t > pc_struct;

    AnalysisData analysisData;

    pixels_linear_neigh_access(pc_struct,(uint64_t)apr.orginal_dimensions(0),(uint64_t)apr.orginal_dimensions(1),(uint64_t)apr.orginal_dimensions(2),num_repeats,analysisData);

    particle_linear_neigh_access(apr,num_repeats,analysisData);

    APRIterator<uint16_t> neighbour_iterator(apr);

    ExtraPartCellData<uint16_t> neigh_xm(apr);

    timer.start_timer("APR parallel iterator neighbour loop");

    for (int i = 0; i < num_repeats; ++i) {

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_parallel_iterator,neighbour_iterator)
        for (part = 0; part < apr.num_parts_total; ++part) {
            //needed step for any parallel loop (update to the next part)
            apr_parallel_iterator.set_iterator_to_particle_by_number(part);

            //compute neighbours as previously, now using the apr_parallel_iterator (APRIterator), instead of the apr class for access.
            apr_parallel_iterator.update_all_neighbours();

            //loop over all the neighbours and set the neighbour iterator to it
            for (int dir = 0; dir < 6; ++dir) {
                for (int index = 0; index < apr_parallel_iterator.number_neighbours_in_direction(dir); ++index) {

                    if(neighbour_iterator.set_neighbour_iterator(apr_parallel_iterator, dir, index)){
                        //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
                        apr_parallel_iterator(neigh_xm) += neighbour_iterator(apr.particles_int);
                    }
                }
            }

        }

    }

    timer.stop_timer();

    std::cout << "New parallel iteration took: " << (timer.t2 - timer.t1)/(num_repeats*1.0) << std::endl;



}

template<typename T>
void particle_linear_neigh_access(APR<T>& apr,float num_repeats,AnalysisData& analysis_data){   //  Calculate connected component from a binary mask
    //
    //  Should be written with the neighbour iterators instead.
    //


//    ExtraPartCellData<float> filter_output(apr);
//
//
//    ExtraPartCellData<float> particle_data(apr);
//
//
//    Part_timer timer;
//
//    //initialize variables required
//    uint64_t node_val_pc; // node variable encoding neighbour and cell information
//
//    int x_; // iteration variables
//    int z_; // iteration variables
//    uint64_t j_; // index variable
//    uint64_t curr_key = 0; // key used for accessing and particles and cells
//    PartCellNeigh<uint64_t> neigh_cell_keys;
//    //
//    // Extra variables required
//    //
//
//    timer.verbose_flag = false;
//
//    const int num_dir = 6;
//
//    timer.start_timer("neigh_cell_comp");
//
//    for(int r = 0;r < num_repeats;r++){
//
//        for(uint64_t i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max;i++){
//            //loop over the resolutions of the structure
//            const unsigned int x_num_ = apr.pc_data.x_num[i];
//            const unsigned int z_num_ = apr.pc_data.z_num[i];
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
//            for(z_ = 0;z_ < z_num_;z_++){
//                //both z and x are explicitly accessed in the structure
//                curr_key = 0;
//
//                apr.pc_data.pc_key_set_z(curr_key,z_);
//                apr.pc_data.pc_key_set_depth(curr_key,i);
//
//
//                for(x_ = 0;x_ < x_num_;x_++){
//
//                    apr.pc_data.pc_key_set_x(curr_key,x_);
//
//                    const size_t offset_pc_data = x_num_*z_ + x_;
//
//                    const size_t j_num = apr.pc_data.data[i][offset_pc_data].size();
//
//                    //the y direction loop however is sparse, and must be accessed accordinagly
//                    for(j_ = 0;j_ < j_num;j_++){
//
//
//                        float part_int= 0;
//
//                        //particle cell node value, used here as it is requried for getting the particle neighbours
//                        node_val_pc = apr.pc_data.data[i][offset_pc_data][j_];
//
//                        if (!(node_val_pc&1)){
//                            //Indicates this is a particle cell node
//                            //y_coord++;
//
//                            apr.pc_data.pc_key_set_j(curr_key,j_);
//
//                            //for(int dir = 0;dir < num_dir;dir++){
//                            // pc_data.get_neighs_face(curr_key,node_val_pc,dir,neigh_cell_keys);
//                            // }
//
//                            apr.pc_data.get_neighs_all(curr_key,node_val_pc,neigh_cell_keys);
//
//
//                            for(int dir = 0;dir < num_dir;dir++){
//                                //loop over the nieghbours
//                                for(int n = 0; n < neigh_cell_keys.neigh_face[dir].size();n++){
//                                    // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
//                                    uint64_t neigh_key = neigh_cell_keys.neigh_face[dir][n];
//
//                                    if(neigh_key > 0){
//                                        //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)
//                                        part_int+= particle_data.get_val(neigh_key);
//                                    }
//
//                                }
//                            }
//
//                            filter_output.data[i][offset_pc_data][j_] = part_int;
//
//                        } else {
//                            // Inidicates this is not a particle cell node, and is a gap node
//
//                        }
//
//                    }
//
//                }
//
//            }
//        }
//
//    }
//
//    timer.stop_timer();
//
//    float time = (timer.t2 - timer.t1)/num_repeats;
//
//    analysis_data.add_float_data("neigh_part_linear_total",time);
//    analysis_data.add_float_data("neigh_part_linear_perm",time/(1.0*apr.num_parts_total/1000000.0));
//
//    std::cout << "Get neigh particle linear: " << time << std::endl;
//    std::cout << "per 1000000 particles took: " << time/(1.0*apr.num_parts_total/1000000.0) << std::endl;

}