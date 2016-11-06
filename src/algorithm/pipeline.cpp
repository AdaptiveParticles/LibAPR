
#include <algorithm>
#include <iostream>

#include "pipeline.h"
#include "../data_structures/meshclass.h"
#include "../io/readimage.h"

#include "gradient.hpp"
#include "../data_structures/particle_map.hpp"
#include "../data_structures/Tree/Content.hpp"
#include "../data_structures/Tree/LevelIterator.hpp"
#include "../data_structures/Tree/Tree.hpp"
#include "../data_structures/Tree/PartCellBase.hpp"
#include "../data_structures/Tree/PartCellStructure.hpp"
#include "level.hpp"
#include "../io/writeimage.h"
#include "../io/write_parts.h"



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


void create_sparse_graph_format(Particle_map<float>& part_map);


bool compare_y_coords(PartCellStructure<float,uint64_t>& pc_struct){
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t y_;
    uint64_t j_;
    uint64_t status;
    uint64_t status_org;
    
    
    bool pass_test = true;
    
    //Neighbour Routine Checking
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num = pc_struct.pc_data.x_num[i];
        const unsigned int z_num = pc_struct.pc_data.z_num[i];
        
        
        for(z_ = 0;z_ < z_num;z_++){
            
            for(x_ = 0;x_ < x_num;x_++){
                
                const size_t offset_pc_data = x_num*z_ + x_;
                y_coord = 0;
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    uint64_t node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (node_val&1){
                        
                        uint64_t next_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        uint64_t prev_coord = (node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT;
                        
                        uint64_t y_coord_prev = y_coord;
                        uint64_t y_coord_diff = y_coord +  ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        
                        //get the index gap node
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        
                        if((y_coord_diff != (y_coord)) & (y_coord > 0)){
                            std::cout << " FAILED COORDINATE TEST" << std::endl;
                            pass_test = false;
                        }
                        
                        
                        y_coord--;
                        
                        
                        
                    } else {
                        //normal node
                        y_coord++;
                        
                        
                        
                    }
                }
                
            }
        }
        
    }
    
    std::cout << "Y_coordinate test" << std::endl;
    
    return pass_test;
}

bool compare_sparse_rep_neighpart_with_part_map(const Particle_map<float>& part_map,PartCellStructure<float,uint64_t>& pc_struct){
    //
    //
    //  Tests the particle sampling and particle neighbours;
    //
    //
    //
    //
    
    
    
    
    
    //initialize
    uint64_t node_val;
    uint64_t y_coord;
    int x_;
    int z_;
    uint64_t y_;
    uint64_t j_;
    uint64_t status;
    uint64_t status_org;
    
    
    bool pass_test = true;
    
    //Neighbour Routine Checking
    
    for(int i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num = pc_struct.pc_data.x_num[i];
        const unsigned int z_num = pc_struct.pc_data.z_num[i];
        
        
        for(z_ = 0;z_ < z_num;z_++){
            
            for(x_ = 0;x_ < x_num;x_++){
                
                const size_t offset_pc_data = x_num*z_ + x_;
                y_coord = 0;
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                const size_t offset_part_map_data_0 = part_map.downsampled[i].y_num*part_map.downsampled[i].x_num*z_ + part_map.downsampled[i].y_num*x_;
                
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    uint64_t node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (node_val&1){
                        
                        uint64_t next_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        uint64_t prev_coord = (node_val & PREV_COORD_MASK) >> PREV_COORD_SHIFT;
                        
                        uint64_t y_coord_prev = y_coord;
                        uint64_t y_coord_diff = y_coord +  ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        
                        //get the index gap node
                        y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                        
                        if((y_coord_diff != y_coord) & (y_coord > 0)){
                            int stop = 1;
                        }
                        
                        
                        y_coord--;
                        
                        
                        
                        
                    } else {
                        //normal node
                        y_coord++;
                        
                        
                        //get and check status
                        status = (node_val & STATUS_MASK) >> STATUS_SHIFT;
                        status_org = part_map.layers[i].mesh[offset_part_map_data_0 + y_coord];
                        
                        
                        //get the index gap node
                        
                        uint64_t curr_key = 0;
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        pc_struct.part_data.access_data.pc_key_set_z(curr_key,z_);
                        pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                        pc_struct.part_data.access_data.pc_key_set_depth(curr_key,i);
                        
                        PartCellNeigh<uint64_t> neigh_keys;
                        PartCellNeigh<uint64_t> neigh_cell_keys;
                        
                        
                        //neigh_keys.resize(0);
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        uint64_t part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        if((x_==3) & (z_ ==3) & (y_coord == 3) & (i == 2)){
                            int stop = 1;
                        }
                        
                        
                        //loop over the particles
                        for(int p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            pc_struct.part_data.access_data.pc_key_set_partnum(curr_key,p);
                            pc_struct.part_data.get_part_neighs_all(p,node_val,curr_key,status,part_offset,neigh_cell_keys,neigh_keys,pc_struct.pc_data);
                            
                            //First check your own intensity;
                            float own_int = pc_struct.part_data.get_part(curr_key);
                            
                            if(status == SEED){
                                
                                uint64_t curr_depth = pc_struct.pc_data.pc_key_get_depth(curr_key) + 1;
                                
                                uint64_t part_num = pc_struct.pc_data.pc_key_get_partnum(curr_key);
                                
                                uint64_t curr_x = pc_struct.pc_data.pc_key_get_x(curr_key)*2 + pc_struct.pc_data.seed_part_x[part_num];
                                uint64_t curr_z = pc_struct.pc_data.pc_key_get_z(curr_key)*2 + pc_struct.pc_data.seed_part_z[part_num];
                                uint64_t curr_y = y_coord*2 + pc_struct.pc_data.seed_part_y[part_num];
                                
                                curr_x = std::min(curr_x,(uint64_t)(part_map.downsampled[curr_depth].x_num-1));
                                curr_z = std::min(curr_z,(uint64_t)(part_map.downsampled[curr_depth].z_num-1));
                                curr_y = std::min(curr_y,(uint64_t)(part_map.downsampled[curr_depth].y_num-1));
                                
                                const size_t offset_part_map = part_map.downsampled[curr_depth].y_num*part_map.downsampled[curr_depth].x_num*curr_z + part_map.downsampled[curr_depth].y_num*curr_x;
                                
                                if(own_int == (offset_part_map + curr_y)){
                                    //correct value
                                } else {
                                    
                                   
                                    
                                    std::cout << "Particle Intensity Error" << std::endl;
                                    pass_test = false;
                                }
                                
                                
                            } else {
                                
                                uint64_t curr_depth = pc_struct.pc_data.pc_key_get_depth(curr_key);
                                
            
                                
                                uint64_t curr_x = pc_struct.pc_data.pc_key_get_x(curr_key);
                                uint64_t curr_z = pc_struct.pc_data.pc_key_get_z(curr_key);
                                uint64_t curr_y = y_coord;

                                const size_t offset_part_map = part_map.downsampled[curr_depth].y_num*part_map.downsampled[curr_depth].x_num*curr_z + part_map.downsampled[curr_depth].y_num*curr_x;
                                
                                if(own_int == (offset_part_map + curr_y)){
                                    //correct value
                                } else {
                                    std::cout << "Particle Intensity Error" << std::endl;
                                    pass_test = false;
                                }

                            }
                            
                            
                            //Check the x and z nieghbours, do they exist?
                            for(int face = 0;face < 6;face++){
                                
                                uint64_t x_n = 0;
                                uint64_t z_n = 0;
                                uint64_t y_n = 0;
                                uint64_t depth = 0;
                                uint64_t j_n = 0;
                                uint64_t status_n = 0;
                                uint64_t intensity = 1;
                                uint64_t node_n = 0;
                                
                                
                                for(int n = 0;n < neigh_keys.neigh_face[face].size();n++){
                                    
                                    
                                    pc_struct.pc_data.get_neigh_coordinates_part(neigh_keys,face,n,y_coord,y_n,x_n,z_n,depth);
                                    j_n = (neigh_keys.neigh_face[face][n] & PC_KEY_J_MASK) >> PC_KEY_J_SHIFT;
                                    status_n = pc_struct.pc_data.pc_key_get_status(neigh_keys.neigh_face[face][n]);
                                    
                                    if (neigh_keys.neigh_face[face][n] > 0){
                                        
                                        float own_int = pc_struct.part_data.get_part(neigh_keys.neigh_face[face][n]);
                                        
                                        //calculate y so you can check back in the original structure
                                        
                                        uint64_t depth_ind = pc_struct.pc_data.pc_key_get_depth(neigh_keys.neigh_face[face][n]);
                                        if(status_n == SEED){
                                            depth_ind = depth_ind + 1;
                                        }
                                        
                        
                                        x_n = std::min(x_n,(uint64_t)(part_map.downsampled[depth_ind].x_num-1));
                                        z_n = std::min(z_n,(uint64_t)(part_map.downsampled[depth_ind].z_num-1));
                                        y_n = std::min(y_n,(uint64_t)(part_map.downsampled[depth_ind].y_num-1));
                                        
                                        
                                        const size_t offset_part_map = part_map.downsampled[depth].y_num*part_map.downsampled[depth].x_num*z_n + part_map.downsampled[depth].y_num*x_n;
                                        //status_n = part_map.layers[depth].mesh[offset_part_map + y_n];
                                        
                                        float corr_val = (offset_part_map + y_n);
                                        
                                        if(own_int == (offset_part_map + y_n)){
                                            //correct value
                                        } else {
                                            
                                            
                                            float val = part_map.downsampled[depth_ind].mesh[offset_part_map+y_n];
                                            
                                            uint64_t z_t = floor(own_int/( part_map.downsampled[depth_ind].y_num*part_map.downsampled[depth_ind].x_num));
                                            uint64_t x_t = floor((own_int - z_t*( part_map.downsampled[depth_ind].y_num*part_map.downsampled[depth_ind].x_num))/part_map.downsampled[depth_ind].y_num);
                                            uint64_t y_t = own_int - z_t*( part_map.downsampled[depth_ind].y_num*part_map.downsampled[depth_ind].x_num) - x_t*part_map.downsampled[depth_ind].y_num;
                                            
                                            uint64_t z_c = pc_struct.pc_data.pc_key_get_z(neigh_keys.neigh_face[face][n]);
                                            uint64_t x_c = pc_struct.pc_data.pc_key_get_x(neigh_keys.neigh_face[face][n]);
                                            uint64_t j_c = pc_struct.pc_data.pc_key_get_j(neigh_keys.neigh_face[face][n]);
                                            uint64_t d_c = pc_struct.pc_data.pc_key_get_depth(neigh_keys.neigh_face[face][n]);
                                            
                                            std::cout << "Neighbour Particle Intensity Error" << std::endl;
                                            own_int = pc_struct.part_data.get_part(neigh_keys.neigh_face[face][n]);
                                            pc_struct.pc_data.get_neigh_coordinates_part(neigh_keys,face,n,y_coord,y_n,x_n,z_n,depth);

                                            pass_test = false;
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
    
    std::cout << "Finished Neigh Part test" << std::endl;
    
    
    
    return pass_test;
    
    
    
}





int main(int argc, char **argv) {

    Part_rep part_rep;

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);

    // COMPUTATIONS

    Mesh_data<float> input_image_float;
    Mesh_data<float> gradient, variance;
    {
        Mesh_data<uint16_t> input_image;

        load_image_tiff(input_image, options.input);

        gradient.initialize(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        part_rep.initialize(input_image.y_num, input_image.x_num, input_image.z_num);

        input_image_float = input_image.to_type<float>();

        // After this block, input_image will be freed.
    }

    if(!options.stats_file) {
        // defaults

        part_rep.pars.dy = part_rep.pars.dx = part_rep.pars.dz = 1;
        part_rep.pars.psfx = part_rep.pars.psfy = part_rep.pars.psfz = 1;
        part_rep.pars.rel_error = 0.1;
        part_rep.pars.var_th = 0;
        part_rep.pars.var_th_max = 0;
    }

    Part_timer t;
    t.verbose_flag = true;


    // preallocate_memory
    Particle_map<float> part_map(part_rep);
    preallocate(part_map.layers, gradient.y_num, gradient.x_num, gradient.z_num, part_rep);
    variance.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);
    std::vector<Mesh_data<float>> down_sampled_images;

    // variables for tree
    std::vector<uint64_t> tree_mem(gradient.y_num * gradient.x_num * gradient.z_num * 1.25, 0);
    std::vector<Content> contents(gradient.y_num * gradient.x_num * gradient.z_num, {0});

    Mesh_data<float> temp;
    temp.preallocate(gradient.y_num, gradient.x_num, gradient.z_num, 0);

    t.start_timer("whole");
    
    part_map.downsample(input_image_float);
    
    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);
    
    part_rep.timer.start_timer("get_gradient_3D");
    get_gradient_3D(part_rep, input_image_float, gradient);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("get_variance_3D");
    get_variance_3D(part_rep, input_image_float, variance);
    part_rep.timer.stop_timer();


    part_rep.timer.start_timer("get_level_3D");
    get_level_3D(variance, gradient, part_rep, part_map, temp);
    part_rep.timer.stop_timer();
    
    
    // free memory (not used anymore)
    std::vector<float>().swap( gradient.mesh );
    std::vector<float>().swap( variance.mesh );
    

    part_rep.timer.start_timer("pushing_scheme");
    part_map.pushing_scheme(part_rep);
    part_rep.timer.stop_timer();
    
 
    part_rep.timer.start_timer("estimate_part_intensity");
    
    std::swap(part_map.downsampled[part_map.k_max+1],input_image_float);
    
    Tree<float> tree(part_map, tree_mem, contents);
    part_rep.timer.stop_timer();

    t.stop_timer();


    

    size_t main_elems = 0;
    std::vector<size_t> elems(25, 0);
    std::vector<uint64_t> neighbours(20);
    std::vector<coords3d> part_coords;
    
    part_rep.timer.start_timer("iterating of tree");
    
    uint64_t curr;
    size_t curr_status;
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();
    float intensity = 0;
    
    for(int l = part_rep.pl_map.k_min;l <= part_rep.pl_map.k_max + 1;l++){
        for(LevelIterator<float> it(tree, l); it != it.end(); it++)
        {
            //curr = *it;
            
            //curr_status = tree.get_status(*it);
            
            //it.get_current_particle_coords(part_coords);
            
            neighbours.resize(24);
            tree.get_neighbours(*it, it.get_current_coords(), it.level_multiplier,
                                it.child_index, neighbours);
            //main_elems++;
            
            //elems[neighbours.size()]++;
            
            //raw_content[raw_tree[*it + 2]].intensity += 5;
            
        }
    }

    part_rep.timer.stop_timer();
    
    std::cout << "Size of data structure: " << tree.get_tree_size() << std::endl;
    std::cout << "Number of parts : " << tree.get_content_size() << std::endl;
    std::cout << "Ratio : " << ((float)tree.get_tree_size())/((float)tree.get_content_size()) << std::endl;
    
    std::cout << "Size of data structure (MB): " << tree.get_tree_size()*8.0/1000000.0 << std::endl;
    std::cout << "Size of parts (MB): " << tree.get_content_size()*4.0/1000000.0 << std::endl;
    
    std::cout << "Size of image (MB): " << part_rep.org_dims[0]*part_rep.org_dims[1]*part_rep.org_dims[2]*4.0/1000000.0 << std::endl;
    
    
    //output
    std::string save_loc = options.output;
    std::string file_name = options.stats;
    
//    part_rep.timer.start_timer("write full");
//    write_apr_full_format(part_rep,tree,save_loc,file_name);
//    part_rep.timer.stop_timer();
//    
//    part_rep.timer.start_timer("write tree");
//    write_apr_tree_format(part_rep,tree,save_loc,file_name);
//    part_rep.timer.stop_timer();
//    
//    part_rep.timer.start_timer("write partmap");
//    write_apr_partmap_format(part_rep,part_map,tree,save_loc,file_name);
//    part_rep.timer.stop_timer();
    
    //testing sparse format
    
    part_rep.timer.start_timer("compute new structure");
    PartCellStructure<float,uint64_t> pcell_test(part_map);
    part_rep.timer.stop_timer();
//    
//    pcell_test.pc_data.test_get_neigh_dir();
//    pcell_test.part_data.test_get_part_neigh_dir(pcell_test.pc_data);
//    pcell_test.part_data.test_get_part_neigh_all(pcell_test.pc_data);
//    pcell_test.part_data.test_get_part_neigh_all_memory(pcell_test.pc_data);
//    
//    
    
    compare_y_coords(pcell_test);
    
    Particle_map<float> particle_map;
    
    
    auto &layers = particle_map.layers;
    auto &downsampled = particle_map.downsampled;
    
    layers.resize(4);
    layers[3].x_num = layers[3].y_num = layers[3].z_num = 8;
    layers[2].x_num = layers[2].y_num = layers[2].z_num = 4;
    layers[1].x_num = layers[1].y_num = layers[1].z_num = 2;
    
    layers[1].mesh = {SLOPESTATUS, PARENTSTATUS, SLOPESTATUS, SLOPESTATUS, SLOPESTATUS,
        SLOPESTATUS, PARENTSTATUS, PARENTSTATUS};
    
    layers[2].mesh.resize(64, SLOPESTATUS);
    layers[2].mesh[3] = PARENTSTATUS;
    layers[2].mesh[61] = PARENTSTATUS;
    layers[2].mesh[62] = PARENTSTATUS;
    
    layers[3].mesh.resize(512, NEIGHBOURSTATUS);
    layers[3].mesh[7] = TAKENSTATUS;
    layers[3].mesh[507] = TAKENSTATUS;
    
    downsampled.resize(5);
    
    downsampled[4].x_num = downsampled[4].y_num = downsampled[4].z_num = 16;
    downsampled[3].x_num = downsampled[3].y_num = downsampled[3].z_num = 8;
    downsampled[2].x_num = downsampled[2].y_num = downsampled[2].z_num = 4;
    downsampled[1].x_num = downsampled[1].y_num = downsampled[1].z_num = 2;
    downsampled[1].mesh = {1,0,3,4,5,6,0,8};
    downsampled[2].mesh.resize(64, 1);
    downsampled[3].mesh.resize(512, 2);
    downsampled[4].mesh.resize(4096, 3);
    
    particle_map.k_min = 1;
    particle_map.k_max = 3;
    
    
    // Set the intensities
    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
        
        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
            particle_map.downsampled[depth].mesh[i] = i;
        }
        
    }
    
    
    PartCellStructure<float,uint64_t> pc_small(particle_map);
    
    
    compare_y_coords(pc_small);
    
    compare_sparse_rep_neighpart_with_part_map(particle_map,pc_small);
    
}


