#ifndef _graph_cut_h
#define _graph_cut_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"

#include "instances.inc"

typedef Graph<float,float,float> GraphType;




void construct_max_flow_graph(PartCellStructure<float,uint64_t>& pc_struct,GraphType& g){
    //
    //  Constructs naiive max flow model for APR
    //
    //
    
    Part_timer timer;
    
    
    int num_parts = pc_struct.get_number_parts();
    int num_cells = pc_struct.get_number_cells();
    
    std::cout << "Got part neighbours" << std::endl;
    
    //Get the other part rep information
    
    
    float beta = 8;
    float k_max = pc_struct.depth_max;
    float k_min = pc_struct.depth_min;
    float alpha = 100;
    
    for(int i = 0; i < num_parts; i++){
        //adds the node
        g.add_node();
        
    }
    
    
    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;
    
    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets = {2,2,2};
    
    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets);
    
    //initialize variables required
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    // Extra variables required
    //
    
    uint64_t status=0;
    uint64_t part_offset=0;
    uint64_t p;
    
    //////////////////////////////////
    //
    // Set node values
    //
    //////////////////////////////////
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
    
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //Indicates this is a particle cell node
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        float Ip;
                        
                        float loc_min = adaptive_min.get_val(curr_key);
                        float loc_max = adaptive_max.get_val(curr_key);
                        
                        float cap_s;
                        float cap_t;
                        
                        uint64_t global_part_index;
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            
                            Ip = pc_struct.part_data.get_part(curr_key);
                            global_part_index = pc_struct.part_data.get_global_index(curr_key);
                            
                            cap_s =   alpha*pow(Ip - loc_min,2)/pow(loc_max - loc_min,2);
                            cap_t =   alpha*pow(Ip-loc_max,2)/pow(loc_max - loc_min,2);
                                
                            g.add_tweights((int) global_part_index,   /* capacities */ cap_s, cap_t);
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    
    ////////////////////////////////////
    //
    //  Assign Neighbour Relationships
    //
    /////////////////////////////////////
    
    PartCellNeigh<uint64_t> neigh_part_keys;
    PartCellNeigh<uint64_t> neigh_cell_keys;
    uint64_t node_val_pc;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_pc,node_val_part,curr_key,status,part_offset) firstprivate(neigh_part_keys,neigh_cell_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_struct.pc_data.pc_key_set_z(curr_key,z_);
            pc_struct.pc_data.pc_key_set_depth(curr_key,i);
            
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.pc_data.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_struct.pc_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            pc_struct.part_data.get_part_neighs_all(p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                            
                            uint64_t global_part_index = pc_struct.part_data.get_global_index(curr_key);
                            uint64_t global_part_index_neigh;
                            uint64_t status_neigh;
                            uint64_t depth_neigh;
                            
                            for(uint64_t face = 0;face<6;face++){
                                for(uint64_t n = 0; n < neigh_part_keys.neigh_face[face].size();n++){
                                    uint64_t neigh_key = neigh_part_keys.neigh_face[face][n];
                                    if (neigh_key > 0){
                                        
                                        //////////////////////
                                        //
                                        //  Neighbour Energy
                                        //
                                        ///////////////////////
                                        
                                        //get neighbour details
                                        global_part_index_neigh = pc_struct.part_data.get_global_index(neigh_key); //global index
                                        status_neigh = pc_struct.part_data.access_data.pc_key_get_status(neigh_key);
                                        depth_neigh = pc_struct.part_data.access_data.pc_key_get_depth(neigh_key);
                                        
                                        //float cap = beta*pow(status*status_neigh,2)*pow((-i+k_max + 1)*(-depth_neigh+k_max + 1),4)/pow((1.0)*(k_max+1-k_min),4.0);
                                        //g.add_edge( global_part_index, global_part_index_neigh,    /* capacities */  cap, cap );
                                        
                                        
                                        if(i >= (k_max-1)){
                                            float cap = beta*pow(status*status_neigh,2)*pow((-i+k_max + 1)*(-depth_neigh+k_max + 1),4)/pow((1.0)*(k_max+1-k_min),4.0);
                                            g.add_edge( (int) global_part_index, global_part_index_neigh,    /* capacities */  cap, cap );

                                            
                                        }
                                        else {
                                            float cap = beta*81.0;
                                            g.add_edge( (int) global_part_index, global_part_index_neigh,    /* capacities */  cap, cap );
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

//void get_seg_gc(std::string name,int dim){
//    //
//    //  Bevan Cheeseman 2016
//    //
//    //  Calculate
//    //
//    
//    std::string image_path;
//    
//    image_path = get_path("IMAGE_GEN_PATH");
//    
//    
//    Part_rep p_rep(dim);
//    
//    p_rep.timer.verbose_flag = true;
//    
//    
//    
//    
//    //p_rep.pars.name = "Nat_images/" + name;
//    p_rep.pars.name =  name;
//    
//    read_parts_from_full_hdf5(p_rep,image_path + p_rep.pars.name + "_full.h5");
//    
//    
//    std::vector<std::vector<unsigned int>> neigh_list;
//    p_rep.timer.start_timer("cell list");
//    get_cell_neigh_full(p_rep,neigh_list ,0);
//    p_rep.timer.stop_timer();
//    
//    
//    GraphType *g = new GraphType(p_rep.Ip.data.size() ,p_rep.Ip.data.size()*4 );
//    
//    p_rep.timer.start_timer("construct graph");
//    
//    construct_max_flow_graph(p_rep,*g);
//    
//    p_rep.timer.stop_timer();
//    
//    
//    p_rep.timer.start_timer("max_flow");
//    
//    int flow = g -> maxflow();
//    
//    p_rep.timer.stop_timer();
//    
//    printf("Flow = %d\n", flow);
//    printf("Minimum cut:\n");
//    
//    
//    ///////////////////////////
//    //
//    //	Output Particle Cell Structures
//    //
//    //////////////////////////////
//    
//    p_rep.create_uint16_dataset("Label", p_rep.num_parts);
//    p_rep.part_data_list["Label"].print_flag = 1;
//    
//    
//    Part_data<uint16_t>* Label = p_rep.get_data_ref<uint16_t>("Label");
//    
//    for(int i = 0;i < p_rep.Ip.data.size();i++){
//        
//        if (g->what_segment(i) == GraphType::SOURCE) {
//            Label->data[i] = 255;
//        }
//        else {
//            Label->data[i] = 0;
//        }
//    }
//    
//    delete g;
//    
//    std::string save_loc = image_path;
//    std::string output_file_name = name + "_gc";
//    
//    write_apr_to_hdf5_inc_extra_fields(p_rep,save_loc,output_file_name);
//    
//    Mesh_data<uint16_t> out_image;
//    
//    //output label image
//    if (dim == 3){
//        interp_parts_to_pc(out_image,p_rep,Label->data);
//    } else if (dim == 2) {
//        interp_parts_to_pc_2D(out_image,p_rep,Label->data);
//    }
//    
//    debug_write(out_image, name + "_seg_gc");
//    
//    
//    
//}




#endif
