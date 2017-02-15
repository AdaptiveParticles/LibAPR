#ifndef _graph_cut_h
#define _graph_cut_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"

#include "../../external/maxflow-v3.04.src/instances.inc"

#include "../../benchmarks/analysis/AnalysisData.hpp"
#include "parent_numerics.hpp"
#include "misc_numerics.hpp"

typedef Graph<float,float,float> GraphType;

template<typename T,typename V>
void construct_max_flow_graph_mesh(PartCellStructure<V,T>& pc_struct,GraphType& g,std::array<uint64_t,10> parameters,AnalysisData& analysis_data){
    //
    //  Constructs naiive max flow model for APR
    //
    //
    
    Part_timer timer;
    
    pc_struct.part_data.initialize_global_index();
    
    uint64_t num_parts = pc_struct.get_number_parts();
    
    //std::cout << "Got part neighbours" << std::endl;
    
    //Get the other part rep information
    
    float beta = parameters[0];
    float k_max = pc_struct.depth_max;
    float k_min = pc_struct.depth_min;
    float alpha = parameters[1];
    
    for(int i = 0; i < pc_struct.org_dims[0]*pc_struct.org_dims[1]*pc_struct.org_dims[2]; i++){
        //adds the node
        g.add_node();
        
    }

    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;
    
    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {(unsigned int)parameters[2],(unsigned int)parameters[3],(unsigned int)parameters[4]};
    std::vector<unsigned int> status_offsets_max = {(unsigned int)parameters[5],(unsigned int)parameters[6],(unsigned int)parameters[7]};
    
    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,parameters[8],parameters[9]);
    
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
    
    //adaptive mean
    ExtraPartCellData<float> eng1(pc_struct.part_data.particle_data);
    ExtraPartCellData<float> eng2(pc_struct.part_data.particle_data);

    //////////////////////////////////
    //
    // Set node values
    //
    //////////////////////////////////

    uint64_t counter = 0;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        //ss#spragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
        
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
                            
                            
                            
                            // cap_s =   alpha*pow(Ip - loc_min,2)/pow(loc_max - loc_min,2);
                            // cap_t =   alpha*pow(Ip-loc_max,2)/pow(loc_max - loc_min,2);
                            if((loc_min > 0) & (loc_max > 0)){
                                cap_s =   alpha*(Ip - loc_min);
                                cap_t =   alpha*(loc_max-Ip);
                            } else {
                                cap_s = 0;
                                cap_t = alpha*Ip;
                            }
                            
                            g.add_tweights(global_part_index,   /* capacities */ cap_s, cap_t);
                            
                            eng1.get_part(curr_key) =cap_s +5000;
                            eng2.get_part(curr_key) =cap_t + 5000;
                            
                            counter++;
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    Mesh_data<float> eng_s;
    Mesh_data<float> eng_t;
    
    //pc_struct.interp_parts_to_pc(eng_s,eng1);
    //debug_write(eng_s,"eng1");
    //pc_struct.interp_parts_to_pc(eng_t,eng2);
    //debug_write(eng_t,"eng2");
    
    // Now loop over the mesh and add it in.
    
    counter = 0;
    
    for(int i = 0;i < eng_s.z_num;i++){
        for(int j = 0;j < eng_s.x_num;j++){
            for(int k = 0;k < eng_s.y_num;k++){
                
                g.add_tweights(  counter ,   /* capacities */ eng_s.mesh[counter], eng_t.mesh[counter]);
                counter++;
            }
        }
    }
    
    Mesh_data<uint8_t> status_mesh;
    interp_status_to_mesh(status_mesh,pc_struct);
    
    
   //debug_write(status_mesh,"status_mesh");
    
    ////////////////////////////////////
    //
    //  Assign Neighbour Relationships
    //
    /////////////////////////////////////

    uint64_t neigh_counter = 0;

    const int x_num = status_mesh.x_num;
    const int y_num = status_mesh.y_num;
    const int z_num = status_mesh.z_num;
    
    for(int i = 0;i < status_mesh.z_num;i++){
        for(int j = 0;j < status_mesh.x_num;j++){
            for(int k = 0;k < status_mesh.y_num;k++){
                //neighbours
               
                int yp = std::min(k+1,y_num-1);
                int ym = std::max(k-1,0);
                
                int xp = std::min(j+1,x_num-1);
                int xm = std::max(j-1,0);
                
                int zp = std::min(i+1,z_num-1);
                int zm = std::max(i-1,0);
                
                float cap = 0;
                
                if(yp != k){
                    cap = beta*pow(status_mesh(i,j,k)*status_mesh(i,j,yp),2)/pow(9.0,2);
                    g.add_edge( i*x_num*y_num + j*y_num + k, i*x_num*y_num + j*y_num + yp,    /* capacities */  cap, cap );
                    neigh_counter++;
                }
                
                if(ym != k){
                    cap = beta*pow(status_mesh(i,j,k)*status_mesh(i,j,ym),2)/pow(9.0,2);
                    g.add_edge( i*x_num*y_num + j*y_num + k, i*x_num*y_num + j*y_num + ym,    /* capacities */  cap, cap );
                    neigh_counter++;
                }
                
                if(xp != j){
                    cap = beta*pow(status_mesh(i,j,k)*status_mesh(i,xp,k),2)/pow(9.0,2);
                    g.add_edge( i*x_num*y_num + j*y_num + k, i*x_num*y_num + xp*y_num + k,    /* capacities */  cap, cap );
                    neigh_counter++;
                }
                
                if(xm != j){
                    cap = beta*pow(status_mesh(i,j,k)*status_mesh(i,xm,k),2)/pow(9.0,2);
                    g.add_edge( i*x_num*y_num + j*y_num + k, i*x_num*y_num + xm*y_num + k,    /* capacities */  cap, cap );
                    neigh_counter++;
                }
                if(zp != i){
                    cap = beta*pow(status_mesh(i,j,k)*status_mesh(zp,j,k),2)/pow(9.0,2);
                    g.add_edge( i*x_num*y_num + j*y_num + k, zp*x_num*y_num + j*y_num + k,    /* capacities */  cap, cap );
                    neigh_counter++;
                }
                if(zm != i){
                    cap = beta*pow(status_mesh(i,j,k)*status_mesh(zm,j,k),2)/pow(9.0,2);
                    g.add_edge( i*x_num*y_num + j*y_num + k, zm*x_num*y_num + j*y_num + k,    /* capacities */  cap, cap );
                    neigh_counter++;
                }
                
            }
        }
    }
    
   // std::cout << neigh_counter << std::endl;
    
    analysis_data.add_float_data("mesh_num_neigh",(float)neigh_counter);
    
}

template<typename T,typename V>
void construct_max_flow_graph(PartCellStructure<V,T>& pc_struct,GraphType& g,std::array<uint64_t,10> parameters,AnalysisData& analysis_data){
    //
    //  Constructs naiive max flow model for APR
    //
    //
    
    Part_timer timer;
    
    pc_struct.part_data.initialize_global_index();
    
    uint64_t num_parts = pc_struct.get_number_parts();

    
    //Get the other part rep information
    
    
    float beta = parameters[0];
    float k_max = pc_struct.depth_max;
    float k_min = pc_struct.depth_min;
    float alpha = parameters[1];
    
    for(int i = 0; i < num_parts; i++){
        //adds the node
        g.add_node();
        
    }
    
    
    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;
    
    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {(unsigned int)parameters[2],(unsigned int)parameters[3],(unsigned int)parameters[4]};
    std::vector<unsigned int> status_offsets_max = {(unsigned int)parameters[5],(unsigned int)parameters[6],(unsigned int)parameters[7]};
    
    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,parameters[8],parameters[9]);
    
    Mesh_data<float> output_img;
//    interp_extrapc_to_mesh(output_img,pc_struct,adaptive_min);
//    debug_write(output_img,"adapt_min");
//    
//    interp_extrapc_to_mesh(output_img,pc_struct,adaptive_max);
//    debug_write(output_img,"adapt_max");
    
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
    
    //adaptive mean
    ExtraPartCellData<float> eng1(pc_struct.part_data.particle_data);
    ExtraPartCellData<float> eng2(pc_struct.part_data.particle_data);
    
    
    //////////////////////////////////
    //
    // Set node values
    //
    //////////////////////////////////
    
    
    
    uint64_t counter = 0;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
    
//ss#spragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
        
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
                            
                           // cap_s =   alpha*pow(Ip - loc_min,2)/pow(loc_max - loc_min,2);
                           // cap_t =   alpha*pow(Ip-loc_max,2)/pow(loc_max - loc_min,2);
                            
                            
                            if((loc_min > 0) & (loc_max > 0)){
                                cap_s =   alpha*(Ip - loc_min);
                                cap_t =   alpha*(loc_max-Ip);
                            } else {
                                cap_s = 0;
                                cap_t = alpha*Ip;
                            }
                            
                            g.add_tweights(global_part_index,   /* capacities */ cap_s, cap_t);
                            
                            //eng1.get_part(curr_key) = cap_s/alpha + 5000;
                            //eng2.get_part(curr_key) = cap_t/alpha + 5000;
                            
                            //eng1.get_part(curr_key) = loc_min;
                            //eng2.get_part(curr_key) = loc_max;
                            
                            counter++;
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    
//    pc_struct.interp_parts_to_pc(output_img,eng1);
//    debug_write(output_img,"eng1p");
//    pc_struct.interp_parts_to_pc(output_img,eng2);
//    debug_write(output_img,"eng2p");
    
    
    
    
//    //adaptive mean
//    ExtraPartCellData<float> edge1(pc_struct.part_data.particle_data);
//    ExtraPartCellData<float> edge2(pc_struct.part_data.particle_data);
//    ExtraPartCellData<float> edge3(pc_struct.part_data.particle_data);
//    ExtraPartCellData<float> edge4(pc_struct.part_data.particle_data);
//    ExtraPartCellData<float> edge5(pc_struct.part_data.particle_data);
//    ExtraPartCellData<float> edge6(pc_struct.part_data.particle_data);
//    
    
    ////////////////////////////////////
    //
    //  Assign Neighbour Relationships
    //
    /////////////////////////////////////
    
    PartCellNeigh<uint64_t> neigh_part_keys;
    PartCellNeigh<uint64_t> neigh_cell_keys;
    uint64_t node_val_pc;


    uint64_t neigh_counter = 0;

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
                        
                        float depth_curr = i;
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            pc_struct.part_data.get_part_neighs_all(p,node_val_pc,curr_key,status,part_offset,neigh_cell_keys,neigh_part_keys,pc_struct.pc_data);
                            
                            uint64_t global_part_index = pc_struct.part_data.get_global_index(curr_key);
                            uint64_t global_part_index_neigh;
                            uint64_t status_neigh;
                            float depth_neigh;
                            float cap =0;
                            
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
                                        
                                        //cap = beta*(pow(status*status_neigh,2)/pow(9.0,2))*pow(-i+pc_struct.depth_max+1,2)/(1.0*pow(pc_struct.depth_max-pc_struct.depth_min+1,2));
                                        //g.add_edge( global_part_index, global_part_index_neigh,    /* capacities */  cap, cap );
                                        
                                        cap = beta*pow(status*status_neigh,2)/pow(9.0,2);
                                        g.add_edge( global_part_index, global_part_index_neigh,    /* capacities */  cap, cap );

                                        neigh_counter++;
                                    }
                                }
                                
//                                if(face ==0){
//                                    edge1.get_part(curr_key)=cap;
//                                } else if(face ==1){
//                                     edge2.get_part(curr_key)=cap;
//                                } else if(face ==2){
//                                     edge3.get_part(curr_key)=cap;
//                                }else if(face ==3){
//                                     edge4.get_part(curr_key)=cap;
//                                }else if(face ==4){
//                                     edge5.get_part(curr_key)=cap;
//                                }else if(face ==5){
//                                     edge6.get_part(curr_key)=cap;
//                                }
//                                
                                
                            }
                            
                            
                        }
                    }
                }
            }
        }
    }

    analysis_data.add_float_data("part_num_neigh",(float)neigh_counter);


//    pc_struct.interp_parts_to_pc(output_img,edge1);
//    debug_write(output_img,"edge1");
//    
//    pc_struct.interp_parts_to_pc(output_img,edge2);
//    debug_write(output_img,"edge2");
//    
//    pc_struct.interp_parts_to_pc(output_img,edge3);
//    debug_write(output_img,"edge3");
//    
//    pc_struct.interp_parts_to_pc(output_img,edge4);
//    debug_write(output_img,"edge4");
//    
//    pc_struct.interp_parts_to_pc(output_img,edge5);
//    debug_write(output_img,"edge5");
//    
//    pc_struct.interp_parts_to_pc(output_img,edge6);
//    debug_write(output_img,"edge6");
    
    
    
}
template<typename T,typename V>
void calc_graph_cuts_segmentation(PartCellStructure<V,T>& pc_struct,ExtraPartCellData<uint8_t>& seg_parts,std::array<uint64_t,10> parameters,AnalysisData& analysis_data){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Input the structure it outputs an extra data structure with 0,1 on the particles if in out the region
    //
    //
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    
    uint64_t num_parts = pc_struct.get_number_parts();
    
    GraphType *g = new GraphType(num_parts ,num_parts*6.4 );
    
    timer.start_timer("construct_graph_parts");
    
    construct_max_flow_graph(pc_struct,*g,parameters,analysis_data);
    
    timer.stop_timer();
    

    timer.start_timer("max_flow_parts");
    
    int flow = g -> maxflow();
    
    //int flow2 = g -> maxflow(true);
    
    timer.stop_timer();

    
    //now put in structure
    
    seg_parts.initialize_structure_parts(pc_struct.part_data.particle_data);
    
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
    uint64_t p=0;
    
    //////////////////////////////////
    //
    // Set node values
    //
    //////////////////////////////////
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) if(z_num_*x_num_ > 100)
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
                        
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        uint64_t global_part_index;
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            
                            global_part_index = pc_struct.part_data.get_global_index(curr_key);
                            float temp = 255*(g->what_segment((int)global_part_index) == GraphType::SOURCE);
                            //float temp2 = seg_parts.get_part(curr_key);
                            
                            seg_parts.get_part(curr_key) = temp;
                            
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }


    delete g;

    analysis_data.add_timer(timer);
    
    
}

template<typename T,typename V>
void calc_graph_cuts_segmentation(PartCellStructure<V,T>& pc_struct,ExtraPartCellData<uint8_t>& seg_parts,std::array<uint64_t,10> parameters){
    //
    //  Creating interface without analysis_data
    //
    //
    AnalysisData analysis_data;

    calc_graph_cuts_segmentation(pc_struct,seg_parts,parameters,analysis_data);


};



template<typename T,typename V>
void calc_graph_cuts_segmentation_mesh(PartCellStructure<V,T>& pc_struct,Mesh_data<uint8_t>& seg_mesh,std::array<uint64_t,10> parameters,AnalysisData& analysis_data){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Input the structure it outputs an mesh with 0,1, uses the pixel image to compute the segmentation
    //
    //
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    
    uint64_t num_parts = pc_struct.org_dims[0]*pc_struct.org_dims[1]*pc_struct.org_dims[2];
    
    GraphType *g = new GraphType(num_parts ,num_parts*5.95 );
    
    timer.start_timer("construct_graph_mesh");
    
    construct_max_flow_graph_mesh(pc_struct,*g,parameters,analysis_data);
    
    timer.stop_timer();
    
    
    timer.start_timer("max_flow_mesh");
    
    int flow = g -> maxflow();
    
    //int flow2 = g -> maxflow(true);
    
    timer.stop_timer();

    //now put in structure
    
    //////////////////////////////////
    //
    // Set node values
    //
    //////////////////////////////////
    
    seg_mesh.initialize(pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],0);
    
    for(int i = 0;i < seg_mesh.mesh.size();i++){
        seg_mesh.mesh[i] = 255*(g->what_segment(i) == GraphType::SOURCE);
    }
    

    
    delete g;

    analysis_data.add_timer(timer);
    
    
}

template<typename T,typename V>
void calc_graph_cuts_segmentation_mesh(PartCellStructure<V,T>& pc_struct,Mesh_data<uint8_t>& seg_mesh,std::array<uint64_t,10> parameters){
    //
    //  Creating interface without analysis_data
    //
    //
    AnalysisData analysis_data;

    calc_graph_cuts_segmentation_mesh(pc_struct,seg_mesh,parameters,analysis_data);

};


#endif
