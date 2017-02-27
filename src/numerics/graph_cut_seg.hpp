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
                            
                            //g.add_tweights(global_part_index,   /* capacities */ cap_s, cap_t);
                            
                            eng1.get_part(curr_key) =cap_s;
                            eng2.get_part(curr_key) =cap_t;
                            
                            counter++;
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    Mesh_data<float> eng_s;
    Mesh_data<float> eng_t;
    
    pc_struct.interp_parts_to_pc(eng_s,eng1);
    //debug_write(eng_s,"eng1_mesh");
    pc_struct.interp_parts_to_pc(eng_t,eng2);
    //debug_write(eng_t,"eng2_mesh");
    
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

                            eng1.get_part(curr_key) = cap_s;
                            eng2.get_part(curr_key) = cap_t;

                            counter++;
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    
   // pc_struct.interp_parts_to_pc(output_img,eng1);
   // debug_write(output_img,"eng1p");
   // pc_struct.interp_parts_to_pc(output_img,eng2);
   // debug_write(output_img,"eng2p");
    
    
    
    
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
template<typename T,typename V,typename U>
void construct_max_flow_graph_new(PartCellStructure<V,T>& pc_struct,GraphType& g,ExtraPartCellData<U>& seg_parts,AnalysisData& analysis_data,float Ip_threshold){
    //
    //  Constructs naiive max flow model for APR
    //
    //

    Part_timer timer;

    pc_struct.part_data.initialize_global_index();


    uint64_t num_parts = pc_struct.get_number_parts();


    //Get the other part rep information


    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;

    part_new.particle_data.org_dims = pc_struct.org_dims;

    ExtraPartCellData<float> particle_data;

    part_new.create_particles_at_cell_structure(particle_data);


    float beta = 1000;
    float k_max = pc_struct.depth_max;
    float k_min = pc_struct.depth_min;
    float alpha = 1;

    for(int i = 0; i < num_parts; i++){
        //adds the node
        g.add_node();

    }

    std::vector<float> filter = {.0125,.975,.0125};
    std::vector<float> delta = {1,1,4};

    int num_tap = 1;

    ExtraPartCellData<float> smoothed_parts = adaptive_smooth(pc_data,particle_data,num_tap,filter);

    ExtraPartCellData<float> smoothed_gradient_mag = adaptive_grad(pc_data,smoothed_parts,3,delta);

    //adaptive mean
    ExtraPartCellData<float> adaptive_min;
    ExtraPartCellData<float> adaptive_max;

    //offsets past on cell status (resolution)
    std::vector<unsigned int> status_offsets_min = {1,2,3};
    std::vector<unsigned int> status_offsets_max = {1,2,3};

    get_adaptive_min_max(pc_struct,adaptive_min,adaptive_max,status_offsets_min,status_offsets_max,0,0);

    Mesh_data<float> output_img;

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

    ExtraPartCellData<float> new_adaptive_max;
    ExtraPartCellData<float> new_adaptive_min;

    new_adaptive_min.initialize_structure_parts(pc_struct.part_data.particle_data);
    new_adaptive_max.initialize_structure_parts(pc_struct.part_data.particle_data);



    //////////////////////////////////
    //
    // First convert to particle location data
    //
    //////////////////////////////////

    uint64_t counter = 0;

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

                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);

                        float loc_min = adaptive_min.get_val(curr_key);
                        float loc_max = adaptive_max.get_val(curr_key);



                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering

                            new_adaptive_min.get_part(curr_key) = loc_min;
                            new_adaptive_max.get_part(curr_key) = loc_max;

                        }
                    }

                }

            }

        }
    }


    convert_from_old_structure(adaptive_max,pc_struct,pc_data,new_adaptive_max,false);
    part_new.create_particles_at_cell_structure(new_adaptive_max,adaptive_max);

    convert_from_old_structure(adaptive_min,pc_struct,pc_data,new_adaptive_min,false);
    part_new.create_particles_at_cell_structure(new_adaptive_min,adaptive_min);


    ExtraPartCellData<uint64_t> part_id;
    part_id.initialize_structure_cells(pc_data);

    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information

    // Extra variables required
    //

    //check everything
//
//    Mesh_data<float> check;
//
//    interp_img(check, pc_data, part_new, new_adaptive_min,true);
//
//    debug_write(check,"min");
//
//    interp_img(check, pc_data, part_new, new_adaptive_max,true);
//
//    debug_write(check,"max");
//
//    interp_img(check, pc_data, part_new, smoothed_parts,true);
//
//    debug_write(check,"Ip");

    float Ip_min = Ip_threshold;

    counter = 0;

    for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[i];
        const unsigned int z_num_ = pc_data.z_num[i];
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            pc_data.pc_key_set_z(curr_key,z_);
            pc_data.pc_key_set_depth(curr_key,i);

            for(x_ = 0;x_ < x_num_;x_++){

                pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_data.data[i][offset_pc_data].size();

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        //y_coord++;


                        pc_data.pc_key_set_j(curr_key,j_);

                        part_id.get_val(curr_key) = counter;

                        float Ip = smoothed_parts.get_val(curr_key);
                        float loc_min = new_adaptive_min.get_val(curr_key);
                        float loc_max = new_adaptive_max.get_val(curr_key);

                        float cap_s;
                        float cap_t;

                        if((loc_min > 0) & (loc_max > 0) & (Ip > Ip_min)){
                            cap_s =   alpha*abs((Ip - loc_min)/(loc_max-loc_min));
                            cap_t =   alpha*abs((loc_max-Ip)/(loc_max-loc_min));

                            cap_s =   alpha*abs((Ip - loc_min));
                            cap_t =   alpha*abs((loc_max-Ip));

                            //cap_s =   alpha/(1+exp(-(cap_s-(loc_max*.5 + loc_min*.5))));
                            //cap_t =  alpha/(1+exp(-(cap_t -(loc_max*.5 + loc_min*.5))));

                        } else {
                            cap_s = 0;
                            cap_t = 100000;
                        }


                        g.add_tweights(counter,   /* capacities */ cap_s, cap_t);

                        counter++;


                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node

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

    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //

    uint64_t  neigh_counter= 0;

    timer.verbose_flag = false;


    timer.start_timer("neigh_cell_comp");




    for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){


        float h_depth = pow(2,pc_data.depth_max - i);

        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[i];
        const unsigned int z_num_ = pc_data.z_num[i];
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            pc_data.pc_key_set_z(curr_key,z_);
            pc_data.pc_key_set_depth(curr_key,i);


            for(x_ = 0;x_ < x_num_;x_++){

                pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_data.data[i][offset_pc_data].size();

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node
                        //y_coord++;

                        pc_data.pc_key_set_j(curr_key,j_);

                        pc_data.get_neighs_all(curr_key,node_val_pc,neigh_cell_keys);

                        uint64_t global_index = part_id.get_val(curr_key);

                        float grad = smoothed_gradient_mag.get_val(curr_key);
                        float loc_min = new_adaptive_min.get_val(curr_key);
                        float loc_max = new_adaptive_max.get_val(curr_key);
                        float scale = abs(loc_max - loc_min);
                        float Ip = smoothed_parts.get_val(curr_key);

                        if((loc_min == 0) || (loc_max == 0)){
                            scale = 1000;
                        }

                        std::vector<int> dirs = {0,1,2,3,4,5};

                        //(+direction)
                        //loop over the nieghbours
                        for (int d = 0; d < dirs.size(); ++d) {

                            for (int n = 0; n < neigh_cell_keys.neigh_face[dirs[d]].size(); n++) {
                                // Check if the neighbour exisits (if neigh_cell_value = 0, the neighbour doesn't exist)
                                uint64_t neigh_key = neigh_cell_keys.neigh_face[dirs[d]][n];

                                float delta_s;

                                if(dirs[d]>4){
                                    delta_s = delta[2];
                                } else{
                                    delta_s = delta[0];
                                }


                                if (neigh_key > 0) {
                                    //get information about the nieghbour (need to provide face and neighbour number (n) and the current y coordinate)

                                    uint64_t ndepth = pc_data.pc_key_get_depth(neigh_key);
                                    float h = .5*pow(2,pc_data.depth_max - ndepth) + .5*h_depth;

                                    float Iq = smoothed_parts.get_val(neigh_key);
                                    float grad_n = smoothed_gradient_mag.get_val(neigh_key);
                                    uint64_t global_index_neigh = part_id.get_val(neigh_key);

                                    //scale = 1;

                                    float cap_1;
                                    float cap_2;

                                    if((Ip - Iq)>0.05*scale){
                                        cap_1 = 1;
                                        cap_2 = exp(-10*abs(Ip - Iq)/(h*delta_s));


                                    } else {

                                        cap_2 = 1;
                                        cap_1 = exp(-10*abs(Ip - Iq)/(h*delta_s));

                                    }

                                    grad = abs(Ip - Iq)/(h*delta_s*.1);
                                    grad_n = abs(Ip - Iq)/(h*delta_s*.1);


                                    g.add_edge( global_index, global_index_neigh,    /* capacities */  beta*exp(-10*(grad)/pow(scale,1)), beta*exp(-10*(grad_n)/pow(scale,1)));

                                    //g.add_edge( global_index, global_index_neigh,   /* capacities */  beta*cap_1, beta*cap_2);

                                    neigh_counter++;

                                }

                            }

                        }



                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node

                    }

                }

            }

        }
    }


    seg_parts.initialize_structure_cells(pc_data);


    //float temp2 = seg_parts.get_part(curr_key);
    timer.start_timer("max flow");
    int flow = g.maxflow();
    timer.stop_timer();



    for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++){

        float h_depth = pow(2,pc_data.depth_max - i);
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[i];
        const unsigned int z_num_ = pc_data.z_num[i];
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_pc,curr_key)  firstprivate(neigh_cell_keys) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;

            pc_data.pc_key_set_z(curr_key,z_);
            pc_data.pc_key_set_depth(curr_key,i);

            for(x_ = 0;x_ < x_num_;x_++){

                pc_data.pc_key_set_x(curr_key,x_);

                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_data.data[i][offset_pc_data].size();

                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){

                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_pc = pc_data.data[i][offset_pc_data][j_];

                    if (!(node_val_pc&1)){
                        //Indicates this is a particle cell node

                        pc_data.pc_key_set_j(curr_key,j_);

                        part_id.get_val(curr_key);

                        uint64_t global_part_index = part_id.get_val(curr_key);

                        float temp = 255*(g.what_segment((int)global_part_index) == GraphType::SOURCE);

                        seg_parts.get_val(curr_key) = temp;


                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node

                    }

                }

            }

        }
    }

    analysis_data.add_float_data("part_num_neigh_new",(float)neigh_counter);



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

template<typename T,typename V,typename U>
void calc_graph_cuts_segmentation_new(PartCellStructure<V,T>& pc_struct,ExtraPartCellData<U>& seg_parts,AnalysisData& analysis_data,float Ip_threshold){
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

    construct_max_flow_graph_new(pc_struct,*g,seg_parts,analysis_data,Ip_threshold);

    timer.stop_timer();

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
