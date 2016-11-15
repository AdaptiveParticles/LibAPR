#ifndef _graph_cut_h
#define _graph_cut_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"



typedef Graph<float,float,float> GraphType;

void construct_max_flow_graph(PartCellStructure<float,uint64_t>& pc_struct,GraphType& g){
    //
    //  Constructs naiive max flow model for APR
    //
    //
    
    int k_diff = 1;
    
    Part_timer timer;
    
    
    int num_parts = pc_struct.get_part_num();
    int num_cells = pc_struct.get_cell_num();
    
    std::cout << "Got part neighbours" << std::endl;
    
    //Get the other part rep information
    
    std::vector<uint16_t> empty;
    
    
    float beta = 8;
    float k_max = p_rep.pl_map.k_max;
    float k_min = p_rep.pl_map.k_min;
    float alpha = 100;
    
    for(int i = 0; i < num_parts; i++){
        //adds the node
        g.add_node();
        
    }
    
    
   
    //Loop over parts here
    
    
                float loc_min = push_min_data[cell_ref.second];
                float loc_max = push_max_data[cell_ref.second];
                
                for(int p = curr_index.first; p < curr_index.last;p++){
                    
                    float cap_s =   alpha*pow(p_rep.Ip.data[p] - loc_min,2)/pow(loc_max - loc_min,2);
                    float cap_t =   alpha*pow(p_rep.Ip.data[p]-loc_max,2)/pow(loc_max - loc_min,2);
                    
                    g.add_tweights( p,   /* capacities */ cap_s, cap_t);
                }
    
    
    
    
    
    
    
    //vec_e.resize(p_rep.get_part_num());
    timer.verbose_flag = true;
    
    
    
    for(int i = 0; i < num_parts; i++){
        for(int j = 0; j < verlet_list[i].size();j++){
            
            
            float cap = beta*pow(type_vec.data[i]*type_vec.data[j],2)*pow((-k_vec.data[i]+k_max + 1)*(-k_vec.data[j]+k_max + 1),4)/pow((1.0)*(k_max+1-k_min),4.0);
            g.add_edge( i, verlet_list[i][j],    /* capacities */  cap, cap );
            
            
            if(k_vec.data[i] >= (k_max-1)){
                float cap = beta*pow(type_vec.data[i]*type_vec.data[j],2)*pow((-k_vec.data[i]+k_max + 1)*(-k_vec.data[j]+k_max + 1),4)/pow((1.0)*(k_max+1-k_min),4.0);
                g.add_edge( i, verlet_list[i][j],    /* capacities */  cap, cap );
                
            }
            else {
                float cap = beta*81.0;
                g.add_edge( i, verlet_list[i][j],    /* capacities */  cap, cap );
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
