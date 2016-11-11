/////////////////////
//
//  Loads in and creates test datasets
//
//
/////////////////

#include "create_part_cell_structure.hpp"

void CreateSphereTest::SetUp(){

    std::string test_dir =  get_source_directory();

    std::string name = "files/partcell_files/test_sphere1_pcstruct_part.h5";

    //output
    std::string file_name = test_dir + name;

    read_apr_pc_struct(pc_struct,file_name);
    
    //Now we need to generate the particle map
    particle_map.k_max = pc_struct.depth_max;
    particle_map.k_min = pc_struct.depth_min;
    
    //initialize looping vars
    uint64_t x_;
    uint64_t y_coord;
    uint64_t z_;
    uint64_t j_;
    uint64_t node_val_part;
    uint64_t status;
    
    particle_map.layers.resize(pc_struct.depth_max+1);
    particle_map.downsampled.resize(pc_struct.depth_max+2);
    
    particle_map.downsampled[pc_struct.depth_max + 1].x_num = pc_struct.org_dims[1];
    particle_map.downsampled[pc_struct.depth_max + 1].y_num = pc_struct.org_dims[0];
    particle_map.downsampled[pc_struct.depth_max + 1].z_num = pc_struct.org_dims[2];
    particle_map.downsampled[pc_struct.depth_max + 1].mesh.resize(pc_struct.org_dims[1]*pc_struct.org_dims[0]*pc_struct.org_dims[2]);
    
    std::cout << "DIM1: " << pc_struct.org_dims[1] << std::endl;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        particle_map.layers[i].mesh.resize(x_num_*z_num_*y_num_,0);
        particle_map.layers[i].x_num = x_num_;
        particle_map.layers[i].y_num = y_num_;
        particle_map.layers[i].z_num = z_num_;
        
        particle_map.downsampled[i].x_num = x_num_;
        particle_map.downsampled[i].y_num = y_num_;
        particle_map.downsampled[i].z_num = z_num_;
        particle_map.downsampled[i].mesh.resize(x_num_*z_num_*y_num_,0);
        
        // First create the particle map
        for(z_ = 0;z_ < z_num_;z_++){
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //get the index gap node
                        y_coord++;
                        
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        particle_map.layers[i].mesh[offset_p_map + y_coord] = status;
                        
                    } else {
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
        
    }
    
    
    //intensity set up
    // Set the intensities
    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
        
        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
            particle_map.downsampled[depth].mesh[i] = i;
        }
        
    }
    
    
}