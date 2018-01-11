#ifndef _compression_h
#define _compression_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "src/data_structures/APR/ExtraPartCellData.hpp"
#include "benchmarks/development/Tree/PartCellParent.hpp"

template<typename T>
void get_wavelet_parts(std::vector<T>& parts,uint8_t& scale){
    //
    //  Calculates the intensity values from wavelet co-efficients
    //
    
    constexpr  float t_coeffs[8][8] = {{8.0,1,1,1,1,1,1,1},
        {8.0,-1,1,-1,1,-1,1,-1},
        {8.0,1,-1,-1,1,1,-1,-1},
        {8.0,-1,-1,1,1,-1,-1,1},
        {8.0,1,1,1,-1,-1,-1,-1},
        {8.0,-1,1,-1,-1,1,-1,1},
        {8.0,1,-1,-1,-1,-1,1,1},
        {8.0,-1,-1,1,-1,1,1,-1}};
    
    float temp_scale = pow(2.0,scale);
    
    //scale quantization reversal
    for(int j = 1;j <8;j++){
        parts[j]=parts[j]*temp_scale;
    }
    
    std::vector<T> local_int;
    local_int.resize(8,0);
    std::swap(parts,local_int);
    
    //compute the wavelet co-eff
    for (int j = 0; j < 8; j++) {
        //calculate the wavelet co-efficients (diff, pushing up the mean)
        parts[j] = (1/8.0f)*((local_int[0]*t_coeffs[j][0] + local_int[1]*t_coeffs[j][1] + local_int[2]*t_coeffs[j][2] + local_int[3]*t_coeffs[j][3] + local_int[4]*t_coeffs[j][4] + local_int[5]*t_coeffs[j][5] + local_int[6]*t_coeffs[j][6] + local_int[7]*t_coeffs[j][7]));
    }
    
    
}
template<typename T>
void get_wavelet_coeffs(std::vector<float>& parts,uint8_t& scale,T& mean,float comp_factor,bool flag){
    //
    //  Calculates the wavelet co-efficients
    //
    //
    //
    
    constexpr  int tranform_coeffs[8][8] = {{1,1,1,1,1,1,1,1},
            {1,-1,1,-1,1,-1,1,-1},
            {1,1,-1,-1,1,1,-1,-1},
            {1,-1,-1,1,1,-1,-1,1},
            {1,1,1,1,-1,-1,-1,-1},
            {1,-1,1,-1,-1,1,-1,1},
            {1,1,-1,-1,-1,-1,1,1},
            {1,-1,-1,1,-1,1,1,-1}};
    
    float counter=1;
    
    //this means it will only work for non zero input
    if (flag == true){
        counter = 8.0;
    } else {
        for(int j = 1; j < 8; j++){
            counter += (parts[j] != 0);
        }
    }
    
    std::vector<float> local_int;
    local_int.resize(8,0);
    std::swap(parts,local_int);
    
    //first calculate the mean to push up the tree
    T temp_mean = (1.0/counter)*(local_int[0] + local_int[1] + local_int[2] + local_int[3] + local_int[4] + local_int[5] + local_int[6] + local_int[7]);
    
    //compute the wavelet co-eff
    for (int j = 1; j < 8; j++) {
        //calculate the wavelet co-efficients (diff, pushing up the mean)
        parts[j] = ((local_int[0]*tranform_coeffs[j][0] + local_int[1]*tranform_coeffs[j][1] + local_int[2]*tranform_coeffs[j][2] + local_int[3]*tranform_coeffs[j][3] + local_int[4]*tranform_coeffs[j][4] + local_int[5]*tranform_coeffs[j][5] + local_int[6]*tranform_coeffs[j][6] + local_int[7]*tranform_coeffs[j][7]));
    }
    
    //
    //  Perform Quantization (Lossy Compression)
    //
    
    
    float temp_min = 9999999;
    float temp_max = -999999;
    
    //calculate scale
    for(int j = 1;j < 8;j++){
        temp_min = std::min((float) parts[j],temp_min);
        temp_max = std::max((float) parts[j],temp_max);
    }
    
    float temp = (temp_max-temp_min)/comp_factor;
    
    scale = std::max(0.0,floor(log((temp_max-temp_min)/comp_factor)/log(2)));
    
    //mean = ceil(temp_mean/pow(2.0,scale));
    mean = temp_mean;
    
    for(int j = 1;j < 8;j++){
        parts[j] = ceil(parts[j]/pow(2.0,scale));
    }
    
    
}


template <typename T,typename S,typename Q>
void calc_wavelet_encode(PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<uint8_t>& scale,ExtraPartCellData<Q>& q,ExtraPartCellData<uint8_t>& scale_parent,ExtraPartCellData<T>& mu_parent,ExtraPartCellData<Q>& q_parent,const float comp_factor){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates a truncated and quantized wavelet transform
    //
    
    Part_timer timer;
    timer.verbose_flag = true;
    
    //get the parents
    PartCellParent<uint64_t> pc_parent(pc_struct);
    
    std::cout << "Number parent cels: " << pc_parent.get_cell_num() << std::endl;
    
    scale.initialize_structure_cells(pc_struct.pc_data); //cell structure
    q.initialize_structure_parts(pc_struct.part_data.particle_data); //particle structure
    ExtraPartCellData<T> mu(pc_struct.pc_data);  //cell structure
    
    scale_parent.initialize_structure_cells(pc_parent.neigh_info); //cell structure
    mu_parent.initialize_structure_cells(pc_parent.neigh_info);  //cell structure
    q_parent.initialize_structure_cells(pc_parent.neigh_info);  //cell structure
    
    ///////////////////////////////////
    //
    //
    //  Compute SEED cell haar wavelet transform
    //
    //
    ///////////////////////////////////
    
    
    //first loop of seed cells and do calculations..
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

    
    std::vector<float> parts;
    parts.resize(8);
    
    timer.start_timer("SEED LOOP");
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_part,curr_key,status,part_offset) firstprivate(parts) if(z_num_*x_num_ > 100)
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
                        
                        if(i == pc_struct.pc_data.depth_min){
                            
                            if(status == SEED){
                                
                                float max_t = 0;
                                uint8_t scale_t = 0;
                                for(int p = 0;p < 8;p++){
                                    max_t = std::max(max_t,(float)pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset+p]);
                                }
                            
                                scale_t = std::max(0.0,floor(log((max_t)/comp_factor)/log(2)));

                                for(int p = 0;p < 8;p++){
                                    q.data[i][offset_pc_data][part_offset+p] = ceil(pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset+p]/pow(2.0,scale_t));
                                }
                                scale.data[i][offset_pc_data][j_] = scale_t;
                            } else {
                                uint8_t scale_t = 0;
                                scale_t = std::max(0.0,floor(log((pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset])/comp_factor)/log(2)));
                                q.data[i][offset_pc_data][part_offset] = ceil(pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset]/pow(2.0,scale_t));
                                scale.data[i][offset_pc_data][j_] = scale_t;
                            }
                            
                        } else {
                            
                            if(status == SEED){
                                
                                std::copy(pc_struct.part_data.particle_data.data[i][offset_pc_data].begin() + part_offset,pc_struct.part_data.particle_data.data[i][offset_pc_data].begin() + part_offset + 8,parts.begin());
                                
                                uint8_t scale_t = 0;
                                T mean = 0;
                                
                                get_wavelet_coeffs(parts,scale_t,mean,comp_factor,true);
                                
                                //copy to q particle data
                                std::copy(parts.begin()+1,parts.end(),q.data[i][offset_pc_data].begin()+part_offset+1);
                                mu.data[i][offset_pc_data][j_] = mean;
                                scale.data[i][offset_pc_data][j_] = scale_t;
                                
                                
                                
                                
                            } else {
                                mu.data[i][offset_pc_data][j_] = pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset];
                            }
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                }
            }
        }
    }

    timer.stop_timer();
    
    
    /////////////////////////
    //
    //  Parent Loop
    //
    ////////////////////////////
    
    uint64_t node_val_parent;
    std::vector<uint64_t> children_keys;
    std::vector<uint64_t> children_ind;
    
    children_keys.resize(8,0);
    children_ind.resize(8,0);
    
    
    
    timer.start_timer("PARENT LOOP");
    
    //reverse loop direction
    for(uint64_t i = pc_parent.neigh_info.depth_max;i >= pc_parent.neigh_info.depth_min;i--){
        //loop over the resolutions of the structure
        const unsigned int x_num_ =  pc_parent.neigh_info.x_num[i];
        const unsigned int z_num_ =  pc_parent.neigh_info.z_num[i];
        
        
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,part_offset) firstprivate(parts,children_keys,children_ind) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_parent.neigh_info.pc_key_set_z(curr_key,z_);
            pc_parent.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_parent.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_parent.neigh_info.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_parent = pc_parent.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_parent&1)){
                        //Indicates this is a particle cell node
                        
                        if((x_ == 14) & (z_==3) & (j_==5) & (i==5)){
                            int stop = 1;
                        }
                        
                        pc_parent.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        status = pc_parent.neigh_info.get_status(node_val_parent);
                        
                        //parent has real siblings
                        if(status == 2){
                            
                            //get the children
                            
                            pc_parent.get_children_keys(curr_key,children_keys,children_ind);
                            
                            for(int c = 0;c < children_keys.size();c++){
                                uint64_t child = children_keys[c];
                                
                                if(child > 0){
                                    
                                    if(children_ind[c] == 1){
                                        //
                                        parts[c] = mu.get_val(child);
                                    } else {
                                        parts[c] = mu_parent.get_val(child);
                                    }
                                    
                                    
                                } else {
                                    parts[c] = 0;
                                }
                            }
                            
                            uint8_t scale_t = 0;
                            T mean = 0;
                            
                            //debug here this function with matlab
                            get_wavelet_coeffs(parts,scale_t,mean,comp_factor,false);
                            
                            scale_parent.get_val(curr_key) = scale_t;
                            mu_parent.get_val(curr_key) = mean;
                            
                            //loop over children again and put the q in place
                            
                            for(int c = 1;c < children_keys.size();c++){
                                uint64_t child = children_keys[c];
                                
                                if(child > 0){
                                    
                                    if(children_ind[c] == 1){
                                        //
                                        
                                        uint64_t child_node = pc_struct.part_data.access_data.get_val(child);
                                        
                                        
                                        part_offset = pc_struct.part_data.access_node_get_part_offset(child_node);
                                        pc_struct.part_data.access_data.pc_key_set_index(child,part_offset);
                                        
                                        q.get_part(child) = parts[c];
                                        
                                    } else {
                                        
                                        q_parent.get_val(child) = parts[c];
                                        
                                        
                                        
                                    }
                                    
                                } else {
                                    // Need to think on this case
                                    
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
    
    timer.stop_timer();
    

}

template <typename S,typename Q>
void calc_wavelet_decode(PartCellStructure<S,uint64_t>& pc_struct,std::vector<std::vector<uint8_t>>& scale_out,std::vector<std::vector<Q>>& q_out,std::vector<std::vector<uint8_t>>& scale_parent_out,std::vector<std::vector<uint16_t>>& mu_parent_out,std::vector<std::vector<Q>>& q_parent_out,const float comp_factor){
    //
    //  Bevan Cheeseman 2016
    //
    //  Decodes a truncated and quantized haar wavelet transform
    //
    
    Part_timer timer;
    timer.verbose_flag = true;
    
    //get the parents
    PartCellParent<uint64_t> pc_parent(pc_struct);
    
    ExtraPartCellData<uint16_t> mu(pc_struct.pc_data);  //cell structure
    
    
    ExtraPartCellData<int16_t> q(pc_struct.part_data.particle_data); //particle size
    
    ExtraPartCellData<uint8_t> scale(pc_struct.pc_data); //cell size
    
    ExtraPartCellData<uint8_t> scale_parent(pc_parent.neigh_info); //parent size
    ExtraPartCellData<uint16_t> mu_parent(pc_parent.neigh_info); //parent size
    ExtraPartCellData<int16_t> q_parent(pc_parent.neigh_info); // parent size
    
    q.initialize_data(q_out);
    scale.initialize_data(scale_out);
    scale_parent.initialize_data(scale_parent_out);
    mu_parent.initialize_data(mu_parent_out);
    q_parent.initialize_data(q_parent_out);
    
    //free up the memory
    
    
    for(uint64_t i = pc_struct.depth_min;i <= pc_struct.depth_max;i++){
        std::vector<Q>().swap(q_out[i]);
        std::vector<uint8_t>().swap(scale_out[i]);
    }
    
    for(uint64_t i = pc_struct.depth_min;i < pc_struct.depth_max;i++){
        std::vector<Q>().swap(q_parent_out[i]);
        std::vector<uint8_t>().swap(scale_parent_out[i]);
        std::vector<uint16_t>().swap(mu_parent_out[i]);
    }
    
    ///////////////////////////////////
    //
    //
    //  Reconstruct from parents loop
    //
    //
    ///////////////////////////////////

    /////////////////////////
    //
    //  Parent Loop
    //
    ////////////////////////////
    
    uint64_t z_;
    uint64_t x_;
    uint64_t j_;
    uint64_t curr_key;
    uint64_t part_offset;
    uint64_t status;
    
    
    uint64_t node_val_parent;
    std::vector<uint64_t> children_keys;
    std::vector<uint64_t> children_ind;
    
    children_keys.resize(8,0);
    children_ind.resize(8,0);
    
    std::vector<float> parts;
    std::vector<float> child_parts;
    uint8_t scale_parts;
    
    uint64_t child_status;
    uint64_t child_node;
    
    parts.resize(8,0);
    child_parts.resize(8,0);
    
    timer.start_timer("PARENT LOOP DECODE");
    
    //reverse loop direction
    for(uint64_t i = pc_parent.neigh_info.depth_min;i <= pc_parent.neigh_info.depth_max;i++){
        //loop over the resolutions of the structure
        const unsigned int x_num_ =  pc_parent.neigh_info.x_num[i];
        const unsigned int z_num_ =  pc_parent.neigh_info.z_num[i];
        
//#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_parent,curr_key,status,part_offset,child_status,child_node) firstprivate(children_keys,children_ind,parts,child_parts) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            //both z and x are explicitly accessed in the structure
            curr_key = 0;
            
            pc_parent.neigh_info.pc_key_set_z(curr_key,z_);
            pc_parent.neigh_info.pc_key_set_depth(curr_key,i);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_parent.neigh_info.pc_key_set_x(curr_key,x_);
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_parent.neigh_info.data[i][offset_pc_data].size();
                
                //the y direction loop however is sparse, and must be accessed accordinagly
                for(j_ = 0;j_ < j_num;j_++){
                    
                    
                    //particle cell node value, used here as it is requried for getting the particle neighbours
                    node_val_parent = pc_parent.neigh_info.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_parent&1)){
                        //Indicates this is a particle cell node
                        
                        pc_parent.neigh_info.pc_key_set_j(curr_key,j_);
                        
                        status = pc_parent.neigh_info.get_status(node_val_parent);
                        
                        //parent has real siblings
                        if(status == 2){
                            
                            //get the children
                            
                            pc_parent.get_children_keys(curr_key,children_keys,children_ind);
                            
                            parts[0] = mu_parent.get_val(curr_key);
                            scale_parts = scale_parent.get_val(curr_key);
                            
                            //ignore first value it is from the mean
                            for(int c = 1;c < children_keys.size();c++){
                                uint64_t child = children_keys[c];
                                
                                if(child > 0){
                                    
                                    if(children_ind[c] == 1){
                                        
                                        
                                        
                                        child_node = pc_struct.part_data.access_data.get_val(child);
                                        child_status = pc_struct.part_data.access_node_get_status(child_node);
                                        
                                        part_offset = pc_struct.part_data.access_node_get_part_offset(child_node);
                                        pc_struct.part_data.access_data.pc_key_set_index(child,part_offset);
                                        
                                        
                                        parts[c] = q.get_part(child);
                                        
                                    } else {
                                        parts[c] = q_parent.get_val(child);
                                    }
                                    
                                    
                                } else {
                                    parts[c] = 0;
                                }
                            }
                            
                            //convert to intensities
                            get_wavelet_parts(parts,scale_parts);
                            
                                                       
                            
                            //ignore first value it is from the mean
                            for(int c = 0;c < children_keys.size();c++){
                                uint64_t child = children_keys[c];
                                
                                if(child > 0){
                                    
                                    if(children_ind[c] == 1){
                                        //
                                        
                                        //
                                        child_node = pc_struct.part_data.access_data.get_val(child);
                                        child_status = pc_struct.part_data.access_node_get_status(child_node);
                                        
                                        part_offset = pc_struct.part_data.access_node_get_part_offset(child_node);
                                        pc_struct.part_data.access_data.pc_key_set_index(child,part_offset);

                                        
                                        if(child_status == SEED){
                                            //why not just call it agian here?
                                            child_parts[0] = parts[c];
                                            
                                            //get the values from offset
                                            uint64_t x_c;
                                            uint64_t z_c;
                                            uint64_t j_c;
                                            uint64_t depth_c;
                                            uint8_t child_scale_parts = scale.get_val(child);
                                            
                                            pc_struct.pc_data.get_details_cell(child,x_c,z_c,j_c,depth_c);
                                            
                                            size_t offset_part_data = pc_struct.x_num[depth_c]*z_c + x_c;
                                            
                                            
                                            std::copy(q.data[depth_c][offset_part_data].begin() + part_offset + 1,q.data[depth_c][offset_part_data].begin() + part_offset + 8,child_parts.begin() + 1);
                                            
                                            //convert to intensities
                                            get_wavelet_parts(child_parts,child_scale_parts);
                                            
                                            //then add them into the structure
                                            std::copy(child_parts.begin(),child_parts.begin() + 8,pc_struct.part_data.particle_data.data[depth_c][offset_part_data].begin() + part_offset);
                                            
                                            
                                            
                                        } else {
                                            //if the node is boundary or filler this is the last step
                                            
                                            
                                            pc_struct.part_data.particle_data.get_part(child) = parts[c];
                                            
                                        }

                                       
                                        
                                    }
                                    
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
    
    timer.stop_timer();
    //low res to high res
    
    //now need to loop over highest level
    
    uint64_t node_val_part;
    uint64_t scale_t;
    uint64_t i = pc_struct.pc_data.depth_min;
    
    const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
    const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
    
    
    //#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_part,curr_key,status,part_offset,scale_t,node_val_part) firstprivate(parts) if(z_num_*x_num_ > 100)
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
                    
                    if(status == SEED){
                        
                        scale_t =scale.data[i][offset_pc_data][j_] ;
                        
                        for(int p = 0;p < 8;p++){
                            pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset+p]=pow(2.0,scale_t)*q.data[i][offset_pc_data][part_offset+p];
                        }
                        
                    } else {
                        scale_t= scale.data[i][offset_pc_data][j_] ;
                        pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset]=pow(2.0,scale_t)*q.data[i][offset_pc_data][part_offset];
                    }
                    
                } else {
                    // Inidicates this is not a particle cell node, and is a gap node
                }
            }
        }
    }



}






#endif