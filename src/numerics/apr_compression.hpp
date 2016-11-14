#ifndef _compression_h
#define _compression_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "../data_structures/Tree/PartCellParent.hpp"


template<typename T>
void get_wavelet_coeffs(std::vector<float>& parts,uint8_t& scale,T& mean,float comp_factor){
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
    
    std::vector<float> local_int;
    local_int.resize(8,0);
    std::swap(parts,local_int);
    
    //first calculate the mean to push up the tree
    T temp_mean = (1.0/8.0)*(local_int[0] + local_int[1] + local_int[2] + local_int[3] + local_int[4] + local_int[5] + local_int[6] + local_int[7]);
    
    //compute the wavelet co-eff
    for (int j = 1; j < 8; j++) {
        //calculate the wavelet co-efficients (diff, pushing up the mean)
        parts[j-1] = ((local_int[0]*tranform_coeffs[j][0] + local_int[1]*tranform_coeffs[j][1] + local_int[2]*tranform_coeffs[j][2] + local_int[3]*tranform_coeffs[j][3] + local_int[4]*tranform_coeffs[j][4] + local_int[5]*tranform_coeffs[j][5] + local_int[6]*tranform_coeffs[j][6] + local_int[7]*tranform_coeffs[j][7]));
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
    
    scale = std::max(0.0,floor(log((temp_max-temp_min)/comp_factor)/log(2)));
    
    mean = ceil(temp_mean/pow(2.0,scale));
    
    for(int j = 1;j < 8;j++){
        parts[j] = ceil(parts[j]/pow(2.0,scale));
    }
    
    
}


template <typename T,typename S>
void calc_wavelet_encode(PartCellStructure<S,uint64_t>& pc_struct){
    //
    //
    //
    //
    //
    
    Part_timer timer;
    timer.verbose_flag = true;
    
    float comp_factor = 40;
    
    //get the parents
    PartCellParent<uint64_t> pc_parent(pc_struct);
    
    ExtraPartCellData<uint8_t> scale(pc_struct.pc_data); //cell structure
    ExtraPartCellData<int8_t> q(pc_struct.part_data.particle_data); //particle structure
    ExtraPartCellData<T> mu(pc_struct.pc_data);  //cell structure
    
    ExtraPartCellData<uint8_t> scale_parent(pc_parent.neigh_info); //cell structure
    ExtraPartCellData<T> mu_parent(pc_parent.neigh_info);  //cell structure
    
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
        
        
//#pragma omp parallel for default(shared) private(p,z_,x_,j_,node_val_part,curr_key,status,part_offset) firstprivate(parts) if(z_num_*x_num_ > 100)
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
                            
                            std::copy(pc_struct.part_data.particle_data.data[i][offset_pc_data].begin() + part_offset,pc_struct.part_data.particle_data.data[i][offset_pc_data].begin() + part_offset + 8,parts.begin());
                            
                            uint8_t scale_t = 0;
                            T mean = pc_struct.part_data.particle_data.data[i][offset_pc_data][part_offset];
                            
                            get_wavelet_coeffs(parts,scale_t,mean,comp_factor);
                            
                            //copy to q particle data
                            std::copy(parts.begin(),parts.end(),q.data[i][offset_pc_data].begin());
                            mu.data[i][offset_pc_data][j_] = mean;
                            scale.data[i][offset_pc_data][j_] = scale_t;
                            
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











//template <typename T>
//void calc_compress_encode(Part_rep& p_rep,Part_data<T>& mu,Part_data<int8_t>& q,Part_data<uint8_t>& scale){
//    //
//    //
//    //  Bevan Cheeseman 2016
//    //
//    //  Quantization Compression Scheme
//    //
//    //  mu: is a vector of means
//    //
//    //  q: quantized haar wavelet detail co-efficients
//    //
//    //  scale: (2^s) adaptive quantization scale
//    //
//    //
//    //  Augmented Status': 7 - Means, 6 - Quantized Haar Wavelet
//    //
//    //
//    //
//    
//    //used to determine which quadrant of your upper level you exist in
//    int position_matrix[2][2][2] = {{{0,4},{1,5}},{{2,6},{3,7}}};
//    int curr_position;
//    
//    
//    // 3D Haar Wavelet Transform Coefficients
//    int tranform_coeffs[8][8] = {{1,1,1,1,1,1,1,1},
//        {1,-1,1,-1,1,-1,1,-1},
//        {1,1,-1,-1,1,1,-1,-1},
//        {1,-1,-1,1,1,-1,-1,1},
//        {1,1,1,1,-1,-1,-1,-1},
//        {1,-1,1,-1,-1,1,-1,1},
//        {1,1,-1,-1,-1,-1,1,1},
//        {1,-1,-1,1,-1,1,1,-1}};
//    
//    Cell_id curr_cell;
//    Cell_id parent_cell;
//    
//    int curr_k = 0;
//    
//    //just to check fill the tree
//    fill_particle_cell_tree(p_rep);
//    
//    //first need to set up the different status structures to perform the correct quantization.
//    
//    //loop through current cells then try to add your parents to fill up the tree
//    for (int i = 0; i < p_rep.get_cell_num();i++) {
//        
//        curr_cell = p_rep.pl_map.cells[i];
//        
//        curr_k = curr_cell.k;
//        
//        
//        
//        if (p_rep.status.data[i] == 2){
//            //
//            //  You will use the haar wavelet transform on the current level, and above will be means
//            //
//            
//            get_parent(curr_cell, parent_cell);
//            
//            auto parent_cell_ref = p_rep.pl_map.pl[parent_cell.k].find(parent_cell);
//            
//            if (parent_cell_ref != p_rep.pl_map.pl[parent_cell.k].end()) {
//                // Doesn't exist add to structure
//                p_rep.status.data[parent_cell_ref->second] = 6;
//                
//                
//            } else {
//                std::cout << "Error: this part cell should exist" << std::endl;
//            }
//            
//        }
//        else if (p_rep.status.data[i] == 4 | p_rep.status.data[i] == 5){
//            
//            get_parent(curr_cell, parent_cell);
//            
//            auto parent_cell_ref = p_rep.pl_map.pl[parent_cell.k].find(parent_cell);
//            
//            if (parent_cell_ref != p_rep.pl_map.pl[parent_cell.k].end()) {
//                // Doesn't exist add to structure
//                if(p_rep.status.data[i] == 0){
//                    
//                    p_rep.status.data[parent_cell_ref->second] = 7;
//                    
//                    curr_cell = parent_cell;
//                    
//                    get_parent(curr_cell, parent_cell);
//                    
//                    auto parent_cell_ref = p_rep.pl_map.pl[parent_cell.k].find(parent_cell);
//                    
//                    if (parent_cell_ref != p_rep.pl_map.pl[parent_cell.k].end()) {
//                        p_rep.status.data[parent_cell_ref->second] = 6;
//                    }
//                    
//                    
//                } else {
//                    //already changed do nothing
//                    
//                    
//                }
//                
//            } else {
//                std::cout << "Error: this part cell should exist" << std::endl;
//            }
//            
//            
//        }
//        
//    }
//    
//    
//    float temp_min;
//    float temp_max;
//    
//    float temp_scale = 0;
//    
//    //////////////////////////////////////////////////////////////////////
//    //
//    //  Compute the coeffs and means
//    //
//    /////////////////////////////////////////////////////////////////////
//    
//    
//    std::vector<std::vector<float>> temp_data(p_rep.get_cell_num());
//    std::vector<uint8_t> counter_vec;
//    //counts the number of children a particular cell has.
//    counter_vec.resize(p_rep.get_cell_num(),0);
//    
//    float temp_mean;
//    unsigned int curr_index;
//    
//    std::vector<float> local_int;
//    local_int.resize(8);
//    
//    //loop over all levels of k
//    for (int k_ = p_rep.pl_map.k_max; k_ >= 0; k_--) {
//        for(auto cell_ref : p_rep.pl_map.pl[k_]) {
//            
//            curr_index = cell_ref.second;
//            curr_cell = p_rep.pl_map.cells[curr_index];
//            
//            
//            
//            if (p_rep.status.data[curr_index] == 2) {
//                //reserve space for the wavelet co-efficients
//                temp_data[curr_index].resize(7);
//                
//                
//                //copy the intensities for the cells
//                std::copy(p_rep.Ip.data.begin() + p_rep.pl_map.cell_indices[curr_index].first,p_rep.Ip.data.begin() + p_rep.pl_map.cell_indices[curr_index].last,local_int.begin());
//                
//                //first calculate the mean to push up the tree
//                temp_mean = (1.0/8.0)*(local_int[0] + local_int[1] + local_int[2] + local_int[3] + local_int[4] + local_int[5] + local_int[6] + local_int[7]);
//                
//                for (int j = 1; j < 8; j++) {
//                    //calculate the wavelet co-efficients (diff, pushing up the mean)
//                    temp_data[curr_index][j-1] = ((local_int[0]*tranform_coeffs[j][0] + local_int[1]*tranform_coeffs[j][1] + local_int[2]*tranform_coeffs[j][2] + local_int[3]*tranform_coeffs[j][3] + local_int[4]*tranform_coeffs[j][4] + local_int[5]*tranform_coeffs[j][5] + local_int[6]*tranform_coeffs[j][6] + local_int[7]*tranform_coeffs[j][7]));
//                }
//                
//                
//                //push the mean up the tree
//                get_parent(p_rep.pl_map.cells[curr_index], parent_cell);
//                
//                auto parent_cell_ref = p_rep.pl_map.pl[parent_cell.k].find(parent_cell);
//                
//                if (parent_cell_ref != p_rep.pl_map.pl[parent_cell.k].end()){
//                    temp_data[parent_cell_ref->second].resize(8);
//                    
//                    curr_position = position_matrix[(curr_cell.y-1)%2][(curr_cell.x-1)%2][(curr_cell.z-1)%2];
//                    
//                    //
//                    //  Perform Mean Quantization
//                    //
//                    
//                    
//                    temp_min = 9999999;
//                    temp_max = -999999;
//                    
//                    //calculate scale
//                    for(int j = 0;j < temp_data[curr_index].size();j++){
//                        temp_min = std::min((float)temp_data[curr_index][j],temp_min);
//                        temp_max = std::max((float)temp_data[curr_index][j],temp_max);
//                    }
//                    
//                    temp_scale = std::max(0.0,floor(log((temp_max-temp_min)/p_rep.pars.comp_scale)/log(2)));
//                    
//                    temp_mean = ceil(temp_mean/pow(2.0,temp_scale));
//                    
//                    temp_data[parent_cell_ref->second][curr_position] = temp_mean;
//                    counter_vec[parent_cell_ref->second]++;
//                    
//                    //temp_data[parent_cell_ref->second].push_back(temp_mean);
//                } else {
//                    //error something is wrong, this cell should exist, otherwise you need to have run fill tree before runnign this step
//                    std::cout << "Warning:: You have a parent cell that does not exist in the structure, check that you have run fill tree correclty ,or something else is broken" << std::endl;
//                    
//                }
//                
//            } else if (p_rep.status.data[curr_index] == 7){
//                //in-active cell, shoudl have recieved the means of its lower cells already from below
//                //trasnfer the mean intensities over
//                std::copy(temp_data[curr_index].begin(),temp_data[curr_index].begin() + 8,local_int.begin());
//                temp_data[curr_index].resize(7);
//                
//                
//                //deal with partially filled levels
//                if (counter_vec[curr_index] == 8){
//                    // Has a full set of children
//                    //first calculate the mean to push up the tree
//                    temp_mean = (1.0/8.0)*(local_int[0] + local_int[1] + local_int[2] + local_int[3] + local_int[4] + local_int[5] + local_int[6] + local_int[7]);
//                } else {
//                    // Does not have a full set of children, therefore we fill in those missing children with the average
//                    //first calculate the mean to push up the tree
//                    temp_mean = (1.0/(counter_vec[curr_index]*1.0))*(local_int[0] + local_int[1] + local_int[2] + local_int[3] + local_int[4] + local_int[5] + local_int[6] + local_int[7]);
//                    
//                    //now loop over the values and set any that are zero to the temp_mean;
//                    for(int j = 0;j < 8;j++){
//                        if(local_int[j] ==0){
//                            local_int[j] = temp_mean;
//                        }
//                    }
//                    
//                }
//                
//                for (int j = 1; j < 8; j++) {
//                    //calculate the wavelet co-efficients (diff, pushing up the mean)
//                    temp_data[curr_index][j-1] = ((local_int[0]*tranform_coeffs[j][0] + local_int[1]*tranform_coeffs[j][1] + local_int[2]*tranform_coeffs[j][2] + local_int[3]*tranform_coeffs[j][3] + local_int[4]*tranform_coeffs[j][4] + local_int[5]*tranform_coeffs[j][5] + local_int[6]*tranform_coeffs[j][6] + local_int[7]*tranform_coeffs[j][7]));
//                }
//                
//                //push the mean up the tree
//                get_parent(p_rep.pl_map.cells[curr_index], parent_cell);
//                
//                auto parent_cell_ref = p_rep.pl_map.pl[parent_cell.k].find(parent_cell);
//                
//                if (parent_cell_ref != p_rep.pl_map.pl[parent_cell.k].end()){
//                    temp_data[parent_cell_ref->second].resize(8);
//                    
//                    curr_position = position_matrix[(curr_cell.y-1)%2][(curr_cell.x-1)%2][(curr_cell.z-1)%2];
//                    
//                    
//                    temp_min = 9999999;
//                    temp_max = -999999;
//                    
//                    //calculate scale
//                    for(int j = 0;j < temp_data[curr_index].size();j++){
//                        temp_min = std::min((float)temp_data[curr_index][j],temp_min);
//                        temp_max = std::max((float)temp_data[curr_index][j],temp_max);
//                    }
//                    
//                    temp_scale = std::max(0.0,floor(log((temp_max-temp_min)/p_rep.pars.comp_scale)/log(2)));
//                    
//                    temp_mean = ceil(temp_mean/pow(2.0,temp_scale));
//                    
//                    
//                    temp_data[parent_cell_ref->second][curr_position] = temp_mean;
//                    counter_vec[parent_cell_ref->second]++;
//                    
//                    //temp_data[parent_cell_ref->second].push_back(temp_mean);
//                } else {
//                    //error something is wrong, this cell should exist, otherwise you need to have run fill tree before runnign this step
//                    std::cout << "Warning:: You have a parent cell that does not exist in the structure, check that you have run fill tree correclty ,or something else is broken" << std::endl;
//                    
//                }
//                
//                
//                
//            } else if (p_rep.status.data[curr_index] == 4 | p_rep.status.data[curr_index] == 5){
//                //in both of these cases, you just push up the intensities
//                
//                //push the mean up the tree
//                get_parent(p_rep.pl_map.cells[curr_index], parent_cell);
//                
//                auto parent_cell_ref = p_rep.pl_map.pl[parent_cell.k].find(parent_cell);
//                
//                if (parent_cell_ref != p_rep.pl_map.pl[parent_cell.k].end()){
//                    temp_data[parent_cell_ref->second].resize(8);
//                    
//                    curr_position = position_matrix[(curr_cell.y-1)%2][(curr_cell.x-1)%2][(curr_cell.z-1)%2];
//                    
//                    temp_data[parent_cell_ref->second][curr_position] = p_rep.Ip.data[p_rep.pl_map.cell_indices[curr_index].first];
//                    
//                    counter_vec[parent_cell_ref->second]++;
//                    
//                    //temp_data[parent_cell_ref->second].push_back(p_rep.Ip.data[p_rep.pl_map.cell_indices[curr_index].first]);
//                } else {
//                    //error something is wrong, this cell should exist, otherwise you need to have run fill tree before runnign this step
//                    std::cout << "Warning:: You have a parent cell that does not exist in the structure, check that you have run fill tree correclty ,or something else is broken" << std::endl;
//                    
//                }
//                
//                
//            }
//            
//            
//            
//        }
//    }
//    
//    
//    for (int i = 0; i < temp_data.size();i++){
//        
//        
//        if(p_rep.status.data[i] == 2 ){
//            //
//            //  Haar Wavelet Coefficients Quantized
//            //
//            
//            if (temp_data[i].size() > 0){
//                temp_min = 9999999;
//                temp_max = -999999;
//                
//                //calculate scale
//                for(int j = 0;j < temp_data[i].size();j++){
//                    temp_min = std::min((float)temp_data[i][j],temp_min);
//                    temp_max = std::max((float)temp_data[i][j],temp_max);
//                }
//                
//                temp_scale = std::max(0.0,floor(log((temp_max-temp_min)/p_rep.pars.comp_scale)/log(2)));
//                
//                scale.data.push_back(temp_scale);
//                
//                for(int j = 0;j < temp_data[i].size();j++){
//                    q.data.push_back(ceil(temp_data[i][j]/pow(2.0,temp_scale)));
//                }
//            }
//        } else if (p_rep.status.data[i] == 7 ){
//            //
//            //  Haar Wavelet Coefficients Quantized (Has come from 4's and 5's.. could quantize this more heavily)
//            //
//            
//            if (temp_data[i].size() > 0){
//                temp_min = 9999999;
//                temp_max = -999999;
//                
//                //calculate scale
//                for(int j = 0;j < temp_data[i].size();j++){
//                    temp_min = std::min((float)temp_data[i][j],temp_min);
//                    temp_max = std::max((float)temp_data[i][j],temp_max);
//                }
//                
//                temp_scale = std::max(0.0,floor(log((temp_max-temp_min)/p_rep.pars.comp_scale)/log(2)));
//                
//                scale.data.push_back(temp_scale);
//                
//                for(int j = 0;j < temp_data[i].size();j++){
//                    q.data.push_back(ceil(temp_data[i][j]/pow(2.0,temp_scale)));
//                }
//            }
//            
//            
//        } else if (p_rep.status.data[i] == 6){
//            //
//            //  Means
//            //
//            
//            //  Quantize this as well, but need some scale information of the wavelet co-efficients (therefore need the scale of the children) so have to do afterwards?)
//            
//            
//            //currently no quantization on these
//            for(int j = 0;j < temp_data[i].size();j++){
//                mu.data.push_back(ceil(temp_data[i][j]));
//            }
//        }
//        
//    }
//    
//    
//    
//}
//template <typename T>
//void calc_compress_decode(Part_rep& p_rep,Part_data<T>& mu,Part_data<int8_t>& q,Part_data<uint8_t>& scale){
//    //
//    //
//    //  Bevan Cheeseman 2016
//    //
//    //  Decodes the intensities of the APR encoded by the one layer Haar wavelet scheme
//    //
//    //
//    
//    
//    //used to determine which quadrant of your upper level you exist in
//    int position_matrix[2][2][2] = {{{0,1},{2,3}},{{4,5},{6,7}}};
//    int curr_position;
//    
//    // 3D Haar Wavelet Transform Coefficients
//    int tranform_coeffs[8][8] = {{1,1,1,1,1,1,1,1},
//        {1,-1,1,-1,1,-1,1,-1},
//        {1,1,-1,-1,1,1,-1,-1},
//        {1,-1,-1,1,1,-1,-1,1},
//        {1,1,1,1,-1,-1,-1,-1},
//        {1,-1,1,-1,-1,1,-1,1},
//        {1,1,-1,-1,-1,-1,1,1},
//        {1,-1,-1,1,-1,1,1,-1}};
//    
//    std::vector<std::vector<float>> temp_data(p_rep.get_cell_num());
//    
//    int count_scale = 0;
//    int count_q = 0;
//    int count_mu = 0;
//    
//    float temp_scale;
//    
//    Cell_id curr_cell;
//    std::vector<Cell_id> child_cells;
//    
//    int part_count = 0;
//    
//    
//    //first need to put the co-efficients back in the cells
//    for(int i = 0;i < p_rep.get_cell_num();i++){
//        
//        if (p_rep.status.data[i] == 2 | p_rep.status.data[i] == 7){
//            
//            temp_scale = pow(2.0,scale.data[count_scale]);
//            
//            count_scale++;
//            
//            curr_cell = p_rep.pl_map.cells[i];
//            
//            //should be 7 co-efficients
//            //place the 7 co-efficients in the 1-7 entries leaving 0 for the mean (just works with the stencil def)
//            
//            temp_data[i].resize(9);
//            
//            for(int j = 0;j < 7;j++){
//                float temp = q.data[count_q]*temp_scale;
//                temp_data[i][j+1]= q.data[count_q]*temp_scale;
//                count_q ++;
//            }
//            
//            
//            temp_data[i][8] = temp_scale;
//            
//            if(p_rep.status.data[i]==2){
//                part_count = part_count + 8;
//            }
//            
//            
//        } else if (p_rep.status.data[i] == 6){
//            // 6 status is the mean coefficients for the wavelet transform
//            
//            curr_cell = p_rep.pl_map.cells[i];
//            
//            //get the mean data and pass it down the tree
//            temp_data[i].resize(8);
//            
//            for(int j = 0;j < 8;j++){
//                temp_data[i][j] = mu.data[count_mu];
//                count_mu++;
//            }
//            
//            //neighbours are produced in the same raster order as the co-efficients are stored
//            get_children(curr_cell,child_cells);
//            
//            //loop over the children check if they exist, then add them
//            for(int j = 0;j < child_cells.size();j++){
//                auto child_cell_ref = p_rep.pl_map.pl[child_cells[j].k].find(child_cells[j]);
//                
//                if (child_cell_ref != p_rep.pl_map.pl[child_cells[j].k].end()){
//                    temp_data[child_cell_ref->second].resize(9);
//                    
//                    temp_data[child_cell_ref->second][0] = temp_data[i][j];
//                    
//                } else {
//                    //this is fine, at higher levels not all children will exist
//                }
//            }
//            //passed on the data now so finished.
//            temp_data[i].resize(0);
//            //it is now reset to being a normal in-active cell
//            p_rep.status.data[i] = 0;
//            
//        } else if (p_rep.status.data[i] == 4 | p_rep.status.data[i] == 5){
//            
//            part_count ++;
//        }
//        
//    }
//    
//    
//    ///////////////////////////////////////////////////////////////////////////////////////////////
//    //
//    //  Now we have loaded in and evaluated all the wavelet co-efficients we just need to construct the intensities
//    //
//    ///////////////////////////////////////////////////////////////////////////////////////////////
//    
//    std::vector<float> local_int;
//    local_int.resize(8);
//    
//    //Resize the part intensities
//    p_rep.Ip.data.resize(part_count,0);
//    
//    int curr_index;
//    
//    for (int k_ = (p_rep.pl_map.k_min-1); k_ <= p_rep.pl_map.k_max; k_++) {
//        for(auto cell_ref : p_rep.pl_map.pl[k_]) {
//            
//            curr_cell = p_rep.pl_map.cells[cell_ref.second];
//            
//            curr_index = cell_ref.second;
//            
//            if (p_rep.status.data[curr_index] == 2 ) {
//                //since we are going down the tree, the intiial co-efficient [0] shoudl have been pushed downwards already
//                
//                //adjust the mean value from quantization
//                temp_data[curr_index][0] = temp_data[curr_index][0]*temp_data[curr_index][8];
//                
//                temp_data[curr_index].resize(8);
//                
//                std::copy(temp_data[curr_index].begin(),temp_data[curr_index].end(),local_int.begin());
//                
//                //transform the co_efficientss to the intensities
//                for (int j = 0; j < 8; j++) {
//                    //calculate the wavelet co-efficients (diff, pushing up the mean)
//                    temp_data[curr_index][j] = local_int[0]*tranform_coeffs[j][0] + (1.0/8.0)*((local_int[1]*tranform_coeffs[j][1] + local_int[2]*tranform_coeffs[j][2] + local_int[3]*tranform_coeffs[j][3] + local_int[4]*tranform_coeffs[j][4] + local_int[5]*tranform_coeffs[j][5] + local_int[6]*tranform_coeffs[j][6] + local_int[7]*tranform_coeffs[j][7]));
//                }
//                
//                local_int = temp_data[curr_index];
//                
//                //now update the particle intensities
//                for(int j = 0;j < 8; j++) {
//                    p_rep.Ip.data[p_rep.pl_map.cell_indices[curr_index].first +j] = temp_data[curr_index][j];
//                }
//                
//                temp_data[curr_index].resize(0);
//                
//                
//            } else if (p_rep.status.data[curr_index] ==  7){
//                //
//                //
//                //  Have to reconstruct the solution and then pass down the estimated intensities to the 4 and 5 cells
//                //
//                //
//                //adjust the mean value from quantization
//                temp_data[curr_index][0] = temp_data[curr_index][0]*temp_data[curr_index][8];
//                
//                temp_data[curr_index].resize(8);
//                //  reset back to normal status
//                p_rep.status.data[curr_index] = 0;
//                
//                std::copy(temp_data[curr_index].begin(),temp_data[curr_index].end(),local_int.begin());
//                
//                //transform the co_efficientss
//                for (int j = 0; j < 8; j++) {
//                    //calculate the wavelet co-efficients (diff, pushing up the mean)
//                    temp_data[curr_index][j] = local_int[0]*tranform_coeffs[j][0] + (1.0/8.0)*((local_int[1]*tranform_coeffs[j][1] + local_int[2]*tranform_coeffs[j][2] + local_int[3]*tranform_coeffs[j][3] + local_int[4]*tranform_coeffs[j][4] + local_int[5]*tranform_coeffs[j][5] + local_int[6]*tranform_coeffs[j][6] + local_int[7]*tranform_coeffs[j][7]));
//                }
//                
//                ///////////////////////////////////////////////
//                //
//                //  Now we need to push the intensities to the 4 and 5 cells
//                //
//                /////////////////////////////////////////////
//                
//                //push to your children
//                //neighbours are produced in the same raster order as the co-efficients are stored
//                get_children(curr_cell,child_cells);
//                
//                //loop over the children check if they exist, then add them
//                for(int i = 0;i < child_cells.size();i++){
//                    auto child_cell_ref = p_rep.pl_map.pl[child_cells[i].k].find(child_cells[i]);
//                    
//                    if (child_cell_ref != p_rep.pl_map.pl[child_cells[i].k].end()){
//                        //push the calculated means down the tree to the starting position of the array
//                        p_rep.Ip.data[p_rep.pl_map.cell_indices[child_cell_ref->second].first] = temp_data[curr_index][i];
//                        
//                        //temp_data[child_cell_ref->second][0] = temp_data[curr_index][i];
//                    } else {
//                        //this is fine, at higher levels not all children will exist
//                    }
//                    
//                    
//                }
//                //don't need them anymore they have been pushed down
//                temp_data[curr_index].resize(0);
//                
//            }
//            
//        }
//    }
//    
//    
//    
//}

#endif