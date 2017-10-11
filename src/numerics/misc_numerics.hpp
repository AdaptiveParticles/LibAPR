/////////////////////////////////
//
//
//  Contains Miscellaneous APR numerics codes and output
//
//
//////////////////////////////////
#ifndef MISC_NUM
#define MISC_NUM

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "filter_help/FilterLevel.hpp"

template<typename U,typename V>
void interp_img(Mesh_data<U>& img,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new,ExtraPartCellData<V>& particles_int,const bool val = false);



void create_y_data(ExtraPartCellData<uint16_t>& y_vec,ParticleDataNew<float, uint64_t>& part_new,PartCellData<uint64_t>& pc_data);

template<typename U,typename V>
void interp_slice(Mesh_data<U>& slice,std::vector<std::vector<std::vector<uint16_t>>>& y_vec,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new,int dir,int num);

template<typename U,typename V>
void interp_slice(Mesh_data<U>& slice,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new,ExtraPartCellData<V>& particles_int,int dir,int num);

void create_y_data(std::vector<std::vector<std::vector<uint16_t>>>& y_vec,PartCellStructure<float,uint64_t>& pc_struct);

template<typename V>
void filter_slice(std::vector<V>& filter,std::vector<V>& filter_d,ExtraPartCellData<V>& filter_output,Mesh_data<V>& slice,ExtraPartCellData<uint16_t>& y_vec,const int dir,const int num);

template<typename U>
std::vector<U> create_gauss_filter(float t,int size){
    //
    //  Bevan Cheeseman 2017
    //
    //  Guassian Filter
    //
    //


    std::vector<U> filter;

    filter.resize(size*2 + 1,0);

    float del_x = 1;

    float pi = 3.14;

    float start_x = -size*del_x;

    float x = start_x;

    float factor1 = 1/pow(2*pi*pow(t,2),.5);

    for (int i = 0; i < filter.size(); ++i) {
        filter[i] = factor1*exp(-pow(x,2)/(2*pow(t,2)));
        x += del_x;
    }

    float sum = 0;
    for (int i = 0; i < filter.size(); ++i) {
        sum += fabs(filter[i]);
    }

    for (int i = 0; i < filter.size(); ++i) {
        filter[i] = filter[i]/sum;
    }

    return filter;

}


template<typename U>
void create_LOG_filter(int size,float s,std::vector<U>& filter_f,std::vector<U>& filter_b){
    //
    //  Bevan Cheeseman 2017
    //
    //  Laplacian of Guassians Filters
    //
    //  http://bigwww.epfl.ch/publications/sage0501.pdf
    //

    filter_f.resize(size*2 + 1,0);
    filter_b.resize(size*2 + 1,0);

    float del_x = 1;

    float pi = 3.14;

    float start_x = -size*del_x;

    float x = start_x;

    float C = 1/(pow(2*pi,1.5)*pow(s,3));

    float factor2 = C*((pow(x,2)/pow(s,4)) - (1.0/pow(s,2)));

    for (int i = 0; i < filter_f.size(); ++i) {
        filter_f[i] = C*((pow(x,2)/pow(s,4)) - (1.0/pow(s,2)))*exp(-pow(x,2)/(2*pow(s,2))) ;
        filter_b[i] = exp(-pow(x,2)/(2*pow(s,2)));
        x += del_x;
    }

}



template<typename U>
std::vector<U> create_dog_filter(int size,float t,float K){
    //
    //  Bevan Cheeseman 2017
    //
    //  Difference of Gaussians Filter
    //
    //  https://nenadmarkus.com/posts/3D-DoG/
    //

    std::vector<U> filter;

    filter.resize(size*2 + 1,0);

    float del_x = 1;

    float pi = 3.14;

    float start_x = -size*del_x;

    float x = start_x;

    float factor1 = 1/pow(2*pi*pow(t,2),.5);
    float factor2 = 1/pow(2*pi*pow(K*t,2),.5);

    for (int i = 0; i < filter.size(); ++i) {
        filter[i] = factor1*exp(-pow(x,2)/(2*pow(t,2))) - factor2*exp(-pow(x,2)/(2*pow(K*t,2)));
        x += del_x;
    }

    float sum = 0;
    for (int i = 0; i < filter.size(); ++i) {
        sum += fabs(filter[i]);
    }

    for (int i = 0; i < filter.size(); ++i) {
        filter[i] = filter[i]/sum;
    }

    return filter;

}

template<typename S>
void interp_depth_to_mesh(Mesh_data<uint8_t>& k_img,PartCellStructure<S,uint64_t>& pc_struct,int bound_flag = 1){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes a pc_struct and interpolates the depth to the mesh
    //
    //
    
    //initialize dataset to interp to
    ExtraPartCellData<uint8_t> k_parts(pc_struct.part_data.particle_data);
    
    
    
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
                        
                        uint8_t depth = i + (status == SEED);


                        depth = i + (status < FILLER)*bound_flag;



                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            k_parts.get_part(curr_key) = depth;
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    
    //now interpolate this to the mesh
    pc_struct.interp_parts_to_pc(k_img,k_parts);
    
    
}


template<typename S>
void interp_status_to_mesh(Mesh_data<uint8_t>& status_img,PartCellStructure<S,uint64_t>& pc_struct){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes a pc_struct and interpolates the depth to the mesh
    //
    //
    
    //initialize dataset to interp to
    ExtraPartCellData<uint8_t> status_parts(pc_struct.part_data.particle_data);
    
    
    
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
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            status_parts.get_part(curr_key) = status;
                            
                        }
                        
                    } else {
                        // Inidicates this is not a particle cell node, and is a gap node
                    }
                    
                }
                
            }
            
        }
    }
    
    
    //now interpolate this to the mesh
    pc_struct.interp_parts_to_pc(status_img,status_parts);
    
    
}

template<typename S,typename T>
void interp_extrapc_to_mesh(Mesh_data<T>& output_img,PartCellStructure<S,uint64_t>& pc_struct,ExtraPartCellData<T>& cell_data){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes a pc_struct and interpolates the depth to the mesh
    //
    //
    
    //initialize dataset to interp to
    ExtraPartCellData<T> extra_parts(pc_struct.part_data.particle_data);

    
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
                        
                        float val = cell_data.get_val(curr_key);
                        
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering
                            extra_parts.get_part(curr_key) = val;
                            
                            
                        }
                    }
                    
                }
                
            }
            
        }
    }
    
    
    //now interpolate this to the mesh
    pc_struct.interp_parts_to_pc(output_img,extra_parts);
    
    
}


template<typename U,typename V>
void interp_parts_to_smooth(Mesh_data<U>& out_image,ExtraPartCellData<V>& interp_data,PartCellStructure<float,uint64_t>& pc_struct,std::vector<float> scale_d){


    Mesh_data<U> pc_image;
    Mesh_data<uint8_t> k_img;

    pc_struct.interp_parts_to_pc(pc_image,interp_data);

    interp_depth_to_mesh(k_img,pc_struct,0);

    int filter_offset = 0;

    unsigned int x_num = pc_image.x_num;
    unsigned int y_num = pc_image.y_num;
    unsigned int z_num = pc_image.z_num;

    Mesh_data<U> output_data;
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);

    uint64_t offset_min;
    uint64_t offset_max;

    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    float factor = .1;

    int k_max = pc_struct.depth_max + 1;


//#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    filter_offset = floor(pow(2,k_max - k_img.mesh[j*x_num*y_num + i*y_num + k])/scale_d[0]);

                    offset_max = std::min((int)(k + filter_offset),(int)(y_num-1));
                    offset_min = std::max((int)(k - filter_offset),(int)0);

                    factor = 1.0/(offset_max - offset_min+1);

                    uint64_t f = 0;
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + i*y_num + c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += pc_image.mesh[j*x_num*y_num + i*y_num + c]*factor;
                        f++;
                    }

                }
            }
        }




    std::swap(output_data.mesh,pc_image.mesh);



//#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){

                for(k = 0;k < y_num;k++){

                    filter_offset = floor(pow(2,k_max - k_img.mesh[j*x_num*y_num + i*y_num + k])/scale_d[1]);

                    offset_max = std::min((int)(i + filter_offset),(int)(x_num-1));
                    offset_min = std::max((int)(i - filter_offset),(int)0);

                    factor = 1.0/(offset_max - offset_min+1);

                    uint64_t f = 0;
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += pc_image.mesh[j*x_num*y_num + c*y_num + k]*factor;
                        f++;
                    }

                }
            }
        }

//
//
//    // z loop
//
    std::swap(output_data.mesh,pc_image.mesh);

//#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){


                for(k = 0;k < y_num;k++){

                    filter_offset = floor(pow(2,k_max - k_img.mesh[j*x_num*y_num + i*y_num + k])/scale_d[2]);

                    offset_max = std::min((int)(j + filter_offset),(int)(z_num-1));
                    offset_min = std::max((int)(j - filter_offset),(int)0);

                    factor = 1.0/(offset_max - offset_min+1);

                    uint64_t f = 0;
                    output_data.mesh[j*x_num*y_num + i*y_num + k]=0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){

                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += pc_image.mesh[c*x_num*y_num + i*y_num + k]*factor;
                        f++;
                    }

                }
            }
        }



    out_image = output_data;

}
template<typename T>
void threshold_part(PartCellStructure<float,uint64_t>& pc_struct,ExtraPartCellData<T>& th_output,float threshold){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //
    //
    
    uint64_t x_;
    uint64_t z_;
    uint64_t j_;
    
    th_output.initialize_structure_parts(pc_struct.part_data.particle_data);
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    timer.start_timer("Threshold Loop");
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
        for(uint64_t depth = (pc_struct.depth_max); depth >= (pc_struct.depth_min); depth--){
            
            const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
            
            //initialize layer iterators
            FilterLevel<uint64_t,float> curr_level;
            curr_level.set_new_depth(depth,pc_struct);
            
#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                
                for(x_ = 0;x_ < x_num_;x_++){
                    curr_level.set_new_xz(x_,z_,pc_struct);
                    //the y direction loop however is sparse, and must be accessed accordinagly
                    for(j_ = 0;j_ < curr_level.j_num_();j_++){
                        
                        //particle cell node value, used here as it is requried for getting the particle neighbours
                        bool iscell = curr_level.new_j(j_,pc_struct);
                        
                        if (iscell){
                            curr_level.update_cell(pc_struct);
                            
                            for(int p = 0;p < pc_struct.part_data.get_num_parts(curr_level.status);p++){
                                th_output.data[depth][curr_level.pc_offset][curr_level.part_offset+p] = pc_struct.part_data.particle_data.data[depth][curr_level.pc_offset][curr_level.part_offset+p] > threshold ;
                            }
                            
                        } else {
                            curr_level.update_gap(pc_struct);
                        }
                    }
                    
                }
            }
            
            
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Particle Threshold Size: " << pc_struct.get_number_parts() << " took: " << time << std::endl;
    
    
    
    
}

template<typename U>
void threshold_pixels(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    float threshold = 50;
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 6;
    filter.resize(filter_offset*2 +1,1);
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                
                for(k = 0;k < y_num;k++){
                        
                    output_data.mesh[j*x_num*y_num + i*y_num + k] = input_data.mesh[j*x_num*y_num + i*y_num + k] < threshold;
                    
                }
            }
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Pixel Threshold Size: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}
template<typename S>
void get_coord(const int dir,const CurrentLevel<S, uint64_t> &curr_level,const float step_size,int &dim1,int &dim2){
    //
    //  Bevan Cheeseman 2017
    //

    //calculate real coordinate




    if(dir != 1) {
        //yz case
            //y//z
        dim1 = curr_level.y * step_size;
        dim2 = curr_level.z * step_size;

    } else {
            //yx

        dim1 = curr_level.y * step_size;
        dim2 = curr_level.x * step_size;


    }

}

void get_coord(const int& dir,const int& y,const int& x,const int& z,const float& step_size,int &dim1,int &dim2){
    //
    //  Bevan Cheeseman 2017
    //

    //calculate real coordinate


    if(dir == 0){
        //yz
        dim1 = y * step_size;
        dim2 = z * step_size;
    } else if (dir == 1){
        //xy
        dim1 = x * step_size;
        dim2 = y * step_size;
    } else {
        //zy
        dim1 = z * step_size;
        dim2 = y * step_size;
    }

}
void get_coord_filter(const int& dir,const int& y,const int& x,const int& z,const float& step_size,int &dim1,int &dim2){
    //
    //  Bevan Cheeseman 2017
    //

    //calculate real coordinate


    if(dir == 0){
        //yz
        dim1 = (1.0*y + 0.5) * step_size;
        dim2 = (1.0*z + 0.5)  * step_size;
    } else if (dir == 1){
        //xy
        dim1 = (1.0*x + 0.5) * step_size;
        dim2 = (1.0*y + 0.5) * step_size;
    } else {
        //zy
        dim1 = (1.0*z + 0.5)  * step_size;
        dim2 = (1.0*y + 0.5)  * step_size;
    }

}




template<typename U,typename V>
void interp_slice(PartCellStructure<float,uint64_t>& pc_struct,ExtraPartCellData<V>& interp_data,int dir,int num){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //



    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    //Genearate particle at cell locations, easier access
    ExtraPartCellData<float> particles_int;
    part_new.create_particles_at_cell_structure(particles_int);

    //iterator
    CurrentLevel<float,uint64_t> curr_level(pc_data);


    Mesh_data<U> slice;

    std::vector<unsigned int> x_num_min;
    std::vector<unsigned int> x_num;

    std::vector<unsigned int> z_num_min;
    std::vector<unsigned int> z_num;

    x_num.resize(pc_data.depth_max + 1);
    z_num.resize(pc_data.depth_max + 1);
    x_num_min.resize(pc_data.depth_max + 1);
    z_num_min.resize(pc_data.depth_max + 1);

    int x_dim = ceil(pc_struct.org_dims[0]/2.0)*2;
    int z_dim = ceil(pc_struct.org_dims[1]/2.0)*2;
    int y_dim = ceil(pc_struct.org_dims[2]/2.0)*2;


    if(dir != 1) {
        //yz case
        z_num = pc_data.z_num;

        for (int i = pc_data.depth_min; i <= pc_data.depth_max ; ++i) {
            x_num[i] = num/pow(2,pc_data.depth_max - i) + 1;
            z_num_min[i] = 0;
            x_num_min[i] = num/pow(2,pc_data.depth_max - i);
        }

        slice.initialize(pc_struct.org_dims[0],pc_struct.org_dims[2],1,0);


    } else {
        //yx case
        x_num = pc_data.x_num;

        for (int i = pc_data.depth_min; i <= pc_data.depth_max ; ++i) {
            z_num[i] = num/pow(2,pc_data.depth_max - i) + 1;
            x_num_min[i] = 0;
            z_num_min[i] = num/pow(2,pc_data.depth_max - i);
        }

        slice.initialize(pc_struct.org_dims[0],pc_struct.org_dims[1],1,0);

    }

    Part_timer timer;

    timer.verbose_flag = true;

    timer.start_timer("interp slice");


    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = x_num[depth];
        const unsigned int z_num_ = z_num[depth];

        const unsigned int x_num_min_ = x_num_min[depth];
        const unsigned int z_num_min_ = z_num_min[depth];

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        float temp_int =  curr_level.get_val(particles_int);

                        int dim1 = 0;
                        int dim2 = 0;



                        get_coord(dir,curr_level,step_size,dim1,dim2);

                        //add to all the required rays

                        for (int k = 0; k < step_size; ++k) {
#pragma omp simd
                            for (int i = 0; i < step_size; ++i) {
                                //slice.mesh[dim1 + i + (dim2 + k)*slice.y_num] = temp_int;

                                slice(dim1 + i,(dim2 + k),1) = temp_int;
                            }
                        }


                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }

    timer.stop_timer();

    debug_write(slice,"slice");


}
template<typename U>
void get_slices(PartCellStructure<float,uint64_t>& pc_struct){

    ParticleDataNew<float, uint64_t> part_new;
    //flattens format to particle = cell, this is in the classic access/part paradigm
    part_new.initialize_from_structure(pc_struct);

    //generates the nieghbour structure
    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    //Genearate particle at cell locations, easier access
    ExtraPartCellData<float> particles_int;
    part_new.create_particles_at_cell_structure(particles_int);

    pc_data.org_dims = pc_struct.org_dims;
    part_new.access_data.org_dims = pc_struct.org_dims;
    part_new.particle_data.org_dims = pc_struct.org_dims;

    int dir = 0;
    int num = 800;

    int num_slices = 0;

    if(dir != 1){
        num_slices = pc_struct.org_dims[1];
    } else {
        num_slices = pc_struct.org_dims[2];
    }

    Mesh_data<U> slice;

    Part_timer timer;
    timer.verbose_flag = true;

    timer.start_timer("interp");

    for(int dir = 0; dir < 3;++dir) {

        if (dir != 1) {
            num_slices = pc_struct.org_dims[1];
        } else {
            num_slices = pc_struct.org_dims[2];
        }

        for (int i = 0; i < num_slices; ++i) {
            interp_slice(slice, pc_data, part_new, particles_int, dir, i);
        }
    }

    timer.stop_timer();

    interp_slice(slice,pc_data,part_new,particles_int,dir,150);

    debug_write(slice,"slice2");


    timer.start_timer("set up y");

    ExtraPartCellData<uint16_t> y_vec;

    create_y_data(y_vec,part_new,pc_data);


    timer.stop_timer();

    timer.start_timer("interp 2");

    for(int dir = 0; dir < 3;++dir) {

        if (dir != 1) {
            num_slices = pc_struct.org_dims[1];
        } else {
            num_slices = pc_struct.org_dims[2];
        }

        for (int i = 0; i < num_slices; ++i) {
            interp_slice_opt(slice, y_vec, part_new.particle_data, dir, i);
        }

    }

    timer.stop_timer();


    timer.start_timer("interp 2 old");

    for(int dir = 0; dir < 3;++dir) {

        if (dir != 1) {
            num_slices = pc_struct.org_dims[1];
        } else {
            num_slices = pc_struct.org_dims[2];
        }
        int i = 0;

#pragma omp parallel for default(shared) private(i) firstprivate(slice)
        for (i = 0; i < num_slices; ++i) {
            interp_slice(slice, y_vec, part_new.particle_data, dir, i);
        }

    }

    timer.stop_timer();

    interp_slice(slice, y_vec, part_new.particle_data, 0, 500);
    debug_write(slice,"slice3");

};



template<typename U,typename V>
void interp_img(Mesh_data<U>& img,ExtraPartCellData<uint16_t>& y_vec,ExtraPartCellData<V>& particles_int){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //

    img.initialize(y_vec.org_dims[0],y_vec.org_dims[1],y_vec.org_dims[2],0);

    int z_,x_,j_,y_;

    for(uint64_t depth = (y_vec.depth_min);depth <= y_vec.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = y_vec.x_num[depth];
        const unsigned int z_num_ = y_vec.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        const float step_size = pow(2,y_vec.depth_max - depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_ * z_ + x_;

                for (j_ = 0; j_ <y_vec.data[depth][pc_offset].size(); j_++) {

                        int dim1 = y_vec.data[depth][pc_offset][j_] * step_size;
                        int dim2 = x_ * step_size;
                        int dim3 = z_ * step_size;

                        const V temp_int = particles_int.data[depth][pc_offset][j_];
                        //add to all the required rays

                        const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                        const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                        const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                        for (int q = dim3; q < offset_max_dim3; ++q) {

                            for (int k = dim2; k < offset_max_dim2; ++k) {
#pragma omp simd
                                for (int i = dim1; i < offset_max_dim1; ++i) {
                                    img.mesh[i + (k) * img.y_num + q*img.y_num*img.x_num] = temp_int;
                                }
                            }
                        }


                }
            }
        }
    }




}

template<typename U>
void interp_depth(Mesh_data<U>& img,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //

    img.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],0);

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[depth];
        const unsigned int z_num_ = pc_data.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        int dim1 = curr_level.y * step_size;
                        int dim2 = curr_level.x * step_size;
                        int dim3 = curr_level.z * step_size;

                        const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                        const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                        const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                        for (int q = dim3; q < offset_max_dim3; ++q) {

                            for (int k = dim2; k < offset_max_dim2; ++k) {
#pragma omp simd
                                for (int i = dim1; i < offset_max_dim1; ++i) {
                                    img.mesh[i + (k) * img.y_num + q*img.y_num*img.x_num] = depth;
                                }
                            }
                        }


                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }




}


template<typename U,typename V>
void interp_img(Mesh_data<U>& img,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new,ExtraPartCellData<V>& particles_int,const bool val){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //

    img.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],0);

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[depth];
        const unsigned int z_num_ = pc_data.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        int dim1 = curr_level.y * step_size;
                        int dim2 = curr_level.x * step_size;
                        int dim3 = curr_level.z * step_size;

                        float temp_int;
                        //add to all the required rays
                        if(val){
                            temp_int = curr_level.get_val(particles_int);
                        } else {
                            temp_int = curr_level.get_part(particles_int);
                        }
                        const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                        const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                        const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                        for (int q = dim3; q < offset_max_dim3; ++q) {

                            for (int k = dim2; k < offset_max_dim2; ++k) {
    #pragma omp simd
                                for (int i = dim1; i < offset_max_dim1; ++i) {
                                    img.mesh[i + (k) * img.y_num + q*img.y_num*img.x_num] = temp_int;
                                }
                            }
                        }


                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }




}

template<typename U>
U weight_func(const float d,const float sd){

    //return exp(-pow(d,2)/pow(sd,2))*(d <= sd);
    return exp(-pow(d,2)/pow(sd,1));
    //return (sd*(1.05)-d)/sd;
    //return 1.0f;
}

float square_dist(float x,float y,float a,float b){
    //
    //  Bevan Cheeseman 2017
    //
    //  Calculates the distance of a combination of x,y to a rectangle of sides a.b
    //

    float theta = std::atan2(x,y);
    float d;

    if (std::abs(std::tan(theta)) > a/b) {
        d = a / std::abs(std::sin(theta));
    } else {
        d = b / std::abs(std::cos(theta));
    }

    return d;
}

float cube_dist(float x,float y,float z,float a){
    //
    //  Bevan Cheeseman 2017
    //
    //  Calculates the distance of a combination of x,y,z to a cube with sides a
    //

    float d_p = square_dist(x,y,a,a);

    float dist_2D = sqrt(pow(x,2) + pow(y,2));

    float d = square_dist(dist_2D,z,d_p,a);

    return d;
}

float integral_weight_func(const float x,const float y, const float z,const float r1,const float r2){

    float dist = sqrt(pow( x,2 ) + pow( y,2 ) + pow( z,2 ));

    float dist_c = cube_dist(x,y,z,r1);

    float int_dist;

    if(dist > dist_c){
        int_dist = (dist-dist_c)*(1.0f/r2) + dist_c*(1.0f/r1);

    } else {
        int_dist = dist*(1.0f/r1);
    }

    if (int_dist <=1){
        return weight_func<float>(int_dist,r2/r1);

    } else {

        return 0;
    }


}
bool integral_check_neigh(const float x,const float y, const float z,const float r1,const float r2){

    float dist = sqrt(pow( x,2.0 ) + pow( y,2.0 ) + pow( z,2.0 ));

    float dist_c = cube_dist(x,y,z,r1);

    float int_dist;

    if(dist > dist_c){
        int_dist = (dist-dist_c)*(1.0f/r2) + dist_c*(1.0f/r1);

    } else {
        int_dist = dist*(1.0f/r1);
    }

    if (int_dist <=1){
        return true;

    } else {

        return false;
    }


}




template<typename U,typename V>
void weigted_interp_img(Mesh_data<U>& img,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new,ExtraPartCellData<V>& particles_int,const bool val,bool smooth = true){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //

    img.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],0.0);

    Mesh_data<double> weight_img;
    weight_img.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],0.0);

    Mesh_data<double> weight_int;
    weight_int.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],0.0);



    Mesh_data<float> d_img;

    interp_depth(d_img, pc_data, part_new);

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[depth];
        const unsigned int z_num_ = pc_data.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        int dim1,dim2,dim3;

                        int offset_max_dim1,offset_max_dim2,offset_max_dim3;

                        if (curr_level.status < 0){
                            dim1 = std::max(((float) curr_level.y ) * step_size, 0.0f);
                            dim2 = std::max(((float) curr_level.x ) * step_size, 0.0f);
                            dim3 = std::max(((float) curr_level.z ) * step_size, 0.0f);

                            offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                            offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                            offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                        } else {

                            dim1 = std::max(((float) curr_level.y - 1.00f) * step_size, 0.0f);
                            dim2 = std::max(((float) curr_level.x - 1.00f) * step_size, 0.0f);
                            dim3 = std::max(((float) curr_level.z - 1.00f) * step_size, 0.0f);

                            offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + 3*step_size));
                            offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + 3*step_size));
                            offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + 3*step_size));

                        }

                        float temp_int;
                        //add to all the required rays
                        if(val){
                            temp_int = curr_level.get_val(particles_int);
                        } else {
                            temp_int = curr_level.get_part(particles_int);
                        }

                        float mid_1 = ((float)curr_level.y) * step_size + (step_size-1)/2.0f;
                        float mid_2 = ((float)curr_level.x) * step_size + (step_size-1)/2.0f;
                        float mid_3 = ((float)curr_level.z) * step_size + (step_size-1)/2.0f;

                        if(step_size ==1){
                            mid_1 = curr_level.y;
                            mid_2 = curr_level.x;
                            mid_3 = curr_level.z;
                        }

                        for (int q = dim3; q < offset_max_dim3; ++q) {

                            for (int k = dim2; k < offset_max_dim2; ++k) {

                                for (int i = dim1; i < offset_max_dim1; ++i) {



                                    float neigh_size = pow(2,curr_level.depth_max - d_img.mesh[i + (k) * img.y_num + q*img.y_num*img.x_num]);

                                    double w =  integral_weight_func((i-mid_1),(k-mid_2), (q-mid_3),step_size,neigh_size);

                                    double temp = w;

//                                    if(w > 0.001) {
//
//
//                                        weight_int.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] +=
//                                                ((double) temp_int) * temp;
//                                        weight_img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] += temp;
//
//
//
//                                    }

                                    if(w > 0) {

                                        weight_img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] += w;

                                        temp = w / weight_img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num];


                                        weight_int.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] =
                                                ((double) temp_int) * temp + (1.0f - temp) *
                                                                             weight_int.mesh[i + (k) * img.y_num +
                                                                                             q * img.y_num * img.x_num];
                                    }


                                }
                            }
                        }


                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }


    if(smooth) {
        uint64_t depth = part_new.access_data.depth_max;
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[depth];
        const unsigned int z_num_ = pc_data.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2, curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        int dim1 = std::max(((float) curr_level.y) * step_size, 0.0f);
                        int dim2 = std::max(((float) curr_level.x) * step_size, 0.0f);
                        int dim3 = std::max(((float) curr_level.z) * step_size, 0.0f);

                        float temp_int;
                        //add to all the required rays
                        if (val) {
                            temp_int = curr_level.get_val(particles_int);
                        } else {
                            temp_int = curr_level.get_part(particles_int);
                        }
                        const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + 1 * step_size));
                        const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + 1 * step_size));
                        const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + 1 * step_size));

                        const float mid = 3 * step_size / 2;

                        for (int q = dim3; q < offset_max_dim3; ++q) {

                            for (int k = dim2; k < offset_max_dim2; ++k) {

                                for (int i = dim1; i < offset_max_dim1; ++i) {

                                    //float dist = sqrt(pow( (q-dim3-mid),2 ) + pow( (k-dim2-mid),2 ) + pow( (i-dim1-mid),2 ));
                                    //float w =  weight_func<float>(dist,step_size*2);

                                    weight_int.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] = (temp_int) * 1;
                                    weight_img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] = 1;

                                }
                            }
                        }


                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }







    //then loop over and divide the two

    for(z_ = 0;z_ < img.mesh.size();z_++){

        img.mesh[z_] = round(weight_int.mesh[z_]);
        //weight_img.mesh[z_] = 1000*weight_img.mesh[z_];

    }



    //debug_write(weight_img,"weight_img");
   // debug_write(weight_int,"weight_int");





}
template<typename U,typename V>
void min_max_interp(Mesh_data<U>& min_img,Mesh_data<U>& max_img,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new,ExtraPartCellData<V>& particles_int,const bool val){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //

    min_img.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],660000);
    max_img.initialize(pc_data.org_dims[0],pc_data.org_dims[1],pc_data.org_dims[2],0);

    Mesh_data<uint8_t> d_img;

    interp_depth(d_img, pc_data, part_new);

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth < part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = pc_data.x_num[depth];
        const unsigned int z_num_ = pc_data.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const double step_size = pow(2,curr_level.depth_max - curr_level.depth);

//#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        int dim1,dim2,dim3;

                        int offset_max_dim1,offset_max_dim2,offset_max_dim3;

                        dim1 = round(std::max(((double) curr_level.y - 1.00f) * step_size, 0.0));
                        dim2 = round(std::max(((double) curr_level.x - 1.00f) * step_size, 0.0));
                        dim3 = round(std::max(((double) curr_level.z - 1.00f) * step_size, 0.0));

                        offset_max_dim1 = std::min((int) min_img.y_num, (int) (dim1 + 3*step_size));
                        offset_max_dim2 = std::min((int) min_img.x_num, (int) (dim2 + 3*step_size));
                        offset_max_dim3 = std::min((int) min_img.z_num, (int) (dim3 + 3*step_size));

                        float temp_int;
                        //add to all the required rays
                        if(val){
                            temp_int = curr_level.get_val(particles_int);
                        } else {
                            temp_int = curr_level.get_part(particles_int);
                        }

                        double mid_1 = ((float)curr_level.y) * step_size + (step_size-1)/2.0f;
                        double mid_2 = ((float)curr_level.x) * step_size + (step_size-1)/2.0f;
                        double mid_3 = ((float)curr_level.z) * step_size + (step_size-1)/2.0f;

                        if(step_size ==1){
                            mid_1 = curr_level.y;
                            mid_2 = curr_level.x;
                            mid_3 = curr_level.z;
                        }


                        for (int q = dim3; q < offset_max_dim3; ++q) {

                            for (int k = dim2; k < offset_max_dim2; ++k) {

                                for (int i = dim1; i < offset_max_dim1; ++i) {

                                    double neigh_size = pow(2,curr_level.depth_max - d_img.mesh[i + (k) * min_img.y_num + q*min_img.y_num*min_img.x_num]);

                                  //  max_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num] = std::max(temp_int,max_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num]);
                                 //   min_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num] = std::min(temp_int,min_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num]);
                                    double dist = sqrt(pow( (i-mid_1),2.0 ) + pow( (k-mid_2),2.0 ) + pow( (q-mid_3),2.0 ));


                                    if ( dist <= step_size) {

                                        if ( integral_check_neigh((i-mid_1),(k-mid_2), (q-mid_3),step_size,neigh_size)) {

                                            max_img.mesh[i + (k) * min_img.y_num +
                                                         q * min_img.y_num * min_img.x_num] = std::max(temp_int,
                                                                                                       max_img.mesh[i +
                                                                                                                    (k) *
                                                                                                                    min_img.y_num +
                                                                                                                    q *
                                                                                                                    min_img.y_num *
                                                                                                                    min_img.x_num]);
                                            min_img.mesh[i + (k) * min_img.y_num +
                                                         q * min_img.y_num * min_img.x_num] = std::min(temp_int,
                                                                                                       min_img.mesh[i +
                                                                                                                    (k) *
                                                                                                                    min_img.y_num +
                                                                                                                    q *
                                                                                                                    min_img.y_num *
                                                                                                                    min_img.x_num]);
                                        }
                                    }



//                                    if ( integral_check_neigh(round(i-mid_1),round(k-mid_2), round(q-mid_3),step_size,step_size)) {
//                                        max_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num] = std::max(temp_int,max_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num]);
//                                        min_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num] = std::min(temp_int,min_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num]);
//                                    }

                                }
                            }
                        }



                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }



    uint64_t depth = part_new.access_data.depth_max;
    //loop over the resolutions of the structure
    const unsigned int x_num_ = pc_data.x_num[depth];
    const unsigned int z_num_ = pc_data.z_num[depth];

    const unsigned int x_num_min_ = 0;
    const unsigned int z_num_min_ = 0;

    CurrentLevel<float, uint64_t> curr_level(pc_data);
    curr_level.set_new_depth(depth, part_new);

    const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

//#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
    for (z_ = z_num_min_; z_ < z_num_; z_++) {
        //both z and x are explicitly accessed in the structure

        for (x_ = x_num_min_; x_ < x_num_; x_++) {

            curr_level.set_new_xz(x_, z_, part_new);

            for (j_ = 0; j_ < curr_level.j_num; j_++) {

                bool iscell = curr_level.new_j(j_, part_new);

                if (iscell) {
                    //Indicates this is a particle cell node
                    curr_level.update_cell(part_new);

                    int dim1 = std::max(((float)curr_level.y) * step_size,0.0f);
                    int dim2 = std::max(((float)curr_level.x) * step_size,0.0f);
                    int dim3 = std::max(((float)curr_level.z) * step_size,0.0f);

                    float temp_int;
                    //add to all the required rays
                    if(val){
                        temp_int = curr_level.get_val(particles_int);
                    } else {
                        temp_int = curr_level.get_part(particles_int);
                    }
                    const int offset_max_dim1 = std::min((int) min_img.y_num, (int) (dim1 + 1*step_size));
                    const int offset_max_dim2 = std::min((int) min_img.x_num, (int) (dim2 + 1*step_size));
                    const int offset_max_dim3 = std::min((int) min_img.z_num, (int) (dim3 + 1*step_size));

                    const float mid = 3*step_size/2;

                    for (int q = dim3; q < offset_max_dim3; ++q) {

                        for (int k = dim2; k < offset_max_dim2; ++k) {

                            for (int i = dim1; i < offset_max_dim1; ++i) {

                                max_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num] = temp_int;
                                min_img.mesh[i + (k) * min_img.y_num + q * min_img.y_num * min_img.x_num] = temp_int;

                            }
                        }
                    }


                } else {

                    curr_level.update_gap();

                }


            }
        }
    }







}

template<typename V>
void set_zero_minus_1(ExtraPartCellData<V>& parts){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //



    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth < parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                std::fill(parts.data[depth][pc_offset].begin(),parts.data[depth][pc_offset].end(),0);

            }
        }
    }




}
template<typename V>
void set_zero(ExtraPartCellData<V>& parts){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //



    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth <= parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                std::fill(parts.data[depth][pc_offset].begin(),parts.data[depth][pc_offset].end(),0);

            }
        }
    }




}
template<typename V,class UnaryOperator>
void transform_parts(ExtraPartCellData<V>& parts,ExtraPartCellData<V>& parts2,UnaryOperator op){
    //
    //  Bevan Cheeseman 2017
    //
    //  Takes two particle data sets and adds them, and puts it in the first one
    //

    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth <= parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                std::transform(parts.data[depth][pc_offset].begin(), parts.data[depth][pc_offset].end(), parts2.data[depth][pc_offset].begin(), parts.data[depth][pc_offset].begin(), op);

            }
        }
    }

}
template<typename V,typename U,class BinaryPredicate>
void threshold_parts(ExtraPartCellData<V>& parts,ExtraPartCellData<U>& parts2,float th,float set_val,BinaryPredicate pred){
    //
    //  Bevan Cheeseman 2017
    //
    //  Takes two particle and compares them in some way then replaces them if the condition is met
    //

    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth <= parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < parts.data[depth][pc_offset].size(); ++j_) {

                    float val_th = parts2.data[depth][pc_offset][j_];

                    if(pred(val_th,th)){
                        parts2.data[depth][pc_offset][j_] = set_val;
                    }

                }

            }
        }
    }

}
template<typename V,class BinaryPredicate>
void threshold_parts(ExtraPartCellData<V>& parts,V th,V set_val,BinaryPredicate pred){
    //
    //  Bevan Cheeseman 2017
    //
    //  Takes two particle and compares them in some way then replaces them if the condition is met
    //

    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth <= parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < parts.data[depth][pc_offset].size(); ++j_) {

                    float val_th = parts.data[depth][pc_offset][j_];

                    if(pred(val_th,th)){
                        parts.data[depth][pc_offset][j_] = set_val;
                    }

                }

            }
        }
    }

}


template<typename V>
V square(V input){
    return pow(input,2);
}
template<typename V>
V square_root(V input){
    return sqrt(input);
}

template<typename V,class UnaryOperator>
ExtraPartCellData<V> transform_parts(ExtraPartCellData<V>& parts,UnaryOperator op){
    //
    //  Bevan Cheeseman 2017
    //
    //  Takes two particle data sets and adds them, and puts it in the first one
    //

    ExtraPartCellData<V> output;
    output.initialize_structure_parts(parts);

    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth <= parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                std::transform(parts.data[depth][pc_offset].begin(),parts.data[depth][pc_offset].end(),output.data[depth][pc_offset].begin(),op);

            }
        }
    }

    return output;

}
template<typename V>
ExtraPartCellData<V> multiply_by_depth(ExtraPartCellData<V>& parts){
    //
    //  Bevan Cheeseman 2017
    //
    //  Sets value to depth if non-zero
    //

    ExtraPartCellData<V> output;
    output.initialize_structure_parts(parts);

    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth <= parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        const float step = pow(2,parts.depth_max - depth);


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                for (int i = 0; i < parts.data[depth][pc_offset].size(); ++i) {
                    if((parts.data[depth][pc_offset][i]> 0)) {
                        output.data[depth][pc_offset][i] = (1.0 * (z_ + 0.5)) * step;
                        //output.data[depth][pc_offset][i] = parts.data[depth][pc_offset][i];
                    }
                }

            }
        }
    }

    return output;

}
template<typename V>
ExtraPartCellData<V> multiply_by_dist_center(ExtraPartCellData<V>& parts,ExtraPartCellData<uint16_t>& y_vec){
    //
    //  Bevan Cheeseman 2017
    //
    //  Sets value to depth if non-zero
    //

    float x_c = y_vec.org_dims[1]*.5;
    float y_c = y_vec.org_dims[0]*.5;
    float z_c = y_vec.org_dims[2]*.5;


    ExtraPartCellData<V> output;
    output.initialize_structure_parts(parts);

    int z_,x_,j_,y_;

    for(uint64_t depth = (parts.depth_min);depth <= parts.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = parts.x_num[depth];
        const unsigned int z_num_ = parts.z_num[depth];

        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        const float step = pow(2,parts.depth_max - depth);


#pragma omp parallel for default(shared) private(z_,x_,j_) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                for (int i = 0; i < parts.data[depth][pc_offset].size(); ++i) {
                    if((parts.data[depth][pc_offset][i]> 0)) {

                        unsigned int y_ = y_vec.data[depth][pc_offset][i];
                        float dist = sqrt(pow((1.0 * (z_ + 0.5)) * step - z_c,2) + pow((1.0 * (x_ + 0.5)) * step - x_c,2) + pow((1.0 * (y_ + 0.5)) * step - y_c,2));

                        output.data[depth][pc_offset][i] = dist;
                        //output.data[depth][pc_offset][i] = parts.data[depth][pc_offset][i];
                    }
                }

            }
        }
    }

    return output;

}


template<typename U,typename T,typename V>
ExtraPartCellData<U> convert_cell_to_part(PartCellStructure<V,T>& pc_struct,ExtraPartCellData<U>& input){
    //
    //
    //  Bevan Cheeseman 2017
    //
    //
    //  Converts data from cells to particles
    //
    //


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

    ExtraPartCellData<float> output;

    output.initialize_structure_parts(pc_struct.part_data.particle_data);


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

                        float loc_min = input.get_val(curr_key);


                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            //first set the particle index value in the particle_data array (stores the intensities)
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            //get all the neighbour particles in (+y,-y,+x,-x,+z,-z) ordering

                            output.get_part(curr_key) = loc_min;

                        }
                    }

                }

            }

        }
    }



    return output;


}




template<typename U,typename T,typename S>
void shift_particles_from_cells(ParticleDataNew<S, T>& part_new,ExtraPartCellData<U>& pdata_old){
    //
    //  Bevan Cheesean 2017
    //
    //  Transfers them to align with the part data, to align with particle data no gaps
    //
    //

    ExtraPartCellData<U> pdata_new;

    pdata_new.initialize_structure_parts(part_new.particle_data);

    uint64_t z_,x_,j_,node_val;
    uint64_t part_offset;

    for(uint64_t i = part_new.access_data.depth_min;i <= part_new.access_data.depth_max;i++){

        const unsigned int x_num_ = part_new.access_data.x_num[i];
        const unsigned int z_num_ = part_new.access_data.z_num[i];

#pragma omp parallel for default(shared) private(z_,x_,j_,part_offset,node_val)  if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){

            for(x_ = 0;x_ < x_num_;x_++){
                const size_t offset_pc_data = x_num_*z_ + x_;
                const size_t j_num = part_new.access_data.data[i][offset_pc_data].size();

                int counter = 0;

                for(j_ = 0; j_ < j_num;j_++){
                    //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                    node_val = part_new.access_data.data[i][offset_pc_data][j_];

                    if(!(node_val&1)){

                        pdata_new.data[i][offset_pc_data][counter] = pdata_old.data[i][offset_pc_data][j_];

                        counter++;

                    } else {

                    }

                }
            }
        }
    }

    std::swap(pdata_new,pdata_old);

}

template<typename U,typename V>
void interp_slice(Mesh_data<U>& slice,PartCellData<uint64_t>& pc_data,ParticleDataNew<float, uint64_t>& part_new,ExtraPartCellData<V>& particles_int,int dir,int num){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //


    std::vector<unsigned int> x_num_min;
    std::vector<unsigned int> x_num;

    std::vector<unsigned int> z_num_min;
    std::vector<unsigned int> z_num;

    x_num.resize(pc_data.depth_max + 1);
    z_num.resize(pc_data.depth_max + 1);
    x_num_min.resize(pc_data.depth_max + 1);
    z_num_min.resize(pc_data.depth_max + 1);


    if(dir != 1) {
        //yz case
        z_num = pc_data.z_num;

        for (int i = pc_data.depth_min; i <= pc_data.depth_max ; ++i) {
            x_num[i] = num/pow(2,pc_data.depth_max - i) + 1;
            z_num_min[i] = 0;
            x_num_min[i] = num/pow(2,pc_data.depth_max - i);
        }

        slice.initialize(pc_data.org_dims[0],pc_data.org_dims[2],1,0);


    } else {
        //yx case
        x_num = pc_data.x_num;

        for (int i = pc_data.depth_min; i <= pc_data.depth_max ; ++i) {
            z_num[i] = num/pow(2,pc_data.depth_max - i) + 1;
            x_num_min[i] = 0;
            z_num_min[i] = num/pow(2,pc_data.depth_max - i);
        }

        slice.initialize(pc_data.org_dims[0],pc_data.org_dims[1],1,0);

    }

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = x_num[depth];
        const unsigned int z_num_ = z_num[depth];

        const unsigned int x_num_min_ = x_num_min[depth];
        const unsigned int z_num_min_ = z_num_min[depth];

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        int dim1 = 0;
                        int dim2 = 0;

                        get_coord(dir,curr_level,step_size,dim1,dim2);

                        //add to all the required rays

//                        for (int k = 0; k < step_size; ++k) {
//pragma omp simd
//                            for (int i = 0; i < step_size; ++i) {
//                                slice.mesh[dim1 + i + (dim2 + k)*slice.y_num] = temp_int;
//
//                                //slice(dim1 + i,(dim2 + k),1) = temp_int;
//                            }
//                        }

                        //add to all the required rays

                        const float temp_int =  curr_level.get_val(particles_int);

                        const int offset_max_dim1 = std::min((int)slice.y_num,(int)(dim1 + step_size));
                        const int offset_max_dim2 = std::min((int)slice.x_num,(int)(dim2 + step_size));


                        for (int k = dim2; k < offset_max_dim2; ++k) {
#pragma omp simd
                            for (int i = dim1; i < offset_max_dim1; ++i) {
                                slice.mesh[ i + (k)*slice.y_num] = temp_int;

                            }
                        }


                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }




}

void create_y_offsets(ExtraPartCellData<uint16_t>& y_off,ParticleDataNew<float, uint64_t>& part_new,PartCellData<uint64_t>& pc_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Creates y index
    //


    //first add the layers
    y_off.depth_max = pc_data.depth_max;
    y_off.depth_min = pc_data.depth_min;

    y_off.z_num.resize(y_off.depth_max+1);
    y_off.x_num.resize(y_off.depth_max+1);

    y_off.data.resize(y_off.depth_max+1);

    y_off.org_dims = pc_data.org_dims;

    for(uint64_t i = y_off.depth_min;i <= y_off.depth_max;i++){
        y_off.z_num[i] = pc_data.z_num[i];
        y_off.x_num[i] = pc_data.x_num[i];
        y_off.data[i].resize( y_off.z_num[i]*y_off.x_num[i]);
    }

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = part_new.access_data.x_num[depth];
        const unsigned int z_num_ = part_new.access_data.z_num[depth];

        CurrentLevel<float, uint64_t> curr_level(pc_data);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                int counter = 0;

                int y = 0;

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        y++;

                    } else {

                        curr_level.update_gap();

                        int y_curr = y;

                        y += ((curr_level.node_val & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);

                        if(y > 0){

                            y_off.data[depth][curr_level.pc_offset].push_back(y_curr);
                            y_off.data[depth][curr_level.pc_offset].push_back(y);

                        }

                        y--;


                    }


                }
            }
        }
    }



}


void create_y_data(ExtraPartCellData<uint16_t>& y_vec,ParticleDataNew<float, uint64_t>& part_new,PartCellData<uint64_t>& pc_data){
    //
    //  Bevan Cheeseman 2017
    //
    //  Creates y index
    //


    y_vec.initialize_structure_parts(part_new.particle_data);

    y_vec.org_dims = part_new.access_data.org_dims;

    int z_,x_,j_,y_;

    for(uint64_t depth = (part_new.access_data.depth_min);depth <= part_new.access_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = part_new.access_data.x_num[depth];
        const unsigned int z_num_ = part_new.access_data.z_num[depth];

        CurrentLevel<float, uint64_t> curr_level(part_new);
        curr_level.set_new_depth(depth, part_new);

        const float step_size = pow(2,curr_level.depth_max - curr_level.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level) if(z_num_*x_num_ > 100)
        for (z_ = 0; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = 0; x_ < x_num_; x_++) {

                curr_level.set_new_xz(x_, z_, part_new);

                int counter = 0;

                for (j_ = 0; j_ < curr_level.j_num; j_++) {

                    bool iscell = curr_level.new_j(j_, part_new);

                    if (iscell) {
                        //Indicates this is a particle cell node
                        curr_level.update_cell(part_new);

                        y_vec.data[depth][curr_level.pc_offset][counter] = curr_level.y;

                        counter++;
                    } else {

                        curr_level.update_gap();

                    }


                }
            }
        }
    }



}
void create_y_data(ExtraPartCellData<uint16_t>& y_vec,ParticleDataNew<float, uint64_t>& part_new){

    PartCellData<uint64_t> pc_data_temp;

    create_y_data(y_vec,part_new,pc_data_temp);

}

template<typename U>
void interp_slice(Mesh_data<U>& slice,ExtraPartCellData<uint16_t>& y_vec,ExtraPartCellData<U>& particle_data,const int dir,const int num){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //
    //


    std::vector<unsigned int> x_num_min;
    std::vector<unsigned int> x_num;

    std::vector<unsigned int> z_num_min;
    std::vector<unsigned int> z_num;

    x_num.resize(y_vec.depth_max + 1,0);
    z_num.resize(y_vec.depth_max + 1,0);
    x_num_min.resize(y_vec.depth_max + 1,0);
    z_num_min.resize(y_vec.depth_max + 1,0);


    if(dir != 1) {
        //yz case
        z_num = y_vec.z_num;

        for (int i = y_vec.depth_min; i <= y_vec.depth_max ; ++i) {
            x_num[i] = num/pow(2,y_vec.depth_max - i) + 1;
            z_num_min[i] = 0;
            x_num_min[i] = num/pow(2,y_vec.depth_max - i);
        }

    } else {
        //dir = 1 case
        //yx case
        x_num = y_vec.x_num;

        for (int i = y_vec.depth_min; i <= y_vec.depth_max ; ++i) {
            z_num[i] = num/pow(2,y_vec.depth_max - i) + 1;
            x_num_min[i] = 0;
            z_num_min[i] = num/pow(2,y_vec.depth_max - i);
        }


    }

    if(dir == 0){
        //yz
        slice.initialize(y_vec.org_dims[0],y_vec.org_dims[2],1,0);
    } else if (dir == 1){
        //xy
        slice.initialize(y_vec.org_dims[1],y_vec.org_dims[0],1,0);

    } else if (dir == 2){
        //zy
        slice.initialize(y_vec.org_dims[2],y_vec.org_dims[0],1,0);

    }

    int z_,x_,j_,y_;

    for(uint64_t depth = (y_vec.depth_min);depth <= y_vec.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_max = x_num[depth];
        const unsigned int z_num_max = z_num[depth];

        const unsigned int x_num_min_ = x_num_min[depth];
        const unsigned int z_num_min_ = z_num_min[depth];

        const unsigned int x_num_ = y_vec.x_num[depth];

        const float step_size = pow(2,y_vec.depth_max - depth);

//#pragma omp parallel for default(shared) private(z_,x_,j_)
        for (z_ = z_num_min_; z_ < z_num_max; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_max; x_++) {
                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < y_vec.data[depth][pc_offset].size(); j_++) {

                    int dim1 = 0;
                    int dim2 = 0;

                    const int y = y_vec.data[depth][pc_offset][j_];

                    get_coord(dir,y,x_,z_,step_size,dim1,dim2);

                    const float temp_int = particle_data.data[depth][pc_offset][j_];

                    //add to all the required rays
                    const int offset_max_dim1 = std::min((int)slice.y_num,(int)(dim1 + step_size));
                    const int offset_max_dim2 = std::min((int)slice.x_num,(int)(dim2 + step_size));


                    for (int k = dim2; k < offset_max_dim2; ++k) {
#pragma omp simd
                        for (int i = dim1; i < offset_max_dim1; ++i) {
                            slice.mesh[ i + (k)*slice.y_num] = temp_int;

                        }
                    }


                }
            }
        }
    }



}
template<typename U>
void interp_slice_opt(Mesh_data<U>& slice,ExtraPartCellData<uint16_t>& y_vec,ExtraPartCellData<U>& particle_data,const int dir,const int num){
    //
    //  Bevan Cheeseman 2016
    //
    //  Takes in a APR and creates piece-wise constant image
    //
    //

    std::vector<unsigned int> x_num_min;
    std::vector<unsigned int> x_num;

    std::vector<unsigned int> z_num_min;
    std::vector<unsigned int> z_num;

    x_num.resize(y_vec.depth_max + 1,0);
    z_num.resize(y_vec.depth_max + 1,0);
    x_num_min.resize(y_vec.depth_max + 1,0);
    z_num_min.resize(y_vec.depth_max + 1,0);

    if(num == 0) {
        if (dir == 0) {
            //yz
            slice.initialize(y_vec.org_dims[0], y_vec.org_dims[2], 1, 0);
        } else if (dir == 1) {
            //xy
            slice.initialize(y_vec.org_dims[1], y_vec.org_dims[0], 1, 0);

        } else if (dir == 2) {
            //zy
            slice.initialize(y_vec.org_dims[2], y_vec.org_dims[0], 1, 0);

        }
    }


    if (dir != 1) {
        //yz case
        z_num = y_vec.z_num;

        for (int i = y_vec.depth_min; i < y_vec.depth_max; ++i) {

            int step = pow(2, y_vec.depth_max - i);
            int coord = num/step;

            int check1 = (coord*step);

            if(num == check1){
                x_num[i] = num/step + 1;
                x_num_min[i] = num/step;
            }
            z_num_min[i] = 0;
        }
        x_num[y_vec.depth_max] = num + 1;
        x_num_min[y_vec.depth_max] = num;

    } else {
        //yx case
        x_num = y_vec.x_num;

        for (int i = y_vec.depth_min; i < y_vec.depth_max; ++i) {

            int step = pow(2, y_vec.depth_max - i);
            int coord = num/step;

            int check1 = (coord*step);

            if(num == check1){
                z_num[i] = num/step + 1;
                z_num_min[i] = num/step;
            }
            x_num_min[i] = 0;
        }

        z_num[y_vec.depth_max] = num + 1;
        z_num_min[y_vec.depth_max] = num;

    }

    int z_=0;
    int x_=0;
    int j_ = 0;
    int y_ = 0;

    for(uint64_t depth = (y_vec.depth_min);depth <= y_vec.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_max = x_num[depth];
        const unsigned int z_num_max = z_num[depth];

        const unsigned int x_num_min_ = x_num_min[depth];
        const unsigned int z_num_min_ = z_num_min[depth];

        const unsigned int x_num_ = y_vec.x_num[depth];

        const float step_size = pow(2,y_vec.depth_max - depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) if(dir == 1)
        for (z_ = z_num_min_; z_ < z_num_max; z_++) {
            //both z and x are explicitly accessed in the structure
#pragma omp parallel for default(shared) private(x_,j_) if(dir != 1)
            for (x_ = x_num_min_; x_ < x_num_max; x_++) {
                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < y_vec.data[depth][pc_offset].size(); j_++) {

                    int dim1 = 0;
                    int dim2 = 0;

                    const int y = y_vec.data[depth][pc_offset][j_];

                    get_coord(dir,y,x_,z_,step_size,dim1,dim2);

                    const float temp_int = particle_data.data[depth][pc_offset][j_];

                    //add to all the required rays
                    const int offset_max_dim1 = std::min((int)slice.y_num,(int)(dim1 + step_size));
                    const int offset_max_dim2 = std::min((int)slice.x_num,(int)(dim2 + step_size));


                    for (int k = dim2; k < offset_max_dim2; ++k) {
                        for (int i = dim1; i < offset_max_dim1; ++i) {
                            slice.mesh[ i + (k)*slice.y_num] = temp_int;

                        }
                    }


                }
            }
        }
    }



}

template<typename U>
std::vector<U> shift_filter(std::vector<U>& filter){
    //
    //  Filter shift for non resolution part locations
    //
    //  Bevan Cheeseman 2017
    //

    std::vector<U> filter_d;

    filter_d.resize(filter.size() + 1,0);

    float factor = 0.5/4.0;

    filter_d[0] = filter[0]*factor;

    for (int i = 0; i < (filter.size() -1); ++i) {
        filter_d[i + 1] = filter[i]*factor + filter[i+1]*factor;
    }

    filter_d.back() = filter.back()*factor;

    return filter_d;

}

template<typename U>
void filter_apr_dir(ExtraPartCellData<uint16_t>& y_vec,ExtraPartCellData<U>& filter_output,ExtraPartCellData<U>& filter_input,std::vector<U>& filter,std::vector<U>& filter_d,const int dir){
    //
    //
    //
    //

    Mesh_data<U> slice;

    int num_slices;

    if (dir != 1) {
        num_slices = y_vec.org_dims[1];
    } else {
        num_slices = y_vec.org_dims[2];
    }

    if (dir == 0) {
        //yz
        slice.initialize(y_vec.org_dims[0], y_vec.org_dims[2], 1, 0);
    } else if (dir == 1) {
        //xy
        slice.initialize(y_vec.org_dims[1], y_vec.org_dims[0], 1, 0);

    } else if (dir == 2) {
        //zy
        slice.initialize(y_vec.org_dims[2], y_vec.org_dims[0], 1, 0);

    }

    int i = 0;
#pragma omp parallel for default(shared) private(i) firstprivate(slice) schedule(static)
    for (i = 0; i < num_slices; ++i) {
        interp_slice(slice, y_vec, filter_input, dir, i);

        filter_slice(filter,filter_d,filter_output,slice,y_vec,dir,i);
    }

}
template<typename U>
void get_slice(Mesh_data<U>& input_img,Mesh_data<U>& slice,const int dir,const int num){
    //
    //  To use the algorithms we must transpose the slice in the correct directions
    //
    //

    if (dir == 0) {
        //yz
        slice.initialize(input_img.y_num, input_img.z_num, 1, 0);

        int i = 0;
        int j = 0;

#pragma omp simd
        for (int i = 0; i < slice.x_num; ++i) {
            for (int j = 0; j < slice.y_num; ++j) {
                slice.mesh[j + i*slice.y_num] = input_img.mesh[j + i*input_img.y_num*input_img.x_num + num*input_img.y_num];
            }
        }


    } else if (dir == 1) {
        //xy
        slice.initialize(input_img.x_num, input_img.y_num, 1, 0);

#pragma omp simd
        for (int i = 0; i < slice.x_num; ++i) {
            for (int j = 0; j < slice.y_num; ++j) {
                slice.mesh[j + i*slice.y_num] = input_img.mesh[i + j*input_img.y_num + num*input_img.y_num*input_img.x_num];
            }
        }


    } else if (dir == 2) {
        //zy
        slice.initialize(input_img.z_num, input_img.y_num, 1, 0);

#pragma omp simd
        for (int i = 0; i < slice.x_num; ++i) {
            for (int j = 0; j < slice.y_num; ++j) {
                slice.mesh[j + i*slice.y_num] = input_img.mesh[i + j*input_img.y_num*input_img.x_num + num*input_img.y_num];
            }
        }

    }



}



template<typename U>
void filter_apr_mesh_dir(Mesh_data<U>& input_img,ExtraPartCellData<uint16_t>& y_vec,ExtraPartCellData<U>& filter_output,ExtraPartCellData<U>& filter_input,std::vector<U>& filter,std::vector<U>& filter_d,const int dir){
    //
    //  This convolves at APR locations from an input mesh
    //

    Mesh_data<U> slice;

    int num_slices;

    if (dir != 1) {
        num_slices = y_vec.org_dims[1];
    } else {
        num_slices = y_vec.org_dims[2];
    }

    if (dir == 0) {
        //yz
        slice.initialize(y_vec.org_dims[0], y_vec.org_dims[2], 1, 0);
    } else if (dir == 1) {
        //xy
        slice.initialize(y_vec.org_dims[1], y_vec.org_dims[0], 1, 0);

    } else if (dir == 2) {
        //zy
        slice.initialize(y_vec.org_dims[2], y_vec.org_dims[0], 1, 0);

    }

    int i = 0;
#pragma omp parallel for default(shared) private(i) firstprivate(slice) schedule(guided)
    for (i = 0; i < num_slices; ++i) {
        get_slice(input_img,slice,dir,i);

        filter_slice(filter,filter_d,filter_output,slice,y_vec,dir,i);
    }

}

template<typename U,typename S>
void convert_from_old_structure(ExtraPartCellData<U>& particle_data,PartCellStructure<U,uint64_t>& pc_struct,PartCellData<S>& pc_data,ExtraPartCellData<U>& old_part_data,bool type){


    //first add the layers
    particle_data.depth_max = pc_struct.depth_max + 1;
    particle_data.depth_min = pc_struct.depth_min;


    particle_data.z_num.resize(particle_data.depth_max+1);
    particle_data.x_num.resize(particle_data.depth_max+1);

    particle_data.data.resize(particle_data.depth_max+1);

    for(uint64_t i = particle_data.depth_min;i < particle_data.depth_max;i++){
        particle_data.z_num[i] = pc_struct.z_num[i];
        particle_data.x_num[i] = pc_struct.x_num[i];
        particle_data.data[i].resize(particle_data.z_num[i]*particle_data.x_num[i]);
    }

    particle_data.z_num[particle_data.depth_max] = pc_struct.org_dims[2];
    particle_data.x_num[particle_data.depth_max] = pc_struct.org_dims[1];
    particle_data.data[particle_data.depth_max].resize(particle_data.z_num[particle_data.depth_max]*particle_data.x_num[particle_data.depth_max]);

    pc_data.z_num.resize(pc_data.depth_max+1);
    pc_data.x_num.resize(pc_data.depth_max+1);
    pc_data.y_num.resize(pc_data.depth_max+1);


    for(uint64_t i = pc_data.depth_min;i < pc_data.depth_max;i++){
        pc_data.z_num[i] = pc_struct.z_num[i];
        pc_data.x_num[i] = pc_struct.x_num[i];
        pc_data.y_num[i] = pc_struct.y_num[i];

    }

    pc_data.z_num[pc_data.depth_max] = pc_struct.org_dims[2];
    pc_data.x_num[pc_data.depth_max] = pc_struct.org_dims[1];
    pc_data.y_num[pc_data.depth_max] = pc_struct.org_dims[0];


    //initialize loop variables
    int x_;
    int z_;
    int y_;

    int x_seed;
    int z_seed;
    int y_seed;

    uint64_t j_;

    uint64_t status;
    uint64_t node_val;
    uint16_t node_val_part;

    //next initialize the entries;
    Part_timer timer;
    timer.verbose_flag = false;

    std::vector<uint16_t> temp_exist;
    std::vector<uint16_t> temp_location;

    timer.start_timer("intiialize access data structure");

    for(uint64_t i = pc_data.depth_max;i >= pc_data.depth_min;i--){

        const unsigned int x_num = pc_data.x_num[i];
        const unsigned int z_num = pc_data.z_num[i];


        const unsigned int x_num_seed = pc_data.x_num[i-1];
        const unsigned int z_num_seed = pc_data.z_num[i-1];

        temp_exist.resize(pc_data.y_num[i]);
        temp_location.resize(pc_data.y_num[i]);

#pragma omp parallel for default(shared) private(j_,z_,x_,y_,node_val,status,z_seed,x_seed,node_val_part) firstprivate(temp_exist,temp_location) if(z_num*x_num > 100)
        for(z_ = 0;z_ < z_num;z_++){

            for(x_ = 0;x_ < x_num;x_++){

                std::fill(temp_exist.begin(), temp_exist.end(), 0);
                std::fill(temp_location.begin(), temp_location.end(), 0);

                if( i < pc_data.depth_max){
                    //access variables
                    const size_t offset_pc_data = x_num*z_ + x_;
                    const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();

                    y_ = 0;

                    //first loop over
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val = pc_struct.pc_data.data[i][offset_pc_data][j_];
                        node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];

                        if(!(node_val&1)){
                            //normal node
                            y_++;
                            //create pindex, and create status (0,1,2,3) and type
                            status = (node_val & STATUS_MASK) >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.
                            uint16_t part_offset = (node_val_part & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;

                            if(status > SEED){
                                temp_exist[y_] = status;
                                temp_location[y_] = part_offset;
                            }

                        } else {

                            y_ = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_--;
                        }
                    }
                }

                x_seed = x_/2;
                z_seed = z_/2;

                if( i > pc_data.depth_min){
                    //access variables
                    size_t offset_pc_data = x_num_seed*z_seed + x_seed;
                    const size_t j_num = pc_struct.pc_data.data[i-1][offset_pc_data].size();


                    y_ = 0;

                    //first loop over
                    for(j_ = 0; j_ < j_num;j_++){
                        //raster over both structures, generate the index for the particles, set the status and offset_y_coord diff

                        node_val_part = pc_struct.part_data.access_data.data[i-1][offset_pc_data][j_];
                        node_val = pc_struct.pc_data.data[i-1][offset_pc_data][j_];

                        if(!(node_val&1)){
                            //normal node
                            y_++;
                            //create pindex, and create status (0,1,2,3) and type
                            status = (node_val & STATUS_MASK) >> STATUS_SHIFT;  //need the status masks here, need to move them into the datastructure I think so that they are correctly accessible then to these routines.
                            uint16_t part_offset = (node_val_part & Y_PINDEX_MASK_PARTICLE) >> Y_PINDEX_SHIFT_PARTICLE;

                            if(status == SEED){
                                temp_exist[2*y_] = status;
                                temp_exist[2*y_+1] = status;

                                temp_location[2*y_] = part_offset + (z_&1)*4 + (x_&1)*2;
                                temp_location[2*y_+1] = part_offset + (z_&1)*4 + (x_&1)*2 + 1;

                            }

                        } else {

                            y_ = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_--;
                        }
                    }
                }


                size_t first_empty = 0;

                size_t offset_pc_data = x_num*z_ + x_;
                size_t offset_pc_data_seed = x_num_seed*z_seed + x_seed;
                size_t curr_index = 0;
                size_t prev_ind = 0;

                //first value handle the duplication of the gap node



                size_t part_total= 0;

                for(y_ = 0;y_ < temp_exist.size();y_++){

                    status = temp_exist[y_];

                    if(status> 0){
                        curr_index+= 1 + prev_ind;
                        prev_ind = 0;
                        part_total++;
                    } else {
                        prev_ind = 1;
                    }
                }


                //initialize particles
                particle_data.data[i][offset_pc_data].resize(part_total);

                curr_index = 0;
                prev_ind = 1;
                size_t prev_coord = 0;

                size_t part_counter=0;



                for(y_ = 0;y_ < temp_exist.size();y_++){

                    status = temp_exist[y_];

                    if((status> 0)){

                        curr_index++;


                        //lastly retrieve the intensities
                        if(status == SEED){
                            //seed from up one level
                            particle_data.data[i][offset_pc_data][part_counter] = old_part_data.data[i-1][offset_pc_data_seed][temp_location[y_]];
                        }
                        else {
                            //non seed same level
                            particle_data.data[i][offset_pc_data][part_counter] = old_part_data.data[i][offset_pc_data][temp_location[y_]];
                        }

                        part_counter++;


                        prev_ind = 0;
                    } else {

                    }
                }


            }

        }
    }





}


template<typename V>
void filter_slice(std::vector<V>& filter,std::vector<V>& filter_d,ExtraPartCellData<V>& filter_output,Mesh_data<V>& slice,ExtraPartCellData<uint16_t>& y_vec,const int dir,const int num){

    int filter_offset = (filter.size()-1)/2;


    std::vector<unsigned int> x_num_min;
    std::vector<unsigned int> x_num;

    std::vector<unsigned int> z_num_min;
    std::vector<unsigned int> z_num;

    x_num.resize(y_vec.depth_max + 1,0);
    z_num.resize(y_vec.depth_max + 1,0);
    x_num_min.resize(y_vec.depth_max + 1,0);
    z_num_min.resize(y_vec.depth_max + 1,0);

    std::vector<bool> first_flag;
    first_flag.resize(y_vec.depth_max);

    if (dir != 1) {
        //yz case
        z_num = y_vec.z_num;

        for (int i = y_vec.depth_min; i < y_vec.depth_max; ++i) {

            int step = pow(2, y_vec.depth_max - i);
            int coord = num/step;

            int check1 = ((1.0*coord+.25)*step);
            int check2 = ((1.0*coord+.25)*step) + 1;

            if((num == check1) ){
                x_num[i] = num/step + 1;
                x_num_min[i] = num/step;
                first_flag[i] = false;
            } else if ((num == check2 )){
                x_num[i] = num/step + 1;
                x_num_min[i] = num/step;
                first_flag[i] = false;
            }
            z_num_min[i] = 0;
        }
        x_num[y_vec.depth_max] = num + 1;
        x_num_min[y_vec.depth_max] = num;

    } else {
        //yx case
        x_num = y_vec.x_num;

        for (int i = y_vec.depth_min; i < y_vec.depth_max; ++i) {

            int step = pow(2, y_vec.depth_max - i);
            int coord = num/step;

            int check1 = floor((1.0*coord+.25)*step);
            int check2 = floor((1.0*coord+.25)*step) + 1;

            if((num == check1) ){
                z_num[i] = num/step + 1;
                z_num_min[i] = num/step;
                first_flag[i] = false;
            } else if ((num == check2 )) {
                z_num[i] = num/step + 1;
                z_num_min[i] = num/step;
                first_flag[i] = false;
            }

            x_num_min[i] = 0;
        }

        z_num[y_vec.depth_max] = num + 1;
        z_num_min[y_vec.depth_max] = num;

    }

    int z_,x_,j_,y_;

    for(uint64_t depth = (y_vec.depth_min);depth <= y_vec.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_max = x_num[depth];
        const unsigned int z_num_max = z_num[depth];

        const unsigned int x_num_min_ = x_num_min[depth];
        const unsigned int z_num_min_ = z_num_min[depth];

        const unsigned int x_num_ = y_vec.x_num[depth];

        const float step_size = pow(2,y_vec.depth_max - depth);

//#pragma omp parallel for default(shared) private(z_,x_,j_)
        for (z_ = z_num_min_; z_ < z_num_max; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_max; x_++) {
                const unsigned int pc_offset = x_num_*z_ + x_;

                for (j_ = 0; j_ < y_vec.data[depth][pc_offset].size(); j_++) {

                    int dim1 = 0;
                    int dim2 = 0;

                    const int y = y_vec.data[depth][pc_offset][j_];

                    get_coord_filter(dir,y,x_,z_,step_size,dim1,dim2);

                    if(depth == y_vec.depth_max) {

                        const int offset_max = std::min((int)(dim1 + filter_offset),(int)(slice.y_num-1));
                        const int offset_min = std::max((int)(dim1 - filter_offset),(int)0);

                        int f = 0;
                        V temp = 0;

                        for (int c = offset_min; c <= offset_max; c++) {

                            //need to change the below to the vector
                            temp += slice.mesh[c + (dim2) * slice.y_num] * filter[f];
                            f++;
                        }

                        filter_output.data[depth][pc_offset][j_] += temp;
                    } else {

                        const int offset_max = std::min((int)(dim1 + filter_offset + 1),(int)(slice.y_num-1));
                        const int offset_min = std::max((int)(dim1 - filter_offset),(int)0);

                        int f = 0;
                        V temp = 0;

                        const int dim2p = std::min(dim2 + 1,slice.x_num-1);
                        const int dim2m = std::min(dim2,slice.x_num-1);

                        for (int c = offset_min; c <= offset_max; c++) {

                            //need to change the below to the vector
                            temp += (slice.mesh[c + (dim2m) * slice.y_num] + slice.mesh[c + (dim2p) * slice.y_num])  * filter_d[f];
                            f++;
                        }

                        filter_output.data[depth][pc_offset][j_] += temp;

                    }



                }
            }
        }
    }


}



#endif
    
    




