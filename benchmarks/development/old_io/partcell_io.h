//
//
//  Part Play Library
//
//  Bevan Cheeseman 2015
//
//  read_parts.h
//
//
//  Created by cheesema on 11/19/15.
//
//  This header contains the functions for loading in the particle representations from file
//
//

#ifndef _partcell_io_h
#define _partcell_io_h

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "benchmarks/development/old_io/hdf5functions.h"
#include "src/io/hdf5functions_blosc.h"
#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "write_parts.h"
#include "writeimage.h"
#include "benchmarks/development/old_numerics/apr_compression.hpp"
#include "src/data_structures/APR/APR.hpp"


template<typename T>
void write_apr_full_format(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name);

template<typename T>
void write_apr_pc_struct(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name);


template<typename T,typename U>
void write_apr_wavelet(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name,float comp_factor);


template<typename T>
void read_apr_full_format(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name);


template<typename T>
void write_apr_full_format(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes APR to hdf5 file, including xdmf to make readible by paraview, and includes any fields that are flagged extra to print
    //
    //

    register_bosc();

    std::cout << "Writing parts to hdf5 file, in full format..." << std::endl;
    
    //containers for all the variables
    std::vector<uint8_t> k_vec;
    
    std::vector<uint8_t> type_vec;
    
    std::vector<uint16_t> x_c;
    std::vector<uint16_t> y_c;
    std::vector<uint16_t> z_c;
    
    std::vector<uint16_t> Ip;
    
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    
    k_vec.resize(num_parts);
    type_vec.resize(num_parts);
    
    x_c.resize(num_parts);
    z_c.resize(num_parts);
    y_c.resize(num_parts);
    
    Ip.resize(num_parts);
    
    //initialize
    uint64_t node_val_part;
    uint64_t y_coord;
    int x_;
    int z_;
 
    uint64_t j_;
    uint64_t status;
    uint64_t curr_key=0;
    uint64_t part_offset=0;
    
    
    
    uint64_t p;
    
    uint64_t counter = 0;
    
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.pc_data.x_num[i];
        const unsigned int z_num_ = pc_struct.pc_data.z_num[i];
        
        //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
        
        // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
//#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_part,curr_key,part_offset,status) if(z_num_*x_num_ > 100)
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            pc_struct.part_data.access_data.pc_key_set_depth(curr_key,i);
            pc_struct.part_data.access_data.pc_key_set_z(curr_key,z_);
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                y_coord = 0;
                
                for(j_ = 0;j_ < j_num;j_++){
                    
                    
                    node_val_part = pc_struct.part_data.access_data.data[i][offset_pc_data][j_];
                    
                    if (!(node_val_part&1)){
                        //get the index gap node
                        y_coord++;
                        
                        pc_struct.part_data.access_data.pc_key_set_j(curr_key,j_);
                        
                        //neigh_keys.resize(0);
                        status = pc_struct.part_data.access_node_get_status(node_val_part);
                        part_offset = pc_struct.part_data.access_node_get_part_offset(node_val_part);
                        
                        pc_struct.part_data.access_data.pc_key_set_status(curr_key,status);
                        
                
                        
                        
                        //loop over the particles
                        for(p = 0;p < pc_struct.part_data.get_num_parts(status);p++){
                            pc_struct.part_data.access_data.pc_key_set_index(curr_key,part_offset+p);
                            pc_struct.part_data.access_data.pc_key_set_partnum(curr_key,p);
                            
                            //set the cooridnates info
                            pc_struct.part_data.access_data.get_coordinates_part_full(y_coord,curr_key,x_c[counter],z_c[counter],y_c[counter],k_vec[counter],type_vec[counter]);
                            //get the intensity
                            Ip[counter] = pc_struct.part_data.get_part(curr_key);
                            
                            counter++;
                            
                        }
                        
                    } else {
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
    }
    

    
    
    std::string hdf5_file_name = save_loc + file_name + "_full_part.h5";
    
    file_name = file_name + "_full_part";
    
    hdf5_create_file(hdf5_file_name);
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id;
    H5G_info_t info;
    
    hsize_t     dims_out[2];

    hsize_t rank = 1;
    
    hsize_t dims;
    
    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);
    
    //Get the group you want to open
    
    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;
    
    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &pc_struct.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &pc_struct.org_dims[0] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &pc_struct.org_dims[2] );
    
    
    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    
    dims_out[0] = 1;
    dims_out[1] = 1;
    
    //just an identifier in here for the reading of the parts
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_max",1,&dims, &pc_struct.depth_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_min",1,&dims, &pc_struct.depth_min );
    //////////////////////////////////////////////////////////////////
    //
    //  Write data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    
    dims = num_parts;
    
    //write the x co_ordinates
    
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"x",rank,&dims, x_c.data() );
    
    //write the y co_ordinates
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"y",rank,&dims, y_c.data() );
    
    //write the z co_ordinates
    
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"z",rank,&dims, z_c.data() );
    
    //write the z co_ordinates
    
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT8,"type",rank,&dims, type_vec.data() );
    
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT8,"k",rank,&dims, k_vec.data() );
    
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"Ip",rank,&dims,Ip.data()  );
    
    // output the file size
    hsize_t file_size;
    H5Fget_filesize(fid, &file_size);
    
    std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    //create the xdmf file
    write_full_xdmf_xml(save_loc,file_name,num_parts);
    
    std::cout << "Writing Complete" << std::endl;
    
}

template<typename T>
void write_apr_pc_struct_old(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
    //
    //
    
    
    
    std::cout << "Writing parts to hdf5 file, in pc_struct format..." << std::endl;
    
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    

    
    //initialize
    uint64_t node_val_part;
    uint64_t y_coord;
    int x_;
    int z_;
    
    uint64_t j_;
    uint64_t status;
    uint64_t curr_key=0;
    uint64_t part_offset=0;
    
    
    //Neighbour Routine Checking
    
    uint64_t p;
    
    std::string hdf5_file_name = save_loc + file_name + "_pcstruct_part.h5";
    
    file_name = file_name + "_pcstruct_part";
    
    hdf5_create_file(hdf5_file_name);
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id;
    H5G_info_t info;
    
    hsize_t     dims_out[2];
    
    hsize_t rank = 1;
    
    hsize_t dims;
    hsize_t dim_a=1;
    
    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);
    
    //Get the group you want to open
    
    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;
    
    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &pc_struct.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &pc_struct.org_dims[0] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &pc_struct.org_dims[2] );
    
    
    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    
    dims_out[0] = 1;
    dims_out[1] = 1;
    
    //just an identifier in here for the reading of the parts
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );
    
      //////////////////////////////////////////////////////////////////
    //
    //  Write data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    
    uint64_t depth_min = pc_struct.depth_min;
    std::vector<uint8_t> p_map;
    std::vector<uint16_t> Ip;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        
        //write the vals
        
        Ip.resize(0);
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.part_data.particle_data.data[i][offset_pc_data].size();
                
                uint64_t curr_size = Ip.size();
                Ip.resize(curr_size+ j_num);
                
                std::copy(pc_struct.part_data.particle_data.data[i][offset_pc_data].begin(),pc_struct.part_data.particle_data.data[i][offset_pc_data].end(),Ip.begin() + curr_size);
                
            }
            
        }
        
        if(Ip.size() > 0){
            //write the parts
            dims = Ip.size();
            std::string name = "Ip_"+std::to_string(i);
            hdf5_write_data(obj_id,H5T_NATIVE_UINT16,name.c_str(),rank,&dims, Ip.data());
            
            name = "Ip_size_"+std::to_string(i);
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);
            
            p_map.resize(x_num_*z_num_*y_num_,0);
            
            std::fill(p_map.begin(), p_map.end(), 0);
            
            
            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
            
            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_part,curr_key,part_offset,status,y_coord) if(z_num_*x_num_ > 100)
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
                            p_map[offset_p_map + y_coord] = status;
                            
                        } else {
                            
                            y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                            y_coord--;
                        }
                        
                    }
                    
                }
                
            }
            
            dims = p_map.size();
            name = "p_map_"+std::to_string(i);
            hdf5_write_data(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, p_map.data());
            
            name = "p_map_x_num_"+std::to_string(i);
            hsize_t attr = x_num_;
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
        
            attr = y_num_;
            name = "p_map_y_num_"+std::to_string(i);
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
            
            attr = z_num_;
            name = "p_map_z_num_"+std::to_string(i);
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
            
            
        } else {
            depth_min = i+1;
        }
        
    }

    hsize_t attr = depth_min;
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &pc_struct.depth_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );
    
    
    // output the file size
    hsize_t file_size;
    H5Fget_filesize(fid, &file_size);
    
    std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    std::cout << "Writing Complete" << std::endl;
    
    
    
    
}
template<typename T>
void write_apr_pc_struct(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
    //
    //
    
    
    
    //std::cout << "Writing parts to hdf5 file, in pc_struct format..." << std::endl;
    
    
    uint64_t num_cells = pc_struct.get_number_cells();
    uint64_t num_parts = pc_struct.get_number_parts();
    
    
    
    //initialize
    uint64_t node_val_part;
    uint64_t y_coord;
    int x_;
    int z_;
    
    uint64_t j_;
    uint64_t status;
    uint64_t curr_key=0;
    uint64_t part_offset=0;
    
    
    //Neighbour Routine Checking
    
    uint64_t p;
    
    register_bosc();
    
    std::string hdf5_file_name = save_loc + file_name + "_pcstruct_part.h5";
    
    file_name = file_name + "_pcstruct_part";
    
    hdf5_create_file(hdf5_file_name);
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id;
    H5G_info_t info;
    
    hsize_t     dims_out[2];
    
    hsize_t rank = 1;
    
    hsize_t dims;
    hsize_t dim_a=1;
    
    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);
    
    //Get the group you want to open
    
    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;
    
    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &pc_struct.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &pc_struct.org_dims[0] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &pc_struct.org_dims[2] );
    
    
    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    
    dims_out[0] = 1;
    dims_out[1] = 1;
    
    //just an identifier in here for the reading of the parts
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

    // New parameter and background data

    if(pc_struct.pars.name.size() == 0){
        pc_struct.pars.name = "no_name";
        pc_struct.name = "no_name";
    }

    hdf5_write_string(pr_groupid,"name",pc_struct.pars.name);

    std::string git_hash = exec_blosc("git rev-parse HEAD");

    hdf5_write_string(pr_groupid,"githash",git_hash);

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"lambda",1,dims_out, &pc_struct.pars.lambda );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"var_th",1,dims_out, &pc_struct.pars.var_th );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"var_th_max",1,dims_out, &pc_struct.pars.var_th_max );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"I_th",1,dims_out, &pc_struct.pars.I_th );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"dx",1,dims_out, &pc_struct.pars.dx );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"dy",1,dims_out, &pc_struct.pars.dy );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"dz",1,dims_out, &pc_struct.pars.dz );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"psfx",1,dims_out, &pc_struct.pars.psfx );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"psfy",1,dims_out, &pc_struct.pars.psfy );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"psfz",1,dims_out, &pc_struct.pars.psfz );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"rel_error",1,dims_out, &pc_struct.pars.rel_error);

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"aniso",1,dims_out, &pc_struct.pars.aniso);

    //////////////////////////////////////////////////////////////////
    //
    //  Write data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    
    uint64_t depth_min = pc_struct.depth_min;
    std::vector<uint8_t> p_map;
    std::vector<uint16_t> Ip;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        
        //write the vals
        
        Ip.resize(0);
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.part_data.particle_data.data[i][offset_pc_data].size();
                
                uint64_t curr_size = Ip.size();
                Ip.resize(curr_size+ j_num);
                
                std::copy(pc_struct.part_data.particle_data.data[i][offset_pc_data].begin(),pc_struct.part_data.particle_data.data[i][offset_pc_data].end(),Ip.begin() + curr_size);
                
            }
            
        }
        
        dims = Ip.size();
        
        std::string name = "Ip_size_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);
        
        if(Ip.size() > 0){
            //write the parts
            
            name = "Ip_"+std::to_string(i);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT16,name.c_str(),rank,&dims, Ip.data());
            
        }
        p_map.resize(x_num_*z_num_*y_num_,0);
        
        std::fill(p_map.begin(), p_map.end(), 0);
        
        
        //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
        
        // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_part,curr_key,part_offset,status,y_coord) if(z_num_*x_num_ > 100)
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
                        p_map[offset_p_map + y_coord] = status;
                        
                    } else {
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
        
        dims = p_map.size();
        name = "p_map_"+std::to_string(i);
        hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, p_map.data());
        
        name = "p_map_x_num_"+std::to_string(i);
        hsize_t attr = x_num_;
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
        
        attr = y_num_;
        name = "p_map_y_num_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
        
        attr = z_num_;
        name = "p_map_z_num_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
        
    }
    
    hsize_t attr = depth_min;
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &pc_struct.depth_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );
    
    
    // output the file size
    hsize_t file_size;
    H5Fget_filesize(fid, &file_size);
    
    std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    std::cout << "Writing Complete" << std::endl;
    
    
    
    
}


template<typename T>
void write_apr_pc_struct_hilbert(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
    //
    //



    //std::cout << "Writing parts to hdf5 file, in pc_struct format..." << std::endl;


    uint64_t num_cells = pc_struct.get_number_cells();
    uint64_t num_parts = pc_struct.get_number_parts();



    //initialize
    uint64_t node_val_part;
    uint64_t y_coord;
    int x_;
    int z_;

    uint64_t j_;
    uint64_t status;
    uint64_t curr_key=0;
    uint64_t part_offset=0;


    //Neighbour Routine Checking

    uint64_t p;

    register_bosc();

    std::string hdf5_file_name = save_loc + file_name + "_pcstruct_part.h5";

    file_name = file_name + "_pcstruct_part";

    hdf5_create_file(hdf5_file_name);

    //hdf5 inits
    hid_t fid, pr_groupid, obj_id;
    H5G_info_t info;

    hsize_t     dims_out[2];

    hsize_t rank = 1;

    hsize_t dims;
    hsize_t dim_a=1;

    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //Get the group you want to open

    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;

    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &pc_struct.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &pc_struct.org_dims[0] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &pc_struct.org_dims[2] );


    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

    dims_out[0] = 1;
    dims_out[1] = 1;

    //just an identifier in here for the reading of the parts

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

    // New parameter and background data

    hdf5_write_string(pr_groupid,"name",pc_struct.pars.name);

    std::string git_hash = exec_blosc("git rev-parse HEAD");

    hdf5_write_string(pr_groupid,"githash",git_hash);

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"lambda",1,dims_out, &pc_struct.pars.lambda );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"var_th",1,dims_out, &pc_struct.pars.var_th );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"var_th_max",1,dims_out, &pc_struct.pars.var_th_max );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"I_th",1,dims_out, &pc_struct.pars.I_th );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"dx",1,dims_out, &pc_struct.pars.dx );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"dy",1,dims_out, &pc_struct.pars.dy );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"dz",1,dims_out, &pc_struct.pars.dz );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"psfx",1,dims_out, &pc_struct.pars.psfx );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"psfy",1,dims_out, &pc_struct.pars.psfy );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"psfz",1,dims_out, &pc_struct.pars.psfz );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"rel_error",1,dims_out, &pc_struct.pars.rel_error);

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"aniso",1,dims_out, &pc_struct.pars.aniso);

    //////////////////////////////////////////////////////////////////
    //
    //  Write data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////

    uint64_t depth_min = pc_struct.depth_min;
    std::vector<uint8_t> p_map;
    std::vector<uint16_t> Ip;

    APR<float> curr_apr(pc_struct);


    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){


        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];


        //write the vals




        p_map.resize(x_num_*z_num_*y_num_,0);

        std::fill(p_map.begin(), p_map.end(), 0);


        //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

        // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_part,curr_key,part_offset,status,y_coord) if(z_num_*x_num_ > 100)
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
                        p_map[offset_p_map + y_coord] = status;

                    } else {

                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }

                }

            }

        }

        dims = p_map.size();
        std::string name = "p_map_"+std::to_string(i);
        hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, p_map.data());

        name = "p_map_x_num_"+std::to_string(i);
        hsize_t attr = x_num_;
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

        attr = y_num_;
        name = "p_map_y_num_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

        attr = z_num_;
        name = "p_map_z_num_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);





    }



    for(uint64_t depth = (curr_apr.particles_int_old.depth_min);depth <= curr_apr.particles_int_old.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = curr_apr.particles_int_old.x_num[depth];
        const unsigned int z_num_ = curr_apr.particles_int_old.z_num[depth];
        unsigned int y_num_ = 0;
        if(depth == curr_apr.particles_int_old.depth_max) {
             y_num_ = pc_struct.org_dims[0];
        } else{
            y_num_ = pc_struct.y_num[depth];
        }
        const unsigned int x_num_min_ = 0;
       const unsigned int z_num_min_ = 0;

        Ip.resize(0);
//
//        MeshData<uint16_t> int_temp;
//        int_temp.initialize(y_num_,x_num_,z_num_,0);
//
//        for (z_ = z_num_min_; z_ < z_num_; z_++) {
//            //both z and x are explicitly accessed in the structure
//
//            for (x_ = x_num_min_; x_ < x_num_; x_++) {
//
//                const unsigned int pc_offset = x_num_*z_ + x_;
//
//                for (j_ = 0; j_ < curr_apr.y_vec.data[depth][pc_offset].size(); j_++) {
//
//                    const int y = curr_apr.y_vec.data[depth][pc_offset][j_];
//
//                    int_temp(y,x_,z_) = curr_apr.particles_intensities.data[depth][pc_offset][j_];
//
//                }
//
//            }
//        }
//
//
//
//        int size_d = pow(2,3*depth);
//        unsigned nBits = depth + 1;
//        unsigned nDims = 3;
//        uint64_t counter = 0;
//
//        for (int j = 0; j < size_d; ++j) {
//            bitmask_t index = j;
//            bitmask_t coord[3] = {0,0,0};
//
//            hilbert_i2c(nDims,nBits, index, coord);
//
//            if((coord[0] < y_num_) & (coord[1] < x_num_) & (coord[2] < z_num_)){
//
//                if(int_temp.mesh[y_num_*x_num_*coord[2] + y_num_*coord[1] + coord[0]]> 0){
//                    counter++;
//                };
//
//            }
//
//        }
//
//        std::cout << counter << std::endl;



        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                const size_t j_num = curr_apr.particles_int_old.data[depth][pc_offset].size();

                uint64_t curr_size = Ip.size();
                Ip.resize(curr_size+ j_num);

                std::copy(curr_apr.particles_int_old.data[depth][pc_offset].begin(),curr_apr.particles_int_old.data[depth][pc_offset].end(),Ip.begin() + curr_size);

            }
        }


        dims = Ip.size();

        std::string name = "Ip_size_"+std::to_string(depth);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);

        if(Ip.size() > 0){
            //write the parts

            name = "Ip_"+std::to_string(depth);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT16,name.c_str(),rank,&dims, Ip.data());

        }


    }



    hsize_t attr = depth_min;
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &pc_struct.depth_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );


    // output the file size
    hsize_t file_size;
    H5Fget_filesize(fid, &file_size);

    std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;

    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);

    std::cout << "Writing Complete" << std::endl;




}



template<typename T,typename U>
void write_apr_wavelet(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name,float comp_factor){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the APR to the particle cell structure using Haar wavelet compression
    //
    //
    
    
    
    std::cout << "Writing parts to hdf5 file, in pc_struct format..." << std::endl;
    
    
    uint64_t num_cells = pc_struct.get_number_cells();
    uint64_t num_parts = pc_struct.get_number_parts();
    
    //initialize
    uint64_t node_val_part;
    uint64_t y_coord;
    int x_;
    int z_;
    
    uint64_t j_;
    uint64_t status;
    uint64_t curr_key=0;
    uint64_t part_offset=0;
    
    
    //Neighbour Routine Checking
    
    uint64_t p;
    
    register_bosc();
    
    std::string hdf5_file_name = save_loc + file_name + "_pcstruct_part_wavelet.h5";
    
    file_name = file_name + "_pcstruct_part_wavelet";
    
    hdf5_create_file(hdf5_file_name);
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id;
    H5G_info_t info;
    
    hsize_t     dims_out[2];
    
    hsize_t rank = 1;
    
    hsize_t dims;
    hsize_t dim_a=1;
    
    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);
    
    //Get the group you want to open
    
    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;
    
    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &pc_struct.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &pc_struct.org_dims[0] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &pc_struct.org_dims[2] );
    
    
    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    
    dims_out[0] = 1;
    dims_out[1] = 1;
    
    //just an identifier in here for the reading of the parts
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );
    
    
    //////////////////////////////////////////////////////////////////
    //
    //  Write data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    
    
    //////////////////////////////////////////////////
    //
    //
    //
    //  Get wavelet coefficients
    //
    //
    /////////////////////////////////////////////////////
    
    
    ExtraPartCellData<U> q; //particle size
    
    ExtraPartCellData<uint8_t> scale; //cell size
    
    ExtraPartCellData<uint8_t> scale_parent; //parent size
    ExtraPartCellData<T> mu_parent; //parent size
    ExtraPartCellData<U> q_parent; // parent size
    
    calc_wavelet_encode(pc_struct,scale,q,scale_parent,mu_parent,q_parent,comp_factor);
    

    
    
    uint64_t depth_min = pc_struct.depth_min;
    std::vector<uint8_t> p_map;
    
    //Initialize the output variables
    
    //particle loop
    std::vector<U> q_out;
    
    //cell loop
    std::vector<uint8_t> scale_out;
    
    //parent loop
    std::vector<uint8_t> scale_parent_out;
    std::vector<uint16_t> mu_parent_out;
    std::vector<U> q_parent_out;
    
    // output the file size
    hsize_t file_size;
    
    hsize_t file_size_prev = 0;
    
    
    std::string name;
        
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        
        ///////////////////////////////
        //
        // parent loop
        //
        //////////////////////////////
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];

        
        
        
        if ( i <= (pc_struct.depth_max - 1)){
            
            //scale_out loop
            scale_parent_out.resize(0);
            mu_parent_out.resize(0);
            q_parent_out.resize(0);
            
            for(z_ = 0;z_ < z_num_;z_++){
                
                curr_key = 0;
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    const size_t offset_pc_data = x_num_*z_ + x_;
                    
                    const size_t j_num = scale_parent.data[i][offset_pc_data].size();
                    
                    uint64_t curr_size = scale_parent_out.size();
                    
                    scale_parent_out.resize(curr_size+ j_num);
                    mu_parent_out.resize(curr_size+ j_num);
                    q_parent_out.resize(curr_size+ j_num);
                    
                    std::copy(scale_parent.data[i][offset_pc_data].begin(),scale_parent.data[i][offset_pc_data].end(),scale_parent_out.begin() + curr_size);
                    std::copy(mu_parent.data[i][offset_pc_data].begin(),mu_parent.data[i][offset_pc_data].end(),mu_parent_out.begin() + curr_size);
                    std::copy(q_parent.data[i][offset_pc_data].begin(),q_parent.data[i][offset_pc_data].end(),q_parent_out.begin() + curr_size);
                    
                }
                
            }
            
        }

        
        //parent data
        if ( i <= (pc_struct.depth_max - 1)){
            
            dims = scale_parent_out.size();
            
            name = "parent_size_"+std::to_string(i);
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);
            
            if(scale_parent_out.size() > 0){
            
                name = "scale_parent_"+std::to_string(i);
                hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, scale_parent_out.data());
            
                name = "mu_parent_"+std::to_string(i);
                hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT16,name.c_str(),rank,&dims, mu_parent_out.data());
            
                name = "q_parent_"+std::to_string(i);
                hdf5_write_data_blosc(obj_id,H5T_NATIVE_INT8,name.c_str(),rank,&dims, q_parent_out.data());
            }
            
        }
    }
        
        
    H5Fget_filesize(fid, &file_size);
        
    std::cout << "Parent Filesize: " << (file_size-file_size_prev)*1.0/1000000.0 << " MB" << std::endl;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        /////////////////////////////////
        //
        // particle loop
        //
        ///////////////////////////////////
        
        
        //q_out loop
        q_out.resize(0);
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.part_data.particle_data.data[i][offset_pc_data].size();
                
                uint64_t curr_size = q_out.size();
                q_out.resize(curr_size+ j_num);
                
                std::copy(q.data[i][offset_pc_data].begin(),q.data[i][offset_pc_data].end(),q_out.begin() + curr_size);
                
            }
            
        }
        
        
        dims = q_out.size();
        name = "part_size_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);
        
        
        
        if(q_out.size() > 0){
            
            /////////////////////
            //
            //  Perform writing to disk
            //
            ////////////////////
            
            //part data
            
            //write the q
            dims = q_out.size();
            name = "q_"+std::to_string(i);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_INT8,name.c_str(),rank,&dims, q_out.data(),BLOSC_ZSTD,6,0);
            
        }
        
    }
    
    file_size_prev = file_size;
    H5Fget_filesize(fid, &file_size);
        
    std::cout << "Q Filesize: " << (file_size-file_size_prev)*1.0/1000000.0 << " MB" << std::endl;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        /////////////////////////////////
        //
        // cell loop
        //
        ///////////////////////////////////
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];

        
        
        //scale_out loop
        scale_out.resize(0);
        
        for(z_ = 0;z_ < z_num_;z_++){
            
            curr_key = 0;
            
            for(x_ = 0;x_ < x_num_;x_++){
                
                const size_t offset_pc_data = x_num_*z_ + x_;
                
                const size_t j_num = pc_struct.pc_data.data[i][offset_pc_data].size();
                
                uint64_t curr_size = scale_out.size();
                scale_out.resize(curr_size+ j_num);
                
                std::copy(scale.data[i][offset_pc_data].begin(),scale.data[i][offset_pc_data].end(),scale_out.begin() + curr_size);
                
            }
            
        }
        
        dims = scale_out.size();
        name = "cell_size_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);
        
        if(scale_out.size() > 0){
            //cell data
            dims = scale_out.size();
            name = "scale_"+std::to_string(i);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, scale_out.data());
        }

    }
    
    file_size_prev = file_size;
    H5Fget_filesize(fid, &file_size);
        
    std::cout << "Scale_full Filesize: " << (file_size-file_size_prev)*1.0/1000000.0 << " MB" << std::endl;
        
        
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
    
        //////////////////////////
        //
        //  Structure Data
        //
        //////////////////////////////
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        
        p_map.resize(x_num_*z_num_*y_num_,0);
        
        std::fill(p_map.begin(), p_map.end(), 0);
        
        //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.
        
        // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,p,node_val_part,curr_key,part_offset,status,y_coord) if(z_num_*x_num_ > 100)
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
                        p_map[offset_p_map + y_coord] = status;
                        
                    } else {
                        
                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }
                    
                }
                
            }
            
        }
        
        dims = p_map.size();
        name = "p_map_"+std::to_string(i);
        hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, p_map.data());
        
        name = "p_map_x_num_"+std::to_string(i);
        hsize_t attr = x_num_;
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
        
        attr = y_num_;
        name = "p_map_y_num_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
        
        attr = z_num_;
        name = "p_map_z_num_"+std::to_string(i);
        hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);
        
    }
    
    file_size_prev = file_size;
    H5Fget_filesize(fid, &file_size);
    
    std::cout << "P_map Filesize: " << (file_size-file_size_prev)*1.0/1000000.0 << " MB" << std::endl;
    
    hsize_t attr = depth_min;
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &pc_struct.depth_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );
    
    
    // output the file size
    H5Fget_filesize(fid, &file_size);
    
    std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    std::cout << "Writing Complete" << std::endl;
    
    
    
    
}

template<typename T,typename U>
void read_apr_wavelet(PartCellStructure<T,uint64_t>& pc_struct,std::string file_name)
{
    
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id;
    H5G_info_t info;
    
    register_bosc();
    
    int num_parts,num_cells;
    
    fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    
    //Get the group you want to open
    
    pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    //Getting an attribute
    obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);
    
    //Load the attributes

    
    /////////////////////////////////////////////
    //  Get metadata
    //
    //////////////////////////////////////////////
    
    float comp_factor = 40; //will have to read this in
    
    
    attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[0]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[1]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[2]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"depth_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.depth_max) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"depth_min",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.depth_min) ;
    H5Aclose(attr_id);
    
    
    std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;
    
    
    //////////////////////////////
    //
    //  Initialize output vectors
    //
    ////////////////////////////
    
    std::vector<std::vector<uint8_t>> p_map_load;
    
    //particle loop
    std::vector<std::vector<U>> q_out;
    
    //cell loop
    std::vector<std::vector<uint8_t>> scale_out;
    
    //parent loop
    std::vector<std::vector<uint8_t>> scale_parent_out;
    std::vector<std::vector<uint16_t>> mu_parent_out;
    std::vector<std::vector<U>> q_parent_out;
    
    
    p_map_load.resize(pc_struct.depth_max+1);
    q_out.resize(pc_struct.depth_max+1);
    scale_out.resize(pc_struct.depth_max+1);
    
    //parent level have one less resolution level
    scale_parent_out.resize(pc_struct.depth_max);
    mu_parent_out.resize(pc_struct.depth_max);
    q_parent_out.resize(pc_struct.depth_max);
    
    std::string name;
    
    pc_struct.x_num.resize(pc_struct.depth_max+1);
    pc_struct.z_num.resize(pc_struct.depth_max+1);
    pc_struct.y_num.resize(pc_struct.depth_max+1);
    
    // pc loops
    
    for(int i = pc_struct.depth_min;i <= pc_struct.depth_max; i++){
        
        //get the info
        
        int x_num;
        name = "p_map_x_num_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&x_num) ;
        H5Aclose(attr_id);
        
        int y_num;
        name = "p_map_y_num_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&y_num) ;
        H5Aclose(attr_id);
        
        int z_num;
        name = "p_map_z_num_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&z_num) ;
        H5Aclose(attr_id);
        
        int q_num;
        name = "part_size_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&q_num);
        H5Aclose(attr_id);
        
        int scale_num;
        name = "cell_size_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&scale_num);
        H5Aclose(attr_id);
        
        //initialize
        p_map_load[i].resize(x_num*y_num*z_num);
        q_out[i].resize(q_num);
        scale_out[i].resize(scale_num);
        
        if(p_map_load[i].size()>0){
            name = "p_map_"+std::to_string(i);
            //Load the data then update the particle dataset
            hdf5_load_data(obj_id,H5T_NATIVE_UINT8,p_map_load[i].data(),name.c_str());
        }
        
        if(q_out[i].size()>0){
            name = "q_"+std::to_string(i);
            hdf5_load_data(obj_id,H5T_NATIVE_INT8,q_out[i].data(),name.c_str());
        }
        
        if(scale_out[i].size()>0){
            name = "scale_"+std::to_string(i);
            hdf5_load_data(obj_id,H5T_NATIVE_UINT8,scale_out[i].data(),name.c_str());
        }
            
        pc_struct.x_num[i] = x_num;
        pc_struct.y_num[i] = y_num;
        pc_struct.z_num[i] = z_num;
        
    }
    
    //parent loops
    
    for(int i = pc_struct.depth_min;i < pc_struct.depth_max; i++){
        
        //get the info
    
        int parent_num;
        name = "parent_size_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&parent_num);
        H5Aclose(attr_id);
        
        //initialize
        q_parent_out[i].resize(parent_num);
        mu_parent_out[i].resize(parent_num);
        scale_parent_out[i].resize(parent_num);
        
        if(parent_num > 0){
            
            name = "q_parent_"+std::to_string(i);
            hdf5_load_data(obj_id,H5T_NATIVE_INT8,q_parent_out[i].data(),name.c_str());
            
            name = "scale_parent_"+std::to_string(i);
            hdf5_load_data(obj_id,H5T_NATIVE_UINT8,scale_parent_out[i].data(),name.c_str());
            
            name = "mu_parent_"+std::to_string(i);
            hdf5_load_data(obj_id,H5T_NATIVE_UINT16,mu_parent_out[i].data(),name.c_str());
        }
    }
    
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    pc_struct.initialize_pc_read(p_map_load);
    
    calc_wavelet_decode(pc_struct,scale_out,q_out,scale_parent_out,mu_parent_out,q_parent_out,comp_factor);
    
}

template<typename T>
void read_apr_pc_struct(PartCellStructure<T,uint64_t>& pc_struct,std::string file_name)
{
    
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id;
    H5G_info_t info;
    
    //need to register the filters so they work properly
    register_bosc();
    
    int num_parts,num_cells;
    
    fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    
    //Get the group you want to open
    
    pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    //Getting an attribute
    obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);
    
    //Load the attributes
    
    
    /////////////////////////////////////////////
    //  Get metadata
    //
    //////////////////////////////////////////////

    hid_t       aid, atype, attr;

    aid = H5Screate(H5S_SCALAR);

    pc_struct.name.reserve(100);

    //std::string string_out;

    //std::vector<char> string_out;
    //string_out.resize(80);

    //atype = H5Tcopy (H5T_C_S1);

    char string_out[100];

    for (int j = 0; j < 100; ++j) {
        string_out[j] = 0;
    }

    attr_id = 	H5Aopen(pr_groupid,"name",H5P_DEFAULT);

    atype = H5Aget_type(attr_id);

    hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);

    H5Aread(attr_id,atype_mem,string_out) ;
    H5Aclose(attr_id);

    pc_struct.name= string_out;
    pc_struct.pars.name = string_out;

    attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[0]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[1]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[2]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"depth_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.depth_max) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"depth_min",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.depth_min) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"lambda",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.lambda ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"var_th",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.var_th ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"var_th_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.var_th_max ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"I_th",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.I_th ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"dx",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.dx ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"dy",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.dy ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"dz",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.dz ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"psfx",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.psfx ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"psfy",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.psfy ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"psfz",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.psfz ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"rel_error",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.rel_error ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"aniso",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.aniso ) ;
    H5Aclose(attr_id);

    std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;
    
    std::vector<std::vector<uint8_t>> p_map_load;
    std::vector<std::vector<uint16_t>> Ip;
    
    p_map_load.resize(pc_struct.depth_max+1);
    Ip.resize(pc_struct.depth_max+1);
    
    std::string name;
    
    pc_struct.x_num.resize(pc_struct.depth_max+1);
    pc_struct.z_num.resize(pc_struct.depth_max+1);
    pc_struct.y_num.resize(pc_struct.depth_max+1);
    
    for(int i = pc_struct.depth_min;i <= pc_struct.depth_max; i++){
        
        //get the info
        
        int x_num;
        name = "p_map_x_num_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&x_num) ;
        H5Aclose(attr_id);
        
        int y_num;
        name = "p_map_y_num_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&y_num) ;
        H5Aclose(attr_id);
        
        int z_num;
        name = "p_map_z_num_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&z_num) ;
        H5Aclose(attr_id);
        
        int Ip_num;
        name = "Ip_size_"+std::to_string(i);
        
        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&Ip_num);
        H5Aclose(attr_id);
        
        p_map_load[i].resize(x_num*y_num*z_num);
        Ip[i].resize(Ip_num);
        
        if(p_map_load[i].size() > 0){
            name = "p_map_"+std::to_string(i);
            //Load the data then update the particle dataset
            hdf5_load_data(obj_id,H5T_NATIVE_UINT8,p_map_load[i].data(),name.c_str());
        }
        
        if(Ip[i].size()>0){
            name = "Ip_"+std::to_string(i);
            hdf5_load_data(obj_id,H5T_NATIVE_UINT16,Ip[i].data(),name.c_str());
        }
        
        pc_struct.x_num[i] = x_num;
        pc_struct.y_num[i] = y_num;
        pc_struct.z_num[i] = z_num;
        
    }
    
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    pc_struct.initialize_structure_read(p_map_load,Ip);
    
    
}

template<typename T>
void read_write_apr_pc_struct(PartCellStructure<T,uint64_t> pc_struct,std::string file_name)
{


    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id;
    H5G_info_t info;

    //need to register the filters so they work properly
    register_bosc();

    int num_parts,num_cells;

    fid = H5Fopen(file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //Get the group you want to open

    pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );

    //Getting an attribute
    obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);

    //Load the attributes


    /////////////////////////////////////////////
    //  Get metadata
    //
    //////////////////////////////////////////////

    hid_t       aid, atype, attr;
    hsize_t dims;

    aid = H5Screate(H5S_SCALAR);

    pc_struct.name.reserve(100);

    //std::string string_out;

    //std::vector<char> string_out;
    //string_out.resize(80);

    //atype = H5Tcopy (H5T_C_S1);

    char string_out[100];

    for (int j = 0; j < 100; ++j) {
        string_out[j] = 0;
    }

    attr_id = 	H5Aopen(pr_groupid,"name",H5P_DEFAULT);

    atype = H5Aget_type(attr_id);

    hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);

    H5Aread(attr_id,atype_mem,string_out) ;
    H5Aclose(attr_id);

    pc_struct.name= string_out;
    pc_struct.pars.name = string_out;

    attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[0]) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[1]) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.org_dims[2]) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"depth_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.depth_max) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"depth_min",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&pc_struct.depth_min) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"lambda",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.lambda ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"var_th",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.var_th ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"var_th_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.var_th_max ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"I_th",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.I_th ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"dx",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.dx ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"dy",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.dy ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"dz",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.dz ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"psfx",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.psfx ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"psfy",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.psfy ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"psfz",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.psfz ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"rel_error",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.rel_error ) ;
    H5Aclose(attr_id);

    attr_id = 	H5Aopen(pr_groupid,"aniso",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&pc_struct.pars.aniso ) ;
    H5Aclose(attr_id);

    std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;

    std::vector<std::vector<uint8_t>> p_map_load;
    std::vector<std::vector<uint16_t>> Ip;

    p_map_load.resize(pc_struct.depth_max+2);
    Ip.resize(pc_struct.depth_max+2);

    std::string name;



    hsize_t rank = 1;

    for(int i = pc_struct.depth_min;i <= pc_struct.depth_max+1; i++){

        //get the info



        int Ip_num;
        name = "Ip_size_"+std::to_string(i);

        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&Ip_num);
        H5Aclose(attr_id);


        Ip[i].resize(Ip_num);

        if(Ip[i].size()>0){
            name = "Ip_"+std::to_string(i);
            dims = Ip_num;
            hdf5_load_data(obj_id,H5T_NATIVE_UINT16,Ip[i].data(),name.c_str());

            hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT16,name.c_str(),rank,&dims, Ip[i].data());
        }

    }


    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);




}
#endif