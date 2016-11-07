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

#include "hdf5functions.h"
#include "../data_structures/Tree/PartCellStructure.hpp"
#include "write_parts.h"


template<typename T>
void write_apr_full_format(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name);

template<typename T>
void write_apr_pc_struct(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name);

template<typename T>
void read_apr_pc_struct(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name);

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
    
    
    //Neighbour Routine Checking
    
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
void write_apr_pc_struct(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
    //
    //
    
    
    
    std::cout << "Writing parts to hdf5 file, in pc_struct format..." << std::endl;
    
    
    std::vector<uint16_t> Ip;
    
    
    int num_cells = pc_struct.get_number_cells();
    int num_parts = pc_struct.get_number_parts();
    
 
    
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
    
    std::vector<uint8_t> p_map;
    
    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){
        
        
        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];
        
        p_map.resize(x_num_*z_num_*y_num_,0);
        
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
        
        
        //write the vals
        
        dims = p_map.size();
        std::string name = "p_map_"+std::to_string(i);
        hdf5_write_data(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, p_map.data());
        
        
        std::vector<uint16_t> Ip;
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
            name = "Ip_"+std::to_string(i);
            hdf5_write_data(obj_id,H5T_NATIVE_UINT16,name.c_str(),rank,&dims, Ip.data());
        }
    }

    
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



#endif