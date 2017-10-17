//
// Created by cheesema on 17.10.17.
//

#ifndef PARTPLAY_WAVELET_COMP_HPP
#define PARTPLAY_WAVELET_COMP_HPP

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "hdf5functions.h"
#include "hdf5functions_blosc.h"
#include "../data_structures/Tree/PartCellStructure.hpp"
#include "write_parts.h"
#include "writeimage.h"
#include "../numerics/apr_compression.hpp"
#include "../../src/data_structures/APR/APR.hpp"

#include "../../external/blitzwave/src/Wavelet.h"
#include "../../external/blitzwave/src/WaveletDecomp.h"
#include "../../external/blitzwave/src/arrayTools.h"

using namespace blitz;
using namespace bwave;

template<typename T>
void wavelet_th_return(std::vector<T>& input,float Th,int mlevel,Wavelet wl){

    typedef float numtype;

    std::vector<numtype> temp;
    temp.resize(input.size());
    std::copy(input.begin(),input.end(),temp.begin());

    Array<numtype, 1> sig(temp.data(),shape(input.size()),duplicateData);

    WaveletDecomp<1> decomp_t(wl, NONSTD_DECOMP, mlevel);

    WaveletDecomp<1> decomp2(decomp_t, SEPARATED_COEFFS);

    Array<TinyVector<int,1>, 1> idx( decomp2.indices(sig) );

    decomp_t.apply(sig);



    std::cout << idx << std::endl;

    for (int ix=0; ix<(idx.rows()-2); ++ix) {
        TinyVector<int,1> index = idx(ix);
        Array<numtype, 1> coeffs = decomp_t.coeffs(sig, index);

        double thresh = Th / decomp_t.normFactor(index);
        //double thresh = Th;
        numtype ithresh = numtype(thresh);

        double norm_f = decomp_t.normFactor(index);

       // std::cout << coeffs << std::endl;

        for (int i=0; i<coeffs.rows(); ++i)
            coeffs(i) = abs(coeffs(i)) > ithresh ? coeffs(i) : 0;
    }

    std::cout << 1.0*sum(sig==0)/input.size() << std::endl;

    decomp_t.applyInv(sig);

    std::copy(sig.data(),sig.data() + input.size(),input.begin());


}


template<typename T>
void get_wavelet_coeffs(std::vector<int>& output,std::vector<T>& input,float Th,int max_l,Wavelet wl){

    output.resize(input.size());
    std::copy(input.begin(),input.end(),output.begin());

    Array<int, 1> sig(output.data(),shape(output.size()),duplicateData);

    WaveletDecomp<1> decomp_t(wl, NONSTD_DECOMP, max_l);

    decomp_t.apply(sig);

    Array<TinyVector<int,1>, 1> idx( decomp_t.indices(sig) );

    typedef int numtype;

    for (int ix=0; ix<idx.rows(); ++ix) {
        TinyVector<int,1> index = idx(ix);
        Array<numtype, 1> coeffs = decomp_t.coeffs(sig, index);

        double thresh = Th / decomp_t.normFactor(index);
        numtype ithresh = numtype(thresh);


        for (int i=0; i<coeffs.rows(); ++i)
            coeffs(i) = abs(coeffs(i)) > ithresh ? coeffs(i) : 0;
    }

    //std::cout << max(abs(sig)) << std::endl;

    //decomp_t.applyInv(sig);

    output.resize(input.size());

    std::copy(sig.data(),sig.data() + input.size(),output.begin());


}

template<typename T>
void test_wavelet(PartCellStructure<T,uint64_t>& pc_struct,float th,int mlevel,Wavelet wl){
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


    //////////////////////////////////////////////////////////////////
    //
    //  Write data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////

    uint64_t depth_min = pc_struct.depth_min;
    //std::vector<uint8_t> p_map;
    //std::vector<uint16_t> Ip;

    std::vector<std::vector<uint8_t>> p_map_load;
    std::vector<std::vector<uint16_t>> Ip;

    p_map_load.resize(pc_struct.depth_max+1);
    Ip.resize(pc_struct.depth_max+1);


    for(uint64_t i = pc_struct.pc_data.depth_min;i <= pc_struct.pc_data.depth_max;i++){


        const unsigned int x_num_ = pc_struct.x_num[i];
        const unsigned int z_num_ = pc_struct.z_num[i];
        const unsigned int y_num_ = pc_struct.y_num[i];


        //write the vals

        Ip[i].resize(0);

        for(z_ = 0;z_ < z_num_;z_++){

            curr_key = 0;

            for(x_ = 0;x_ < x_num_;x_++){

                pc_struct.part_data.access_data.pc_key_set_x(curr_key,x_);
                const size_t offset_pc_data = x_num_*z_ + x_;

                const size_t j_num = pc_struct.part_data.particle_data.data[i][offset_pc_data].size();

                uint64_t curr_size = Ip[i].size();
                Ip[i].resize(curr_size+ j_num);

                std::copy(pc_struct.part_data.particle_data.data[i][offset_pc_data].begin(),pc_struct.part_data.particle_data.data[i][offset_pc_data].end(),Ip[i].begin() + curr_size);

            }

        }

        float th_l = th/pow(2,(pc_struct.pc_data.depth_max-i)*3);

        int num_levels = floor(log2(Ip[i].size()));
        num_levels = max(2,num_levels-4);

        wavelet_th_return(Ip[i],th_l,num_levels,wl);

        p_map_load[i].resize(x_num_*z_num_*y_num_,0);

        std::fill(p_map_load[i].begin(), p_map_load[i].end(), 0);


        //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

        // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val_part,curr_key,part_offset,status,y_coord) if(z_num_*x_num_ > 100)
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
                        p_map_load[i][offset_p_map + y_coord] = status;

                    } else {

                        y_coord += ((node_val_part & COORD_DIFF_MASK_PARTICLE) >> COORD_DIFF_SHIFT_PARTICLE);
                        y_coord--;
                    }

                }

            }

        }


    }

    pc_struct.initialize_structure_read(p_map_load,Ip);


}

template<typename T>
void write_apr_wavelet(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name,float th,int mlevel,Wavelet wl){
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

    std::string git_hash = exec("git rev-parse HEAD");

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

        std::string name;

        if(Ip.size() > 0){

            float th_l = th/pow(2,(pc_struct.pc_data.depth_max-i)*3);

            std::vector<int> coeffs;
            get_wavelet_coeffs(coeffs,Ip,th_l,mlevel,wl);

            dims = coeffs.size();

            std::string name = "wc_size_"+std::to_string(i);
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);


            //write the parts

            name = "wc_"+std::to_string(i);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_INT32,name.c_str(),rank,&dims, coeffs.data());

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



#endif //PARTPLAY_WAVELET_COMP_HPP
