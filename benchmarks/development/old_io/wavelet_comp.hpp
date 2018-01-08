//
// Created by cheesema on 17.10.17.
//

#ifndef PARTPLAY_WAVELET_COMP_HPP
#define PARTPLAY_WAVELET_COMP_HPP

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

#include "external/blitzwave/src/Wavelet.h"
#include "external/blitzwave/src/WaveletDecomp.h"
#include "external/blitzwave/src/arrayTools.h"

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

    double norm_f = 0;

    for (int ix=0; ix<(idx.rows()-2); ++ix) {
        TinyVector<int,1> index = idx(ix);
        Array<numtype, 1> coeffs = decomp_t.coeffs(sig, index);

        double thresh = Th / decomp_t.normFactor(index);
        //double thresh = Th;
        numtype ithresh = numtype(thresh);

        norm_f = decomp_t.normFactor(index);

       // std::cout << coeffs << std::endl;

        for (int i=0; i<coeffs.rows(); ++i)
            coeffs(i) = abs(coeffs(i)) > ithresh ? coeffs(i) : 0;
    }

    sig = floor(sig);

    std::cout << 1.0*sum(sig==0)/input.size() << std::endl;

    std::cout << max(sig/norm_f) << std::endl;

    //sig = floor(sig);

   // std::cout << sig << std::endl;

    Array<numtype, 1> sig_d(sig.shape());

    for (int i=0; i<idx.rows(); ++i)
        decomp2.coeffs(sig_d, idx(i)) = decomp_t.coeffs(sig, idx(i));

    sig = 0;

    for (int i=0; i<idx.rows(); ++i)
          decomp_t.coeffs(sig, idx(i)) = decomp2.coeffs(sig_d, idx(i));


//    if(input.size() < 700) {
//
//        std::cout << sig << std::endl;
//
//    }

    decomp_t.applyInv(sig);

    std::copy(sig.data(),sig.data() + input.size(),input.begin());


    //std::cout << sig_d << std::endl;


}


template<typename T,typename S>
void get_wavelet_coeffs(std::vector<S>& output,std::vector<T>& input,float Th,int max_l,Wavelet wl){

    typedef float numtype;

    std::vector<numtype> temp;

    temp.resize(input.size());
    std::copy(input.begin(),input.end(),temp.begin());

    Array<numtype, 1> sig(temp.data(),shape(temp.size()),duplicateData);

    WaveletDecomp<1> decomp_t(wl, NONSTD_DECOMP, max_l);

    decomp_t.apply(sig);

    WaveletDecomp<1> decomp2(decomp_t, SEPARATED_COEFFS);

    Array<TinyVector<int,1>, 1> idx( decomp2.indices(sig) );

    for (int ix=0; ix<(idx.rows()-2); ++ix) {
        TinyVector<int,1> index = idx(ix);
        Array<numtype, 1> coeffs = decomp_t.coeffs(sig, index);

        double thresh = Th / decomp_t.normFactor(index);
        numtype ithresh = numtype(thresh);

        for (int i=0; i<coeffs.rows(); ++i)
            coeffs(i) = abs(coeffs(i)) > ithresh ? coeffs(i) : 0;
    }

    //std::cout << max(abs(sig)) << std::endl;

    sig = floor(sig);

    std::cout << 1.0*sum(sig==0)/input.size() << std::endl;

    Array<numtype, 1> sig_d(sig.shape());

    for (int i=0; i<idx.rows(); ++i) {
        decomp2.coeffs(sig_d, idx(i)) = decomp_t.coeffs(sig, idx(i));
    }

    std::cout << 1.0*sum(sig_d==0)/input.size() << std::endl;

    output.resize(input.size());

    std::copy(sig_d.data(),sig_d.data() + input.size(),output.begin());


}

template<typename T,typename S>
void get_recon(std::vector<S>& output,std::vector<T>& input,int max_l,Wavelet wl){

    typedef float numtype;

    std::vector<numtype> temp;

    temp.resize(input.size());
    std::copy(input.begin(),input.end(),temp.begin());

    Array<numtype, 1> sig(temp.data(),shape(temp.size()),duplicateData);

    WaveletDecomp<1> decomp_t(wl, NONSTD_DECOMP, max_l);

    WaveletDecomp<1> decomp2(decomp_t, SEPARATED_COEFFS);

    Array<TinyVector<int,1>, 1> idx( decomp2.indices(sig) );

    Array<numtype, 1> sig_d(sig.shape());

    for (int i=0; i<idx.rows(); ++i) {
        decomp_t.coeffs(sig_d, idx(i)) = decomp2.coeffs(sig, idx(i));
    }

    //std::cout << sig_d << std::endl;

    decomp_t.applyInv(sig_d);

    output.resize(input.size());

    std::copy(sig_d.data(),sig_d.data() + input.size(),output.begin());

    int stop = 1;


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

        if(Ip[i].size() > 0) {

            float th_l = th / pow(2, (pc_struct.pc_data.depth_max - i) * 4);

            if ((pc_struct.pc_data.depth_max - i) > 0) {
                th_l = 0;
            }

            int num_levels = floor(log2(Ip[i].size()));
            num_levels = max(2, num_levels - 4);

            wavelet_th_return(Ip[i], th_l, num_levels, wl);

        }

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

            int num_levels = floor(log2(Ip.size()));
            num_levels = max(2,num_levels-4);

            float th_l = th/pow(2,(pc_struct.pc_data.depth_max-i)*4);

            if((pc_struct.pc_data.depth_max - i) > 0){
                th_l = 0;
            }

            std::vector<float> coeffs;
            //std::vector<int> coeffs;
            get_wavelet_coeffs(coeffs,Ip,th_l,num_levels,wl);

            dims = coeffs.size();

            std::string name = "wc_size_"+std::to_string(i);
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);

            //write the parts
            name = "wc_"+std::to_string(i);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_FLOAT,name.c_str(),rank,&dims, coeffs.data());

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
void read_apr_wavelet(PartCellStructure<T,uint64_t>& pc_struct,std::string file_name,Wavelet wl)
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
        name = "wc_size_"+std::to_string(i);

        attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&Ip_num);
        H5Aclose(attr_id);

        p_map_load[i].resize(x_num*y_num*z_num);
        Ip[i].resize(Ip_num);
        std::vector<float> temp;
        temp.resize(Ip_num);

        if(p_map_load[i].size() > 0){
            name = "p_map_"+std::to_string(i);
            //Load the data then update the particle dataset
            hdf5_load_data(obj_id,H5T_NATIVE_UINT8,p_map_load[i].data(),name.c_str());
        }

        if(Ip[i].size()>0){
            name = "wc_"+std::to_string(i);
            hdf5_load_data(obj_id,H5T_NATIVE_FLOAT,temp.data(),name.c_str());

            //read back the wavelet coeffficients in here
            int num_levels = floor(log2(Ip[i].size()));
            num_levels = max(2,num_levels-4);

            get_recon(Ip[i],temp,num_levels,wl);

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
void re_order_local(APR<float>& curr_apr,std::vector<uint16_t>& Ip,int depth){
//
    //  Bevan Cheeseman 2017
    //
    //  Re-order local
    //

    const unsigned int x_num_ = curr_apr.y_vec.x_num[depth];
    const unsigned int z_num_ = curr_apr.y_vec.z_num[depth];

    std::vector<std::vector<uint16_t>> groups;

    //groups.reserve(200);
    groups.resize(2);

    for (int z_ = 0; z_ < z_num_; z_++) {
        //both z and x are explicitly accessed in the structure

        for (int x_ = 0; x_ < x_num_; x_++) {

            const unsigned int pc_offset = x_num_*z_ + x_;

            const size_t j_num = curr_apr.particles_int.data[depth][pc_offset].size();

            int prev = -1;
            int counter = 0;

            for (int i = 0; i < j_num; ++i) {
                if((curr_apr.y_vec.data[depth][pc_offset][i]-prev) == 1){
                    groups[counter].push_back(curr_apr.particles_int.data[depth][pc_offset][i]);
                } else {
                    counter++;
                    if(counter >= groups.size()) {
                        groups.resize(counter + 1);
                    }
                    groups[counter].push_back(curr_apr.particles_int.data[depth][pc_offset][i]);
                }
                prev = curr_apr.y_vec.data[depth][pc_offset][i];
            }


        }

    }

    //flatten into one array
    uint64_t size = 0;
    for (int j = 0; j < groups.size(); ++j) {
        size += groups[j].size();
    }

    Ip.resize(size);

    size = 0;
    for (int j = 0; j < groups.size(); ++j) {

        std::copy(groups[j].begin(),groups[j].end(),Ip.begin() + size);
        size += groups[j].size();
    }

}
void undo_re_order_local(APR<float>& curr_apr,std::vector<uint16_t>& Ip,int depth){
//
    //  Bevan Cheeseman 2017
    //
    //  Re-order local
    //

    const unsigned int x_num_ = curr_apr.y_vec.x_num[depth];
    const unsigned int z_num_ = curr_apr.y_vec.z_num[depth];

    std::vector<std::vector<uint16_t>> groups;

    //groups.reserve(200);
    groups.resize(2);

    std::vector<uint64_t> group_nums;
    group_nums.resize(1,0);

    for (int z_ = 0; z_ < z_num_; z_++) {
        //both z and x are explicitly accessed in the structure

        for (int x_ = 0; x_ < x_num_; x_++) {

            const unsigned int pc_offset = x_num_*z_ + x_;

            const size_t j_num = curr_apr.particles_int.data[depth][pc_offset].size();

            int prev = -1;
            int counter = 0;

            for (int i = 0; i < j_num; ++i) {
                if((curr_apr.y_vec.data[depth][pc_offset][i]-prev) == 1){
                    group_nums[counter]++;
                } else {
                    counter++;
                    if(counter >= group_nums.size()) {
                        group_nums.resize(counter + 1);
                    }
                    group_nums[counter]++;
                }
                prev = curr_apr.y_vec.data[depth][pc_offset][i];
            }


        }

    }

    //flatten into one array
    uint64_t size = 0;

    groups.resize(group_nums.size());

    size = 0;
    for (int j = 0; j < groups.size(); ++j) {
        groups[j].resize(group_nums[j]);
        std::copy(Ip.begin() + size,Ip.begin() + group_nums[j] + size,groups[j].begin());
        size += groups[j].size();
    }

    uint64_t ip_c=0;

    std::vector<uint64_t> group_nums2;
    group_nums2.resize(1,0);

    // place back
    for (int z_ = 0; z_ < z_num_; z_++) {
        //both z and x are explicitly accessed in the structure

        for (int x_ = 0; x_ < x_num_; x_++) {

            const unsigned int pc_offset = x_num_*z_ + x_;

            const size_t j_num = curr_apr.particles_int.data[depth][pc_offset].size();

            int prev = -1;
            int counter = 0;

            for (int i = 0; i < j_num; ++i) {
                if((curr_apr.y_vec.data[depth][pc_offset][i]-prev) == 1){

                    Ip[ip_c] = groups[counter][group_nums2[counter]];
                    group_nums2[counter]++;
                    ip_c++;

                } else {
                    counter++;
                    if(counter >= group_nums2.size()) {
                        group_nums2.resize(counter + 1);
                        group_nums2[counter] = 0;
                    }
                    Ip[ip_c] = groups[counter][group_nums2[counter]];
                    group_nums2[counter]++;
                    ip_c++;
                }
                prev = curr_apr.y_vec.data[depth][pc_offset][i];
            }


        }

    }



}


template<typename T>
void write_apr_wavelet_partnew(PartCellStructure<T,uint64_t>& pc_struct,std::string save_loc,std::string file_name,float th,Wavelet wl){
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



    for(uint64_t depth = (curr_apr.particles_int.depth_min);depth <= curr_apr.particles_int.depth_max;depth++) {
        //loop over the resolutions of the structure
        const unsigned int x_num_ = curr_apr.y_vec.x_num[depth];
        const unsigned int z_num_ = curr_apr.y_vec.z_num[depth];
        unsigned int y_num_ = 0;
        if(depth == curr_apr.particles_int.depth_max) {
            y_num_ = pc_struct.org_dims[0];
        } else{
            y_num_ = pc_struct.y_num[depth];
        }
        const unsigned int x_num_min_ = 0;
        const unsigned int z_num_min_ = 0;

        Ip.resize(0);

        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                const size_t j_num = curr_apr.particles_int.data[depth][pc_offset].size();

                uint64_t curr_size = Ip.size();
                Ip.resize(curr_size+ j_num);

                std::copy(curr_apr.particles_int.data[depth][pc_offset].begin(),curr_apr.particles_int.data[depth][pc_offset].end(),Ip.begin() + curr_size);

            }
        }


        dims = Ip.size();
        std::vector<float> coeffs;

        if(Ip.size() > 0){
            //write the parts

            re_order_local(curr_apr,Ip,depth);

            int num_levels = floor(log2(Ip.size()));
            num_levels = max(2,num_levels-4);

            float th_l = th/pow(2,(curr_apr.particles_int.depth_max-depth)*3);

            if((curr_apr.particles_int.depth_max - depth) > 1){
                th_l = 0;
            }


            //std::vector<int> coeffs;
            get_wavelet_coeffs(coeffs,Ip,th_l,num_levels,wl);

            dims = coeffs.size();

            std::string name = "wc_size_"+std::to_string(depth);
            hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);

            //write the parts
            name = "wc_"+std::to_string(depth);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_FLOAT,name.c_str(),rank,&dims, coeffs.data());


        }

        if(Ip.size() > 0){

            //read back the wavelet coeffficients in here
            int num_levels = floor(log2(Ip.size()));
            num_levels = max(2,num_levels-4);

            get_recon(Ip,coeffs,num_levels,wl);

            undo_re_order_local(curr_apr,Ip,depth);

        }

        uint64_t curr_size = 0;

        for (z_ = z_num_min_; z_ < z_num_; z_++) {
            //both z and x are explicitly accessed in the structure

            for (x_ = x_num_min_; x_ < x_num_; x_++) {

                const unsigned int pc_offset = x_num_*z_ + x_;

                const size_t j_num = curr_apr.particles_int.data[depth][pc_offset].size();

                std::copy(Ip.begin() + curr_size,Ip.begin() + curr_size + j_num,curr_apr.particles_int.data[depth][pc_offset].begin());

                curr_size += j_num;

            }
        }



    }


    Mesh_data<uint16_t> interp;

    interp_img(interp, curr_apr.y_vec, curr_apr.particles_int);

    debug_write(interp,"alt_interp");


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
