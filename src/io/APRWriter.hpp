//
// Created by cheesema on 14.01.18.
//

#ifndef PARTPLAY_APRWRITER_HPP
#define PARTPLAY_APRWRITER_HPP

#include "src/data_structures/APR/APR.hpp"


template<typename U>
class APR;

class APRWriter {
public:

    template<typename ImageType>
    void read_apr(APR<ImageType>& apr,std::string file_name)
    {

        //currently only supporting 16 bit compress
        APRCompress<ImageType> apr_compress;

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

        apr.name.reserve(100);

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

        apr.name= string_out;
        apr.pars.name = string_out;

        apr.pc_data.org_dims.resize(3);
        apr.pc_data.depth_max = 0;
        apr.pc_data.depth_min = 0;

        attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&apr.pc_data.org_dims[0]) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&apr.pc_data.org_dims[1]) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&apr.pc_data.org_dims[2]) ;
        H5Aclose(attr_id);

//        attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
//        H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
//        H5Aclose(attr_id);
//
//        attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
//        H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
//        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"depth_max",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&apr.pc_data.depth_max) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"depth_min",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&apr.pc_data.depth_min) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"lambda",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.lambda ) ;
        H5Aclose(attr_id);

        int compress_type;
        attr_id = 	H5Aopen(pr_groupid,"compress_type",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&compress_type ) ;
        H5Aclose(attr_id);

        float quantization_factor;
        attr_id = 	H5Aopen(pr_groupid,"quantization_factor",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&quantization_factor ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"sigma_th",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.sigma_th ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"sigma_th_max",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.sigma_th_max ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"I_th",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.Ip_th ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"dx",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.dx ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"dy",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.dy ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"dz",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.dz ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"psfx",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.psfx ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"psfy",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.psfy ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"psfz",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.psfz ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"rel_error",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.rel_error ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"background_intensity_estimate",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.background_intensity_estimate ) ;
        H5Aclose(attr_id);

        attr_id = 	H5Aopen(pr_groupid,"noise_sd_estimate",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_FLOAT,&apr.parameters.noise_sd_estimate ) ;
        H5Aclose(attr_id);

        //std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;

        std::vector<std::vector<uint8_t>> p_map_load;
        std::vector<std::vector<uint16_t>> Ip;

        apr.pc_data.depth_max = apr.pc_data.depth_max;

        p_map_load.resize(apr.pc_data.depth_max);
        Ip.resize(apr.pc_data.depth_max+1);

        std::string name;

        apr.pc_data.x_num.resize(apr.pc_data.depth_max+1);
        apr.pc_data.z_num.resize(apr.pc_data.depth_max+1);
        apr.pc_data.y_num.resize(apr.pc_data.depth_max+1);

        for(int i = apr.pc_data.depth_min;i < apr.pc_data.depth_max; i++){

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


            p_map_load[i].resize(x_num*y_num*z_num);


            if(p_map_load[i].size() > 0){
                name = "p_map_"+std::to_string(i);
                //Load the data then update the particle dataset
                hdf5_load_data_blosc(obj_id,H5T_NATIVE_UINT8,p_map_load[i].data(),name.c_str());
            }


            apr.pc_data.x_num[i] = x_num;
            apr.pc_data.y_num[i] = y_num;
            apr.pc_data.z_num[i] = z_num;

        }

        for(int i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max; i++){

            //get the info

            int Ip_num;
            name = "Ip_size_"+std::to_string(i);

            attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
            H5Aread(attr_id,H5T_NATIVE_INT,&Ip_num);
            H5Aclose(attr_id);


            Ip[i].resize(Ip_num);

            if(Ip[i].size()>0){
                name = "Ip_"+std::to_string(i);
                hdf5_load_data_blosc(obj_id,H5T_NATIVE_UINT16,Ip[i].data(),name.c_str());
            }

        }

        apr.pc_data.y_num[apr.pc_data.depth_max] = apr.pc_data.org_dims[0];
        apr.pc_data.x_num[apr.pc_data.depth_max] = apr.pc_data.org_dims[1];
        apr.pc_data.z_num[apr.pc_data.depth_max] = apr.pc_data.org_dims[2];


        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        //
        //
        //  Transfer back the intensities
        //
        //
        apr.create_partcell_structure(p_map_load);

        apr.particles_int.initialize_structure_cells(apr.pc_data);

        for (int depth = apr.depth_min(); depth <= apr.depth_max(); ++depth) {

            uint64_t counter = 0;

            for (apr.begin(depth); apr.end(depth) != 0 ; apr.it_forward(depth)) {

                apr.curr_level.get_val(apr.particles_int) = Ip[depth][counter];

                counter++;

            }
        }

        if(compress_type > 0){

            apr_compress.set_compression_type(compress_type);
            apr_compress.set_quantization_factor(quantization_factor);

            apr_compress.decompress(apr,apr.particles_int);

        }

    }

    template<typename ImageType>
    void write_apr(APR<ImageType>& apr,std::string save_loc,std::string file_name){
        //compress
        APRCompress<ImageType> apr_compressor;
        apr_compressor.set_compression_type(0);

        write_apr(apr,save_loc,file_name,apr_compressor);

    }

    template<typename ImageType>
    void write_apr(APR<ImageType>& apr,std::string save_loc,std::string file_name,APRCompress<ImageType>& apr_compressor,unsigned int blosc_comp_type = BLOSC_ZSTD,unsigned int blosc_comp_level = 6,unsigned int blosc_shuffle=1){
        //
        //
        //  Bevan Cheeseman 2018
        //
        //  Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
        //
        //

        int compress_type_num = apr_compressor.get_compression_type();
        float quantization_factor = apr_compressor.get_quantization_factor();

        APR_timer write_timer;

        write_timer.verbose_flag = true;

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

        std::string hdf5_file_name = save_loc + file_name + "_apr.h5";

        file_name = file_name + "_apr";

        hdf5_create_file_blosc(hdf5_file_name);

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

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &apr.pc_data.org_dims[1] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &apr.pc_data.org_dims[0] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &apr.pc_data.org_dims[2] );


        obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

        dims_out[0] = 1;
        dims_out[1] = 1;

        //just an identifier in here for the reading of the parts

        int num_parts = apr.num_parts_total;
        int num_cells = apr.num_elements_total;

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

        // New parameter and background data

        if(apr.pars.name.size() == 0){
            apr.pars.name = "no_name";
            apr.name = "no_name";
        }

        hdf5_write_string_blosc(pr_groupid,"name",apr.pars.name);

        std::string git_hash = exec_blosc("git rev-parse HEAD");

        hdf5_write_string_blosc(pr_groupid,"githash",git_hash);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"compress_type",1,dims_out, &compress_type_num);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"quantization_factor",1,dims_out, &quantization_factor);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"lambda",1,dims_out, &apr.parameters.lambda );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"sigma_th",1,dims_out, &apr.parameters.sigma_th );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"sigma_th_max",1,dims_out, &apr.parameters.sigma_th_max );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"I_th",1,dims_out, &apr.parameters.Ip_th );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"dx",1,dims_out, &apr.parameters.dx );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"dy",1,dims_out, &apr.parameters.dy );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"dz",1,dims_out, &apr.parameters.dz );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"psfx",1,dims_out, &apr.parameters.psfx );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"psfy",1,dims_out, &apr.parameters.psfy );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"psfz",1,dims_out, &apr.parameters.psfz );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"rel_error",1,dims_out, &apr.parameters.rel_error);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"noise_sd_estimate",1,dims_out, &apr.parameters.noise_sd_estimate);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_FLOAT,"background_intensity_estimate",1,dims_out, &apr.parameters.background_intensity_estimate);

        //////////////////////////////////////////////////////////////////
        //
        //  Write data to the file
        //
        //
        //
        ///////////////////////////////////////////////////////////////////////

        write_timer.start_timer("intensities");

        uint64_t depth_min = apr.pc_data.depth_min;

        ///
        //
        //  #FIX ME NEEDS TO BE GENERAL TYPE
        //
        std::vector<uint16_t> Ip;

        ExtraPartCellData<ImageType> temp_int;

        if(compress_type_num > 0){

            apr_compressor.compress(apr,temp_int);

        } else {

            temp_int.initialize_structure_cells(apr.pc_data);
            temp_int.data = apr.particles_int.data;

        }

        write_timer.stop_timer();

        write_timer.start_timer("shift");

        apr.shift_particles_from_cells(temp_int);

        write_timer.stop_timer();

        write_timer.start_timer("write int");

        std::string name;

        for(uint64_t i = apr.pc_data.depth_min;i <= apr.pc_data.depth_max;i++){

            const unsigned int x_num_ = apr.pc_data.x_num[i];
            const unsigned int z_num_ = apr.pc_data.z_num[i];
            const unsigned int y_num_ = apr.pc_data.y_num[i];

            //write the vals
            Ip.resize(0);

            uint64_t total_p = 0;

            for(z_ = 0;z_ < z_num_;z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {

                    const size_t offset_pc_data = x_num_ * z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    total_p += j_num;
                }
            }

            Ip.resize(total_p);

            total_p = 0;

            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    std::copy(temp_int.data[i][offset_pc_data].begin(),temp_int.data[i][offset_pc_data].end(),Ip.begin() + total_p);

                    total_p += j_num;

                }

            }

            dims = Ip.size();

            std::string name = "Ip_size_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);

            if(Ip.size() > 0){
                //write the parts

                name = "Ip_"+std::to_string(i);

                hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, Ip.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

            }

        }

        write_timer.stop_timer();


        write_timer.start_timer("pc_data");


        for(uint64_t i = apr.pc_data.depth_min;i < apr.pc_data.depth_max;i++){


            unsigned int x_num_ = apr.pc_data.x_num[i];
            unsigned int z_num_ = apr.pc_data.z_num[i];
            unsigned int y_num_ = apr.pc_data.y_num[i];

            uint64_t node_val;

            std::vector<uint8_t> p_map;

            p_map.resize(x_num_*z_num_*y_num_,0);

            std::fill(p_map.begin(), p_map.end(), 0);

            //For each depth there are two loops, one for SEED status particles, at depth + 1, and one for BOUNDARY and FILLER CELLS, to ensure contiguous memory access patterns.

            // SEED PARTICLE STATUS LOOP (requires access to three data structures, particle access, particle data, and the part-map)
#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_p_map = y_num_*x_num_*z_ + y_num_*x_;

                    const size_t j_num =apr.pc_data.data[i][offset_pc_data].size();

                    y_coord = 0;

                    for(j_ = 0;j_ < j_num;j_++){

                        node_val =apr.pc_data.data[i][offset_pc_data][j_];

                        if (!(node_val&1)){
                            //get the index gap node
                            y_coord++;

                            status =apr.pc_data.get_status(node_val);

                            if(status > 1) {

                                p_map[offset_p_map + y_coord] = status;

                            }

                        } else {

                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure

                        }

                    }

                }

            }


            x_num_ =apr.pc_data.x_num[i+1];
            z_num_ =apr.pc_data.z_num[i+1];
            y_num_ =apr.pc_data.y_num[i+1];

            int x_num_d =apr.pc_data.x_num[i];
            int z_num_d =apr.pc_data.z_num[i];
            int y_num_d =apr.pc_data.y_num[i];

#pragma omp parallel for default(shared) private(z_,x_,j_,node_val,status,y_coord) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;
                    const size_t offset_p_map = y_num_d*x_num_d*(z_/2) + y_num_d*(x_/2);

                    const size_t j_num =apr.pc_data.data[i+1][offset_pc_data].size();

                    y_coord = 0;

                    for(j_ = 0;j_ < j_num;j_++){

                        node_val =apr.pc_data.data[i+1][offset_pc_data][j_];

                        if (!(node_val&1)){
                            //get the index gap node
                            y_coord++;

                            status =apr.pc_data.get_status(node_val);

                            if(status == 1) {

                                p_map[offset_p_map + y_coord/2] = status;

                            }

                        } else {

                            y_coord = (node_val & NEXT_COORD_MASK) >> NEXT_COORD_SHIFT;
                            y_coord--; //set the y_coordinate to the value before the next coming up in the structure
                        }

                    }

                }

            }


            dims = p_map.size();
            name = "p_map_"+std::to_string(i);
            hdf5_write_data_blosc(obj_id,H5T_NATIVE_UINT8,name.c_str(),rank,&dims, p_map.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

            name = "p_map_x_num_"+std::to_string(i);
            hsize_t attr = x_num_d;
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

            attr = y_num_d;
            name = "p_map_y_num_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

            attr = z_num_d;
            name = "p_map_z_num_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a, &attr);

        }

        write_timer.stop_timer();

        hsize_t attr =apr.pc_data.depth_min;
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &apr.pc_data.depth_max );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );

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
    template<typename ImageType,typename T>
    void write_apr_paraview(APR<ImageType>& apr,std::string save_loc,std::string file_name,ExtraPartCellData<T>& parts){
        //
        //
        //  Bevan Cheeseman 2018
        //
        //  Writes the APR and Extra PartCell Data to
        //
        //

        unsigned int blosc_comp_type = BLOSC_ZSTD;
        unsigned int blosc_comp_level = 1;
        unsigned int blosc_shuffle = 2;

        APR_timer write_timer;

        write_timer.verbose_flag = true;

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

        std::string hdf5_file_name = save_loc + file_name + "_paraview.h5";

        file_name = file_name + "_paraview";

        hdf5_create_file_blosc(hdf5_file_name);

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

        obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

        dims_out[0] = 1;
        dims_out[1] = 1;

        //just an identifier in here for the reading of the parts

        int num_parts = apr.num_parts_total;
        int num_cells = apr.num_elements_total;

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

        // New parameter and background data

        if(apr.pars.name.size() == 0){
            apr.pars.name = "no_name";
            apr.name = "no_name";
        }

        hdf5_write_string_blosc(pr_groupid,"name",apr.pars.name);

        std::string git_hash = exec_blosc("git rev-parse HEAD");

        hdf5_write_string_blosc(pr_groupid,"githash",git_hash);

        //////////////////////////////////////////////////////////////////
        //
        //  Write data to the file
        //
        //
        //
        ///////////////////////////////////////////////////////////////////////

        write_timer.start_timer("intensities");

        uint64_t depth_min =apr.pc_data.depth_min;

        std::vector<T> Ip;

        ExtraPartCellData<T> temp_int;

        temp_int = parts;

        write_timer.stop_timer();

        write_timer.start_timer("shift");

        apr.shift_particles_from_cells(temp_int);

        write_timer.stop_timer();

        write_timer.start_timer("write int");

        std::string name;


        uint64_t total_p = 0;


        for(uint64_t i =apr.pc_data.depth_min;i <=apr.pc_data.depth_max;i++) {

            const unsigned int x_num_ =apr.pc_data.x_num[i];
            const unsigned int z_num_ =apr.pc_data.z_num[i];
            const unsigned int y_num_ =apr.pc_data.y_num[i];


            for (z_ = 0; z_ < z_num_; z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {

                    const size_t offset_pc_data = x_num_ * z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    total_p += j_num;
                }
            }
        }


        Ip.resize(total_p);

        total_p = 0;
        for(uint64_t i =apr.pc_data.depth_min;i <=apr.pc_data.depth_max;i++) {

            const unsigned int x_num_ =apr.pc_data.x_num[i];
            const unsigned int z_num_ =apr.pc_data.z_num[i];
            const unsigned int y_num_ =apr.pc_data.y_num[i];

            for (z_ = 0; z_ < z_num_; z_++) {

                for (x_ = 0; x_ < x_num_; x_++) {

                    const size_t offset_pc_data = x_num_ * z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    std::copy(temp_int.data[i][offset_pc_data].begin(), temp_int.data[i][offset_pc_data].end(),
                              Ip.begin() + total_p);

                    total_p += j_num;

                }

            }
        }

        dims = Ip.size();


        //write the parts

        name = "Ip";

        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, Ip.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        write_timer.stop_timer();

        apr.get_part_numbers();

        std::vector<uint16_t> xv;
        xv.reserve(apr.num_parts_total);

        std::vector<uint16_t> yv;
        yv.reserve(apr.num_parts_total);

        std::vector<uint16_t> zv;
        zv.reserve(apr.num_parts_total);

        std::vector<uint8_t> levelv;
        levelv.reserve(apr.num_parts_total);

        std::vector<uint8_t> typev;
        typev.reserve(apr.num_parts_total);


        for(apr.begin();apr.end()!=0;apr.it_forward()){
            xv.push_back((uint16_t)apr.x_global());
            yv.push_back((uint16_t)apr.y_global());
            zv.push_back((uint16_t)apr.z_global());
            levelv.push_back((uint8_t)apr.level());
            typev.push_back((uint8_t)apr.type());
        }

        name = "x";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, xv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "y";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, yv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "z";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, zv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "level";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT8, name.c_str(), rank, &dims, levelv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "type";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT8, name.c_str(), rank, &dims, typev.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        hsize_t attr =apr.pc_data.depth_min;
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &apr.pc_data.depth_max );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );

        // output the file size
        hsize_t file_size;
        H5Fget_filesize(fid, &file_size);

        std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;

        write_main_paraview_xdmf_xml(save_loc,file_name,apr.num_parts_total);

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        std::cout << "Writing Complete" << std::endl;

    }

    template<typename ImageType, typename S>
    void write_particles_only(APR<ImageType> apr, std::string save_loc,std::string file_name,ExtraPartCellData<S>& parts_extra){
        //
        //
        //  Bevan Cheeseman 2018
        //
        //  Writes only the particle data, requires the same APR to be read in correctly.
        //
        //  #FIX_ME Extend me.
        //
        //

        std::cout << "Either uint8, uint16, or float" << std::endl;

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

        std::string hdf5_file_name = save_loc + file_name + "_apr_extra_parts.h5";

        file_name = file_name + "_apr_extra_parts";

        hdf5_create_file_blosc(hdf5_file_name);

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

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &apr.pc_data.org_dims[1] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &apr.pc_data.org_dims[0] );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &apr.pc_data.org_dims[2] );


        obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

        dims_out[0] = 1;
        dims_out[1] = 1;

        //just an identifier in here for the reading of the parts

        int num_parts = apr.num_parts_total;
        int num_cells = apr.num_elements_total;

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

        // New parameter and background data

        if(apr.pars.name.size() == 0){
            apr.pars.name = "no_name";
            apr.name = "no_name";
        }

        hdf5_write_string_blosc(pr_groupid,"name",apr.pars.name);

        std::string git_hash = exec_blosc("git rev-parse HEAD");

        hdf5_write_string_blosc(pr_groupid,"githash",git_hash);


        S val = 0;
        hid_t type = get_type_blosc(val);
        int type_id = type;

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"data_type",1,dims_out, &type_id);

        //////////////////////////////////////////////////////////////////
        //
        //  Write data to the file
        //
        //
        //
        ///////////////////////////////////////////////////////////////////////

        uint64_t depth_min =apr.pc_data.depth_min;

        std::vector<S> Ip;

        ExtraPartCellData<S> temp_int;
        temp_int.initialize_structure_cells(apr.pc_data);
        temp_int.data = parts_extra.data;

        apr.shift_particles_from_cells(temp_int);


        for(uint64_t i =apr.pc_data.depth_min;i <=apr.pc_data.depth_max;i++){

            const unsigned int x_num_ =apr.pc_data.x_num[i];
            const unsigned int z_num_ =apr.pc_data.z_num[i];
            const unsigned int y_num_ =apr.pc_data.y_num[i];

            //write the vals

            Ip.resize(0);

            for(z_ = 0;z_ < z_num_;z_++){

                for(x_ = 0;x_ < x_num_;x_++){

                    const size_t offset_pc_data = x_num_*z_ + x_;

                    const size_t j_num = temp_int.data[i][offset_pc_data].size();

                    uint64_t curr_size = Ip.size();
                    Ip.resize(curr_size+ j_num);

                    std::copy(temp_int.data[i][offset_pc_data].begin(),temp_int.data[i][offset_pc_data].end(),Ip.begin() + curr_size);

                }

            }

            dims = Ip.size();

            std::string name = "Ip_size_"+std::to_string(i);
            hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,name.c_str(),1,&dim_a,&dims);

            if(Ip.size() > 0){
                //write the parts

                name = "Ip_"+std::to_string(i);

                S val = 0;

                hdf5_write_data_blosc(obj_id, get_type_blosc(val), name.c_str(), rank, &dims, Ip.data());

            }

        }

        hsize_t attr =apr.pc_data.depth_min;
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_max",1,&dim_a, &apr.pc_data.depth_max );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"depth_min",1,&dim_a, &attr );

        // output the file size
        hsize_t file_size;
        H5Fget_filesize(fid, &file_size);

        std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        std::cout << "Writing ExtraPartCellData Complete" << std::endl;

    }

    template<typename T,typename ImageType>
    void read_parts_only(APR<ImageType>& apr,std::string file_name,ExtraPartCellData<T>& extra_parts)
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
        std::string name;

        /////////////////////////////////////////////
        //  Get metadata
        //
        //////////////////////////////////////////////

        hid_t       aid, atype, attr;

        aid = H5Screate(H5S_SCALAR);

        int data_type;

        attr_id = 	H5Aopen(pr_groupid,"data_type",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT,&data_type ) ;
        H5Aclose(attr_id);

        hid_t hdf5_data_type = data_type;

        std::vector<std::vector<T>> Ip;

        Ip.resize(apr.pc_data.depth_max+1);

        for(int i =apr.pc_data.depth_min;i <=apr.pc_data.depth_max; i++){

            //get the info

            int Ip_num;
            name = "Ip_size_"+std::to_string(i);

            attr_id = 	H5Aopen(pr_groupid,name.c_str(),H5P_DEFAULT);
            H5Aread(attr_id,H5T_NATIVE_INT,&Ip_num);
            H5Aclose(attr_id);

            Ip[i].resize(Ip_num);

            if(Ip[i].size()>0){
                name = "Ip_"+std::to_string(i);
                hdf5_load_data_blosc(obj_id,hdf5_data_type,Ip[i].data(),name.c_str());
            }

        }

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        extra_parts.initialize_structure_cells(apr.pc_data);

        for (int depth = apr.depth_min(); depth <= apr.depth_max(); ++depth) {

            uint64_t counter = 0;

            for (apr.begin(depth); apr.end() == true ; apr.it_forward(depth)) {

                apr.curr_level.get_val(extra_parts) = Ip[depth][counter];

                counter++;

            }
        }


    }


};


#endif //PARTPLAY_APRWRITER_HPP
