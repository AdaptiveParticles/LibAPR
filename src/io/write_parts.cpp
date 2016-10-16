
#include "write_parts.h"

void write_apr_full_format(Part_rep& p_rep,Tree<float>& tree,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2015
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
    
    
    int num_cells = 0;
    int num_parts = 0 ;
    
    std::vector<coords3d> part_coords;
    uint8_t curr_status = 0;
    
    int counter = 0;
    
    for(int l = p_rep.pl_map.k_min;l <= (p_rep.pl_map.k_max+1);l++){
        for(LevelIterator<float> it(tree, l); it != it.end(); it++)
        {
            
            it.get_current_particle_coords(part_coords);
            curr_status = tree.get_status(*it);
            
                        
            for(int i = 0; i < part_coords.size();i++){
                counter++;
                type_vec.push_back(curr_status);
                k_vec.push_back(l);
                x_c.push_back(part_coords[i].x);
                y_c.push_back(part_coords[i].y);
                z_c.push_back(part_coords[i].z);
                Ip.push_back((uint16_t)(tree.get_content_part(*it,i)).intensity);
            }
            
            if(curr_status > 0){
                num_cells++;
            }
            
        }
    }
    
    num_parts = x_c.size();
    
    std::string hdf5_file_name = save_loc + file_name + "_full_part.h5";
    
    file_name = file_name + "_full_part";
    
    hdf5_create_file(hdf5_file_name);
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id, data_id, type_id, dataspace, type_class, space_id;
    H5G_info_t info;
    
    hsize_t     dims_out[2];
    hsize_t     mdims_out[2];
    hsize_t rank = 1;
    
    hsize_t dims;
    
    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);
    
    //Get the group you want to open
    
    float *buff2 = new float[2];
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
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &p_rep.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &p_rep.org_dims[0] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &p_rep.org_dims[2] );
    
    
    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    
    dims_out[0] = 1;
    dims_out[1] = 1;
    
    //just an identifier in here for the reading of the parts
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );
    
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_max",1,&dims, &p_rep.pl_map.k_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_min",1,&dims, &p_rep.pl_map.k_min );
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
    
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    //create the xdmf file
    write_full_xdmf_xml(save_loc,file_name,num_parts);
    
    
    std::cout << "Writing Complete" << std::endl;
    
}

void write_apr_to_hdf5(Part_rep& p_rep,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2015
    //
    //  Writes APR to hdf5 file, including xdmf to make readible by paraview, and includes any fields that are flagged extra to print
    //
    //

    std::cout << "Writing parts to hdf5 file, in full format..." << std::endl;

    //containers for all the variables
    Part_data<uint8_t> k_vec;

    Part_data<uint8_t> type_vec;

    int num_cells = p_rep.get_active_cell_num();
    int num_parts = p_rep.num_parts;

    k_vec.data.reserve(num_parts);


    p_rep.get_all_part_co_ords();

    Part_data<uint16_t>& x_coords = *p_rep.x;
    Part_data<uint16_t>& y_coords = *p_rep.y;
    Part_data<uint16_t>& z_coords = *p_rep.z;

    get_all_part_k(p_rep,k_vec);

    get_all_part_type(p_rep,type_vec);

    std::string hdf5_file_name = save_loc + file_name + "_full.h5";

    file_name = file_name + "_full";

    hdf5_create_file(hdf5_file_name);

    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id, data_id, type_id, dataspace, type_class, space_id;
    H5G_info_t info;

    hsize_t     dims_out[2];
    hsize_t     mdims_out[2];
    hsize_t rank = 1;

    hsize_t dims;

    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //Get the group you want to open

    float *buff2 = new float[2];
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

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &p_rep.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &p_rep.org_dims[0] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &p_rep.org_dims[2] );


    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

    dims_out[0] = 1;
    dims_out[1] = 1;

    //just an identifier in here for the reading of the parts

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_max",1,&dims, &p_rep.pl_map.k_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_min",1,&dims, &p_rep.pl_map.k_min );
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
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"x",rank,&dims, x_coords.data.data() );

    //write the y co_ordinates
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"y",rank,&dims, y_coords.data.data() );

    //write the z co_ordinates

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"z",rank,&dims, z_coords.data.data() );

    //write the z co_ordinates

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT8,"type",rank,&dims, type_vec.data.data() );

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT8,"k",rank,&dims, k_vec.data.data() );

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"Ip",rank,&dims,p_rep.Ip.data.data()  );


    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);

    //create the xdmf file
    write_full_xdmf_xml(save_loc,file_name,num_parts);


    std::cout << "Writing Complete" << std::endl;

}
void write_apr_to_hdf5_inc_extra_fields(Part_rep& p_rep,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2015
    //
    //  Writes APR to hdf5 file, including xdmf to make readible by paraview, and includes any fields that are flagged extra to print
    //
    //

    std::cout << "Writing parts to hdf5 file, in full format..." << std::endl;

    //containers for all the variables

    Part_data<uint8_t> k_vec;

    Part_data<uint8_t> type_vec;

    int num_cells = p_rep.get_active_cell_num();
    int num_parts = p_rep.num_parts;

    k_vec.data.reserve(num_parts);

    p_rep.get_all_part_co_ords();

    Part_data<uint16_t>& x_coords = *p_rep.x;
    Part_data<uint16_t>& y_coords = *p_rep.y;
    Part_data<uint16_t>& z_coords = *p_rep.z;

    get_all_part_k(p_rep,k_vec);

    get_all_part_type(p_rep,type_vec);

    std::string hdf5_file_name = save_loc + file_name + "_full.h5";

    file_name = file_name + "_full";

    hdf5_create_file(hdf5_file_name);

    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id, data_id, type_id, dataspace, type_class, space_id;
    H5G_info_t info;

    hsize_t     dims_out[2];
    hsize_t     mdims_out[2];
    hsize_t rank = 1;

    hsize_t dims;

    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //Get the group you want to open

    float *buff2 = new float[2];
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

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &p_rep.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &p_rep.org_dims[0] );


    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &p_rep.org_dims[2] );



    // t_groupid = H5G.create(fid,t_stack_name,plist,plist,plist);

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

    dims = num_parts;

    //write the x co_ordinates

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"x",rank,&dims, x_coords.data.data() );

    //write the y co_ordinates
    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"y",rank,&dims, y_coords.data.data() );

    //write the z co_ordinates

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"z",rank,&dims, z_coords.data.data() );

    //write the z co_ordinates

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT8,"type",rank,&dims, type_vec.data.data() );

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT8,"k",rank,&dims, k_vec.data.data() );

    dims = num_parts;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"Ip",rank,&dims,p_rep.Ip.data.data()  );

    std::vector<std::string> extra_data_type;
    std::vector<std::string> extra_data_name;

    int req_size = num_parts;
    int flag_type = 1;

    write_part_data_to_hdf5(p_rep,obj_id,extra_data_type,extra_data_name,flag_type,req_size);


    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);

    //create the xdmf file
    write_full_xdmf_xml_extra(save_loc,file_name,num_parts,extra_data_name,extra_data_type);



    std::cout << "Writing Complete" << std::endl;

}
void write_full_xdmf_xml(std::string save_loc,std::string file_name,int num_parts){
    //
    //
    //
    //


    std::string hdf5_file_name = file_name + ".h5";
    std::string xdmf_file_name = save_loc + file_name + ".xmf";


    std::ofstream myfile;
    myfile.open (xdmf_file_name);

    myfile << "<?xml version=\"1.0\" ?>\n";
    myfile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    myfile << "<Xdmf Version=\"2.0\" xmlns:xi=\"[http://www.w3.org/2001/XInclude]\">\n";
    myfile <<  " <Domain>\n";
    myfile <<  "   <Grid Name=\"parts\" GridType=\"Uniform\">\n";
    myfile <<  "     <Topology TopologyType=\"Polyvertex\" Dimensions=\"" << num_parts << "\"/>\n";
    myfile <<  "     <Geometry GeometryType=\"X_Y_Z\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/x\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/y\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/z\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "     </Geometry>\n";
    myfile <<  "     <Attribute Name=\"Ip\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/Ip\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "     <Attribute Name=\"k\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/k\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "     <Attribute Name=\"Type\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/type\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "   </Grid>\n";
    myfile <<  " </Domain>\n";
    myfile <<  "</Xdmf>\n";

    myfile.close();

}
void write_full_xdmf_xml_extra(std::string save_loc,std::string file_name,int num_parts,std::vector<std::string> extra_data_names,std::vector<std::string> extra_data_types){
    //
    //
    //  Also writes out extra datafields
    //


    std::string hdf5_file_name = file_name + ".h5";
    std::string xdmf_file_name = save_loc + file_name + ".xmf";


    std::ofstream myfile;
    myfile.open (xdmf_file_name);

    myfile << "<?xml version=\"1.0\" ?>\n";
    myfile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    myfile << "<Xdmf Version=\"2.0\" xmlns:xi=\"[http://www.w3.org/2001/XInclude]\">\n";
    myfile <<  " <Domain>\n";
    myfile <<  "   <Grid Name=\"parts\" GridType=\"Uniform\">\n";
    myfile <<  "     <Topology TopologyType=\"Polyvertex\" Dimensions=\"" << num_parts << "\"/>\n";
    myfile <<  "     <Geometry GeometryType=\"X_Y_Z\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/x\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/y\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/z\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "     </Geometry>\n";
    myfile <<  "     <Attribute Name=\"Ip\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/Ip\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "     <Attribute Name=\"k\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/k\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";
    myfile <<  "     <Attribute Name=\"Type\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"1\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/type\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "    </Attribute>\n";

    for (int i = 0; i < extra_data_names.size(); i++) {
        //adds the additional datasets

        std::string num_type;
        std::string prec;


        if (extra_data_types[i]=="uint8_t") {
            num_type = "UInt";
            prec = "1";
        } else if (extra_data_types[i]=="bool"){
            std::cout << "Bool type can't be stored with xdmf change datatype" << std::endl;
        } else if (extra_data_types[i]=="uint16_t"){
            num_type = "UInt";
            prec = "2";
        } else if (extra_data_types[i]=="int16_t"){
            num_type = "Int";
            prec = "2";

        } else if (extra_data_types[i]=="int"){
            num_type = "Int";
            prec = "4";
        }  else if (extra_data_types[i]=="int8_t"){
            num_type = "Int";
            prec = "1";
        }else if (extra_data_types[i]=="float"){
            num_type = "Float";
            prec = "4";
        } else {
            std::cout << "Unknown Type in extra printout encournterd" << std::endl;

        }

        myfile <<  "     <Attribute Name=\""<< extra_data_names[i] <<"\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"" << num_type << "\" Precision=\"" << prec <<"\" Format=\"HDF\">\n";
        myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/" <<  extra_data_names[i] << "\n";
        myfile <<  "       </DataItem>\n";
        myfile <<  "    </Attribute>\n";

    }

    myfile <<  "   </Grid>\n";
    myfile <<  " </Domain>\n";
    myfile <<  "</Xdmf>\n";

    myfile.close();

}
void write_xdmf_xml_only_extra(std::string save_loc,std::string file_name,int num_parts,std::vector<std::string> extra_data_names,std::vector<std::string> extra_data_types){
    //
    //
    //  Also writes out extra datafields
    //


    std::string hdf5_file_name = file_name + ".h5";
    std::string xdmf_file_name = save_loc + file_name + ".xmf";


    std::ofstream myfile;
    myfile.open (xdmf_file_name);

    myfile << "<?xml version=\"1.0\" ?>\n";
    myfile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    myfile << "<Xdmf Version=\"2.0\" xmlns:xi=\"[http://www.w3.org/2001/XInclude]\">\n";
    myfile <<  " <Domain>\n";
    myfile <<  "   <Grid Name=\"parts\" GridType=\"Uniform\">\n";
    myfile <<  "     <Topology TopologyType=\"Polyvertex\" Dimensions=\"" << num_parts << "\"/>\n";
    myfile <<  "     <Geometry GeometryType=\"X_Y_Z\">\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/x\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/y\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"UInt\" Precision=\"2\" Format=\"HDF\">\n";
    myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/z\n";
    myfile <<  "       </DataItem>\n";
    myfile <<  "     </Geometry>\n";


    for (int i = 0; i < extra_data_names.size(); i++) {
        //adds the additional datasets

        std::string num_type;
        std::string prec;


        if (extra_data_types[i]=="uint8_t") {
            num_type = "UInt";
            prec = "1";
        } else if (extra_data_types[i]=="bool"){
            std::cout << "Bool type can't be stored with xdmf change datatype" << std::endl;
        } else if (extra_data_types[i]=="uint16_t"){
            num_type = "UInt";
            prec = "2";
        } else if (extra_data_types[i]=="int16_t"){
            num_type = "Int";
            prec = "2";

        }else if (extra_data_types[i]=="int"){
            num_type = "Int";
            prec = "4";
        }  else if (extra_data_types[i]=="int8_t"){
            num_type = "Int";
            prec = "1";
        }else if (extra_data_types[i]=="float"){
            num_type = "Float";
            prec = "4";
        } else {
            std::cout << "Unknown Type in extra printout encournterd" << std::endl;

        }

        myfile <<  "     <Attribute Name=\""<< extra_data_names[i] <<"\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        myfile <<  "       <DataItem Dimensions=\""<< num_parts <<"\" NumberType=\"" << num_type << "\" Precision=\"" << prec <<"\" Format=\"HDF\">\n";
        myfile <<  "        " << hdf5_file_name << ":/ParticleRepr/t/" <<  extra_data_names[i] << "\n";
        myfile <<  "       </DataItem>\n";
        myfile <<  "    </Attribute>\n";

    }

    myfile <<  "   </Grid>\n";
    myfile <<  " </Domain>\n";
    myfile <<  "</Xdmf>\n";

    myfile.close();

}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Writing the particle cell structures to file, readable by paraview.
//
//////////////////////////////////////////////////////////////////////////////////////////////////
void write_part_cells_to_hdf5(Part_rep& p_rep,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the particle cell structure to file
    //
    //

    std::cout << "Writing particle cell structures to hdf5 file" << std::endl;

    //containers for all the variables


    Part_data<uint8_t>* k_vec;
    Part_data<uint8_t>* type_vec;



    int num_cells = p_rep.get_active_cell_num();
    int num_parts = p_rep.num_parts;

    p_rep.create_uint8_dataset("k",num_cells);
    p_rep.create_uint8_dataset("type",num_cells);

    k_vec = p_rep.get_data_ref<uint8_t>("k");
    type_vec = p_rep.get_data_ref<uint8_t>("type");

    p_rep.part_data_list["k"].print_flag = 1;
    p_rep.part_data_list["type"].print_flag = 1;

    Part_data<uint16_t> x_coords_cell;
    Part_data<uint16_t> y_coords_cell;
    Part_data<uint16_t> z_coords_cell;

    //get the cell information
    get_cell_properties(p_rep,y_coords_cell,x_coords_cell,z_coords_cell,(*k_vec),(*type_vec));

    std::string hdf5_file_name = save_loc + file_name + "_cells.h5";

    file_name = file_name + "_cells";

    hdf5_create_file(hdf5_file_name);



    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id, data_id, type_id, dataspace, type_class, space_id;
    H5G_info_t info;

    hsize_t     dims_out[2];
    hsize_t     mdims_out[2];
    hsize_t rank = 1;

    hsize_t dims;

    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //create the main group
    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

    //Get the group you want to open

    float *buff2 = new float[2];
    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;

    //pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &p_rep.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &p_rep.org_dims[0] );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &p_rep.org_dims[2] );


    // t_groupid = H5G.create(fid,t_stack_name,plist,plist,plist);

    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

    dims_out[0] = 1;
    dims_out[1] = 1;

    //just an identifier in here for the reading of the parts

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );


    //////////////////////////////////////////////////////////////////
    //
    //  Write part_cell data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////


    //write the x co_ordinates

    dims = num_cells;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"x",rank,&dims, x_coords_cell.data.data() );

    //write the y co_ordinates
    dims = num_cells;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"y",rank,&dims, y_coords_cell.data.data() );

    //write the z co_ordinates

    dims = num_cells;
    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,"z",rank,&dims, z_coords_cell.data.data() );

    //write the z co_ordinates


    std::vector<std::string> extra_data_type;
    std::vector<std::string> extra_data_name;


    int req_size = num_cells;
    int flag_type = 1;

    write_part_data_to_hdf5(p_rep,obj_id,extra_data_type,extra_data_name,flag_type,req_size);



    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);

    //create the xdmf file
    write_xdmf_xml_only_extra(save_loc,file_name,num_cells,extra_data_name,extra_data_type);


    std::cout << "Writing Complete" << std::endl;

}
void write_qcompress_to_hdf5(Part_rep p_rep,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the particle structure to file; well cell info; status for location and quantized compression.
    //
    //

    std::cout << "Writing particle data in qcompress form to hdf5 file" << std::endl;

    //containers for all the variables


    int num_cells = p_rep.get_active_cell_num();
    int num_parts = p_rep.num_parts;

    //create the datasets
    p_rep.create_uint8_dataset("k",num_cells);
    p_rep.create_uint8_dataset("s",num_cells);
    p_rep.create_uint8_dataset("scale",num_cells);
    p_rep.create_uint16_dataset("shift",num_cells);

    p_rep.create_int8_dataset("delta",num_parts);

    p_rep.create_uint16_dataset("x",num_cells);
    p_rep.create_uint16_dataset("y",num_cells);
    p_rep.create_uint16_dataset("z",num_cells);

    //set up the pointers
    Part_data<uint8_t>* k;
    Part_data<uint8_t>* s;
    Part_data<uint8_t>* scale;
    Part_data<uint16_t>* shift;
    Part_data<int8_t>* delta;

    Part_data<uint16_t>* x;
    Part_data<uint16_t>* y;
    Part_data<uint16_t>* z;


    //set them to be written
    p_rep.part_data_list["k"].print_flag = 2;
    p_rep.part_data_list["s"].print_flag = 2;
    p_rep.part_data_list["scale"].print_flag = 2;
    p_rep.part_data_list["shift"].print_flag = 2;
    p_rep.part_data_list["delta"].print_flag = 2;

    p_rep.part_data_list["x"].print_flag = 2;
    p_rep.part_data_list["y"].print_flag = 2;
    p_rep.part_data_list["z"].print_flag = 2;


    //get the references

    k = p_rep.get_data_ref<uint8_t>("k");
    s = p_rep.get_data_ref<uint8_t>("s");
    scale = p_rep.get_data_ref<uint8_t>("scale");

    shift = p_rep.get_data_ref<uint16_t>("shift");
    delta = p_rep.get_data_ref<int8_t>("delta");

    x = p_rep.get_data_ref<uint16_t>("x");
    y = p_rep.get_data_ref<uint16_t>("y");
    z = p_rep.get_data_ref<uint16_t>("z");


    //get the cell information
    get_cell_properties(p_rep,(*y),(*x),(*z),(*k),(*s),0,1);

    std::string hdf5_file_name = save_loc + file_name + "_qc.h5";

    file_name = file_name + "_qc";

    hdf5_create_file(hdf5_file_name);

    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id, data_id, type_id, dataspace, type_class, space_id;
    H5G_info_t info;

    hsize_t     dims_out[2];
    hsize_t     mdims_out[2];
    hsize_t rank = 1;

    hsize_t dims;

    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //Get the group you want to open

    float *buff2 = new float[2];
    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;

    //create the main group
    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &p_rep.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &p_rep.org_dims[0] );


    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &p_rep.org_dims[2] );


    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"comp_scale",1,&dims, &p_rep.pars.comp_scale );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_max",1,&dims, &p_rep.pl_map.k_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_min",1,&dims, &p_rep.pl_map.k_min );


    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"rel_error",1,&dims, &p_rep.pars.rel_error );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"len_scale",1,&dims, &p_rep.len_scale );


    // t_groupid = H5G.create(fid,t_stack_name,plist,plist,plist);

    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

    dims_out[0] = 1;
    dims_out[1] = 1;

    //just an identifier in here for the reading of the parts

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

    /////////////////////////////////////////////////////////////////////////////
    //
    //
    //  Perform the q-compress step
    //
    //
    //////////////////////////////////////////////////////////////////////

    float mean_ip,max_ip,min_ip;
    int curr_scale,curr_shift,curr_delta;

    for (int i = 0; i < num_cells; i++){

        if(p_rep.status.data[i] > 1){
            //all the particle cell

            mean_ip = 0;
            max_ip = 0;
            min_ip = 99999999999999;

            //calculate local intensity properties
            for(int p = p_rep.pl_map.cell_indices[i].first;p < p_rep.pl_map.cell_indices[i].last;p++){
                //intensity
                mean_ip += p_rep.Ip.data[p];
                max_ip = std::max(max_ip,(float)p_rep.Ip.data[p]);
                min_ip = std::min(min_ip,(float)p_rep.Ip.data[p]);

            }

            //catch safe to get the border particles
            if (min_ip == 0) {
                //calculate local intensity properties
                min_ip = 99999999999;
                for(int p = p_rep.pl_map.cell_indices[i].first;p < p_rep.pl_map.cell_indices[i].last;p++){
                    //intensity

                    if (p_rep.Ip.data[p] == 0) {
                        p_rep.Ip.data[p] = max_ip/2;
                    }


                    mean_ip += p_rep.Ip.data[p];
                    min_ip = std::min(min_ip,(float)p_rep.Ip.data[p]);

                }
            }

            mean_ip = mean_ip/(p_rep.pl_map.cell_indices[i].last-p_rep.pl_map.cell_indices[i].first);

            //calculate the shift and scale for the particle cell compression
            curr_shift = round(mean_ip/10)*10;

            curr_scale = ceil(log(max_ip - min_ip)/log(2));

            scale->data[i] = curr_scale;
            shift->data[i] = curr_shift;

            //encode the intensities lossy step
            for(int p = p_rep.pl_map.cell_indices[i].first;p < p_rep.pl_map.cell_indices[i].last;p++){
                //intensity
                curr_delta = floor(p_rep.pars.comp_scale*(p_rep.Ip.data[p]  - curr_shift)/(pow(2,curr_scale)));
                delta->data[p] = (curr_delta);
                //int temp = round(curr_shift + (curr_delta*pow(2,curr_scale))/pars.comp_scale);
                //p_rep.Ip.data[p]  = round(curr_shift + (curr_delta*pow(2,curr_scale))/pars.comp_scale);

            }

        }
    }

    //////////////////////////////////////////////////////////////////
    //
    //  Write part_cell data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////

    std::vector<std::string> extra_data_type;
    std::vector<std::string> extra_data_name;

    int req_size = 0;
    int flag_type = 2;

    write_part_data_to_hdf5(p_rep,obj_id,extra_data_type,extra_data_name,flag_type,req_size);

    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);

    //no xdmf file, because they are of different length datatypes

    std::cout << "Writing Complete" << std::endl;

}
void write_nocompress_to_hdf5(Part_rep p_rep,std::string save_loc,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2016
    //
    //  Writes the particle structure to file; cell info; status for location and no compression.
    //
    //

    std::cout << "Writing particle data in nocompress form to hdf5 file" << std::endl;

    //containers for all the variables


    int num_cells = p_rep.get_active_cell_num();
    int num_parts = p_rep.num_parts;

    //create the datasets
    p_rep.create_uint8_dataset("k",num_cells);
    p_rep.create_uint8_dataset("s",num_cells);


    p_rep.create_uint16_dataset("x",num_cells);
    p_rep.create_uint16_dataset("y",num_cells);
    p_rep.create_uint16_dataset("z",num_cells);
    p_rep.create_uint16_dataset("Ip",0);

    //set up the pointers
    Part_data<uint8_t>* k;
    Part_data<uint8_t>* s;

    Part_data<uint16_t>* x;
    Part_data<uint16_t>* y;
    Part_data<uint16_t>* z;

    Part_data<uint16_t>* Ip;


    //set them to be written
    p_rep.part_data_list["k"].print_flag = 2;
    p_rep.part_data_list["s"].print_flag = 2;
    p_rep.part_data_list["Ip"].print_flag = 2;

    p_rep.part_data_list["x"].print_flag = 2;
    p_rep.part_data_list["y"].print_flag = 2;
    p_rep.part_data_list["z"].print_flag = 2;

    k = p_rep.get_data_ref<uint8_t>("k");
    s = p_rep.get_data_ref<uint8_t>("s");

    x = p_rep.get_data_ref<uint16_t>("x");
    y = p_rep.get_data_ref<uint16_t>("y");
    z = p_rep.get_data_ref<uint16_t>("z");

    Ip = p_rep.get_data_ref<uint16_t>("Ip");

    //just assign the intensity
    Ip->data = p_rep.Ip.data;


    //get the cell information
    get_cell_properties(p_rep,(*y),(*x),(*z),(*k),(*s),1,1);

    std::string hdf5_file_name = save_loc + file_name + "_nc.h5";

    file_name = file_name + "_nc";

    hdf5_create_file(hdf5_file_name);

    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id, data_id, type_id, dataspace, type_class, space_id;
    H5G_info_t info;

    hsize_t     dims_out[2];
    hsize_t     mdims_out[2];
    hsize_t rank = 1;

    hsize_t dims;

    fid = H5Fopen(hdf5_file_name.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);

    //Get the group you want to open

    float *buff2 = new float[2];
    //////////////////////////////////////////////////////////////////
    //
    //  Write meta-data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////
    dims = 1;

    //create the main group
    pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"x_num",1,&dims, &p_rep.org_dims[1] );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"y_num",1,&dims, &p_rep.org_dims[0] );


    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"z_num",1,&dims, &p_rep.org_dims[2] );


    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"comp_scale",1,&dims, &p_rep.pars.comp_scale );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_max",1,&dims, &p_rep.pl_map.k_max );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"k_min",1,&dims, &p_rep.pl_map.k_min );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"rel_error",1,&dims, &p_rep.pars.rel_error );
    hdf5_write_attribute(pr_groupid,H5T_NATIVE_FLOAT,"len_scale",1,&dims, &p_rep.len_scale );

    // t_groupid = H5G.create(fid,t_stack_name,plist,plist,plist);

    obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

    dims_out[0] = 1;
    dims_out[1] = 1;

    //just an identifier in here for the reading of the parts

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_parts",1,dims_out, &num_parts );

    hdf5_write_attribute(pr_groupid,H5T_NATIVE_INT,"num_cells",1,dims_out, &num_cells );

    /////////////////////////////////////////////////////////////////////////////
    //
    //
    //  Perform the no-compress step
    //
    //
    //////////////////////////////////////////////////////////////////////



    //////////////////////////////////////////////////////////////////
    //
    //  Write part_cell data to the file
    //
    //
    //
    ///////////////////////////////////////////////////////////////////////

    std::vector<std::string> extra_data_type;
    std::vector<std::string> extra_data_name;

    int req_size = 0;
    int flag_type = 2;

    write_part_data_to_hdf5(p_rep,obj_id,extra_data_type,extra_data_name,flag_type,req_size);

    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);

    //no xdmf file, because they are of different length datatypes

    std::cout << "Writing Complete" << std::endl;

}
void write_part_data_to_hdf5(Data_manager& p_rep,hid_t obj_id,std::vector<std::string>& extra_data_type,std::vector<std::string>& extra_data_name,int flag_type,int req_size){
    //
    //  Bevan Cheeseman 2016
    //
    //  This function writes part_data data types to hdf5 files, according to them satisfying the flag and data length requirements
    //
    //  flag_type: allows to set local part_data variables for printing; or deciding which set to prinrt;
    //
    //  req_size: allows you to determine if you want only a specific type of data printed out; which is requried if you want a valid xdmf file
    //
    //


    hsize_t dims;
    hsize_t rank = 1;

    for(auto const &part_data_info : p_rep.part_data_list) {
        if (part_data_info.second.print_flag == flag_type) {
            // add the name

            if (part_data_info.second.data_type=="uint8_t") {

                Part_data<uint8_t>* data_pointer = p_rep.get_data_ref<uint8_t>(part_data_info.first);
                dims = data_pointer->data.size();
                //check if it is particle data of cell data
                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {

                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);


                    hdf5_write_data(obj_id,H5T_NATIVE_UINT8,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data() );
                }

            } else if (part_data_info.second.data_type=="bool"){
                std::cout << "Bool type can't be stored with xdmf change datatype" << std::endl;
            } else if (part_data_info.second.data_type=="uint16_t"){

                Part_data<uint16_t>* data_pointer = p_rep.get_data_ref<uint16_t>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data(obj_id,H5T_NATIVE_UINT16,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data() );
                }

            }else if (part_data_info.second.data_type=="int16_t"){

                Part_data<int16_t>* data_pointer = p_rep.get_data_ref<int16_t>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data(obj_id,H5T_NATIVE_INT16,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data() );
                }

            }  else if (part_data_info.second.data_type=="int"){

                Part_data<int>* data_pointer = p_rep.get_data_ref<int>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data(obj_id,H5T_NATIVE_INT,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data() );

                }
            } else if (part_data_info.second.data_type=="float"){
                Part_data<float>* data_pointer = p_rep.get_data_ref<float>(part_data_info.first);
                dims = data_pointer->data.size();

                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {
                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_data(obj_id,H5T_NATIVE_FLOAT,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data() );
                }

            } else if (part_data_info.second.data_type=="int8_t") {

                Part_data<int8_t>* data_pointer = p_rep.get_data_ref<int8_t>(part_data_info.first);
                dims = data_pointer->data.size();
                //check if it is particle data of cell data
                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {

                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);


                    hdf5_write_data(obj_id,H5T_NATIVE_INT8,part_data_info.first.c_str(),rank,&dims,data_pointer->data.data() );
                }

            } else if (part_data_info.second.data_type=="string") {
                //ONLY WRITES TEH FIRST STRING!!##!$

                Part_data<std::string>* data_pointer = p_rep.get_data_ref<std::string>(part_data_info.first);
                dims = 1;
                //check if it is particle data of cell data
                if ((data_pointer->data.size() == req_size) | (req_size == 0)) {

                    extra_data_name.push_back(part_data_info.first);
                    extra_data_type.push_back(part_data_info.second.data_type);

                    hdf5_write_string(obj_id,part_data_info.first.c_str(),data_pointer->data[0]);

                }

            }else {
                std::cout << "Unknown Type in extra printout encournterd" << std::endl;

            }
        }
    }

}