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

#ifndef _read_parts_h
#define _read_parts_h

#include "hdf5functions.h"

void read_parts_from_encoded_hdf5(Part_rep& p_rep,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2015
    //
    //  Encoding step for algorithm using scale and shift encoding per particle cell
    //
    //
    
    std::cout << "Loading in particle dataset: " << file_name << std::endl;
    
    //containers for all the variables
    std::vector<uint16_t> x,y,z;
    std::vector<uint8_t> k,scale,s;
    std::vector<uint16_t> shift;
    std::vector<int8_t> delta;
    
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    //  Get the attributes
    //
    //
    /////////////////////////////////////////////////////////////////////////////////////////
    
    
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id;
    H5G_info_t info;
    
    
    int num_parts,num_cells;
    float comp_scale;
    int k_min,k_max;
    //Opening shit up
    
    fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    
    //Get the group you want to open
    
    pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    //Getting an attribute
    obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);
    
    //Load the attributes
#ifdef data_2D
    
    //removed this slice code
    //attr_id = 	H5Aopen(obj_id,"z",H5P_DEFAULT);
    //H5Aread(attr_id,H5T_NATIVE_INT,&part_list->slice) ;
    //H5Aclose(attr_id);
#endif
    
    /////////////////////////////////////////////
    //  Get metadata
    //
    //////////////////////////////////////////////
    float temp;
    
    p_rep.org_dims.resize(3);
    
    attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[0]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[1]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[2]) ;
    H5Aclose(attr_id);

    
    attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"comp_scale",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&comp_scale) ;
    H5Aclose(attr_id);
    
    p_rep.pars.comp_scale = comp_scale;
    
    attr_id = 	H5Aopen(pr_groupid,"rel_error",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&temp) ;
    H5Aclose(attr_id);
    
    p_rep.pars.rel_error = temp;
    
    attr_id = 	H5Aopen(pr_groupid,"len_scale",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_FLOAT,&temp) ;
    H5Aclose(attr_id);
    
    p_rep.pars.len_scale = temp;
    
    attr_id = 	H5Aopen(pr_groupid,"k_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&k_max) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"k_min",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&k_min) ;
    H5Aclose(attr_id);
    
    k.resize(num_cells);
    x.resize(num_cells);
    y.resize(num_cells);
    z.resize(num_cells);
    s.resize(num_cells);
    
    delta.resize(num_parts);
    scale.resize(num_cells);
    shift.resize(num_cells);
    
    //assign the values
    p_rep.num_parts = num_parts;
    
    
    std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;
    
    
    //Load the data then update the particle dataset
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,x.data(),"x");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,y.data(),"y");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,z.data(),"z");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT8,s.data(),"s");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT8,k.data(),"k");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT8,scale.data(),"scale");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,shift.data(),"shift");
    
    hdf5_load_data(obj_id,H5T_NATIVE_INT8,delta.data(),"delta");
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    std::cout << "Data loaded: now decoding and moving to data structures" << std::endl;
    
    //////////////////////////////////////////////////////////////////////////
    //
    //  Load stuff in to particle structure and decode intensity
    //
    ///////////////////////////////////////////////////////////////////////////////
    
    
    //init the cell map structures
    p_rep.pl_map.init_part_map(k_min,k_max);
    
    
    p_rep.status.data.resize(num_cells);
    p_rep.Ip.data.resize(num_parts);
    
    Cell_index curr_cell_index;
    Cell_id curr_cell_id;
    
    int counter = 0;
    
    int num_parts_in_cell;
    
    //loop over all the structures
    
    for(int i = 0; i < num_cells; i++){
        num_parts_in_cell = p_rep.get_num_parts_cell(s[i]);
        
        curr_cell_index.first = counter;
        curr_cell_index.last = counter + num_parts_in_cell;
        curr_cell_index.cindex = i;
        
        curr_cell_id.x = x[i];
        curr_cell_id.y = y[i];
        curr_cell_id.z = z[i];
        curr_cell_id.k = k[i];
        
        //add the cell to the required datastructures
        p_rep.pl_map.add_cell_to_map(curr_cell_id,curr_cell_index);
        
        p_rep.status.data[i] = s[i];
        
        for(int j = 0;j < num_parts_in_cell;j++){
            p_rep.Ip.data[counter] = round(shift[i]+ (delta[counter]*pow(2,scale[i]))/comp_scale);
            counter++;
        }
        
    }
    

    
    std::cout << "Load Complete" << std::endl;
    
    
}
void read_parts_from_full_hdf5(Part_rep& p_rep,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2015
    //
    //  Read a full type dataset back in ; shouldn't really be done; because it requires a bit of heavy lifting to figure out what the part rep was to begin with.
    //
    //
    
    std::cout << "Loading in particle dataset: " << file_name << std::endl;
    
    //containers for all the variables
    std::vector<uint16_t> x;
    std::vector<uint16_t> y;
    std::vector<uint16_t> z;
    std::vector<uint16_t> Ip;
    std::vector<uint8_t> k,type;

    
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    //  Get the attributes
    //
    //
    /////////////////////////////////////////////////////////////////////////////////////////
    
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id;
    H5G_info_t info;
    
    
    int num_parts,num_cells;
    float comp_scale;
    int k_min,k_max;
    //Opening shit up
    
    fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    
    //Get the group you want to open
    
    pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    //Getting an attribute
    obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);
    
    //Load the attributes
#ifdef data_2D
    
    //removed this slice code
    //attr_id = 	H5Aopen(obj_id,"z",H5P_DEFAULT);
    //H5Aread(attr_id,H5T_NATIVE_INT,&part_list->slice) ;
    //H5Aclose(attr_id);
#endif
    
    /////////////////////////////////////////////
    //  Get metadata
    //
    //////////////////////////////////////////////
    

    
    p_rep.org_dims.resize(3);
    
    attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[0]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[1]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[2]) ;
    H5Aclose(attr_id);
    
    
    attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"k_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&k_max) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"k_min",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&k_min) ;
    H5Aclose(attr_id);
    
    k.resize(num_parts);
    x.resize(num_parts);
    y.resize(num_parts);
    z.resize(num_parts);
    type.resize(num_parts);
    Ip.resize(num_parts);
    
    //assign the values
    p_rep.num_parts = num_parts;
    
    
    std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;
    
    
    //Load the data then update the particle dataset
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,x.data(),"x");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,y.data(),"y");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,z.data(),"z");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT8,type.data(),"type");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT8,k.data(),"k");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,Ip.data(),"Ip");
    
    
    //close things
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    std::cout << "Data loaded: now decoding and moving to data structures" << std::endl;
    
    //////////////////////////////////////////////////////////////////////////
    //
    //  Load stuff in to particle structure and decode intensity
    //
    ///////////////////////////////////////////////////////////////////////////////
    
    
    //init the cell map structures
    p_rep.pl_map.init_part_map(k_min,k_max);
    
    
    p_rep.status.data.resize(num_cells);
    p_rep.Ip.data.resize(num_parts);
    
    Cell_index curr_cell_index;
    Cell_id curr_cell_id;
    
    int counter = 0;
    
    int num_parts_in_cell=0;
    
    //loop over all the structures
    Cell_id curr_cell,prev_cell;
    
    float k_factor;
    
    int cell_counter = 0;


    //Need to re-establish the partcell structures; relies on the intensities being in order of the part cells
    for(int i = 0; i < num_parts; i++){
        
        curr_cell.k = k[i];
        
        k_factor = 4*pow(2,k_max - curr_cell.k);
        
        curr_cell.x = ceil(x[i]/k_factor);
        curr_cell.y = ceil(y[i]/k_factor);
        curr_cell.z = ceil(z[i]/k_factor);
        
        if (curr_cell == prev_cell){
            //continue until you have reached the end of the cell
            
            num_parts_in_cell++;
            
        } else {
            prev_cell = curr_cell;
            //add the cell to the required datastructures
            
            if (type[i] == 1) {
                p_rep.status.data[cell_counter] = 2;
            } else if (type[i]==2){
                p_rep.status.data[cell_counter] = 4;
                
            }  else if (type[i]==3){
                p_rep.status.data[cell_counter] = 5;
            }  else if (type[i]==4){
                p_rep.status.data[cell_counter] = 0;
            }

            curr_cell_index.first = counter;
            curr_cell_index.last = counter + num_parts_in_cell;
            curr_cell_index.cindex = cell_counter;
            
            p_rep.pl_map.add_cell_to_map(curr_cell,curr_cell_index);
            
            counter += num_parts_in_cell;
            num_parts_in_cell = 0;
            cell_counter ++;
            counter ++;


            
        }
        
        
        
    }

    //Now leave the xyz data computed as;
    p_rep.Ip.data = Ip;
    
    p_rep.create_uint16_dataset("x_coords", p_rep.num_parts);
    p_rep.create_uint16_dataset("y_coords", p_rep.num_parts);
    p_rep.create_uint16_dataset("z_coords", p_rep.num_parts);
    
    Part_data<uint16_t>* x_coords = (p_rep.get_data_ref<uint16_t>("x_coords"));
    Part_data<uint16_t>* y_coords = (p_rep.get_data_ref<uint16_t>("y_coords"));
    Part_data<uint16_t>* z_coords = (p_rep.get_data_ref<uint16_t>("z_coords"));
    

    x_coords->data = x;
    y_coords->data = y;
    z_coords->data = z;



    std::cout << "Load Complete" << std::endl;
    
    
}
void read_cells_hdf5(Part_rep& p_rep,std::string file_name){
    //
    //
    //  Bevan Cheeseman 2015
    //
    //  Read a full type dataset back in ; shouldn't really be done; because it requires a bit of heavy lifting to figure out what the part rep was to begin with.
    //
    //
    
    std::cout << "Loading in cell dataset: " << file_name << std::endl;
    
    //containers for all the variables
    std::vector<uint16_t> x,y,z;
    std::vector<uint8_t> k,type;
    
    
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    //  Get the attributes
    //
    //
    /////////////////////////////////////////////////////////////////////////////////////////
    
    
    //hdf5 inits
    hid_t fid, pr_groupid, obj_id,attr_id;
    H5G_info_t info;
    
    
    int num_parts,num_cells;
    float comp_scale;
    int k_min,k_max;
    //Opening shit up
    
    fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    
    //Get the group you want to open
    
    pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
    H5Gget_info( pr_groupid, &info );
    
    //Getting an attribute
    obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);
    
    //Load the attributes
#ifdef data_2D
    
    //removed this slice code
    //attr_id = 	H5Aopen(obj_id,"z",H5P_DEFAULT);
    //H5Aread(attr_id,H5T_NATIVE_INT,&part_list->slice) ;
    //H5Aclose(attr_id);
#endif
    
    /////////////////////////////////////////////
    //  Get metadata
    //
    //////////////////////////////////////////////
    
    
    
    p_rep.org_dims.resize(3);
    
    attr_id = 	H5Aopen(pr_groupid,"y_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[0]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"x_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[1]) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"z_num",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&p_rep.org_dims[2]) ;
    H5Aclose(attr_id);
    
    
    attr_id = 	H5Aopen(pr_groupid,"num_parts",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_parts) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"num_cells",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&num_cells) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"k_max",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&k_max) ;
    H5Aclose(attr_id);
    
    attr_id = 	H5Aopen(pr_groupid,"k_min",H5P_DEFAULT);
    H5Aread(attr_id,H5T_NATIVE_INT,&k_min) ;
    H5Aclose(attr_id);
    
    k.resize(num_cells);
    x.resize(num_cells);
    y.resize(num_cells);
    z.resize(num_cells);
    type.resize(num_cells);

    
    //assign the values
    p_rep.num_parts = num_parts;
    
    
    std::cout << "Number particles: " << num_parts << " Number Cells: " << num_cells << std::endl;
    
    
    //Load the data then update the particle dataset
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,x.data(),"x");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,y.data(),"y");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT16,z.data(),"z");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT8,type.data(),"type");
    
    hdf5_load_data(obj_id,H5T_NATIVE_UINT8,k.data(),"k");
    
    
    //close shiz
    H5Gclose(obj_id);
    H5Gclose(pr_groupid);
    H5Fclose(fid);
    
    std::cout << "Data loaded: now decoding and moving to data structures" << std::endl;
    
    //////////////////////////////////////////////////////////////////////////
    //
    //  Load stuff in to particle structure and decode intensity
    //
    ///////////////////////////////////////////////////////////////////////////////
    
    
    //init the cell map structures
    p_rep.pl_map.init_part_map(k_min,k_max);
    
    
    p_rep.status.data.resize(num_cells);
    p_rep.Ip.data.resize(num_parts);
    
    Cell_index curr_cell_index;

    
    //loop over all the structures
    Cell_id curr_cell,prev_cell;
    
    
    //Need to re-establish the partcell structures; relies on the intensities being in order of the part cells
    for(int i = 0; i < num_cells; i++){
        
        curr_cell.k = k[i];
        
        curr_cell.x = x[i];
        curr_cell.y = y[i];
        curr_cell.z = z[i];
        
        curr_cell_index.first = -1;
        curr_cell_index.last = -1;
        curr_cell_index.cindex = i;
        
        if (type[i] == 1) {
            p_rep.status.data[i] = 2;
        } else if (type[i]==2){
            p_rep.status.data[i] = 4;
            
        }  else if (type[i]==3){
            p_rep.status.data[i] = 5;
        }  else if (type[i]==4){
            p_rep.status.data[i] = 0;
        }
        
        p_rep.pl_map.add_cell_to_map(curr_cell,curr_cell_index);
        
        
        
    }
    
    std::cout << "Load Complete" << std::endl;
    
    
}



#endif
