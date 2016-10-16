//
//
//  Part Play Library
//
//  Bevan Cheeseman 2015
//
//  write_parts.h
//
//
//  Created by cheesema on 11/19/15.
//
//  This header contains the functions for writing the particle representation with also a xdmf file, such that it can be loaded into Paraview or similar
//
//

#ifndef PARTPLAY_WRITE_PARTS_H
#define PARTPLAY_WRITE_PARTS_H

#include "hdf5functions.h"
#include "../data_structures/structure_parts.h"
#include "../data_structures/Tree/Tree.hpp"
#include "../data_structures/Tree/LevelIterator.hpp"

#include <fstream>
#include <vector>


void write_apr_full_format(Part_rep& p_rep,Tree<float>& tree,std::string save_loc,std::string file_name);

void write_full_xdmf_xml(std::string save_loc, std::string file_name, int num_parts);

void write_full_xdmf_xml_extra(std::string save_loc, std::string file_name, int num_parts,
                               std::vector<std::string> extra_data_names, std::vector<std::string> extra_data_types);

void write_part_data_to_hdf5(Data_manager& p_rep,hid_t obj_id, std::vector<std::string>& extra_data_type,
                             std::vector<std::string>& extra_data_name, int flag_type, int req_size);

void write_apr_to_hdf5(Part_rep& p_rep, std::string save_loc, std::string file_name);
void write_apr_to_hdf5_inc_extra_fields(Part_rep& p_rep, std::string save_loc, std::string file_name);
void write_full_xdmf_xml(std::string save_loc, std::string file_name, int num_parts);
void write_full_xdmf_xml_extra(std::string save_loc, std::string file_name, int num_parts,
                               std::vector<std::string> extra_data_names, std::vector<std::string> extra_data_types);
void write_xdmf_xml_only_extra(std::string save_loc, std::string file_name, int num_parts,
                               std::vector<std::string> extra_data_names, std::vector<std::string> extra_data_types);

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Writing the particle cell structures to file, readable by paraview.
//
//////////////////////////////////////////////////////////////////////////////////////////////////
void write_part_cells_to_hdf5(Part_rep& p_rep, std::string save_loc, std::string file_name);
void write_qcompress_to_hdf5(Part_rep p_rep, std::string save_loc, std::string file_name);
void write_nocompress_to_hdf5(Part_rep p_rep, std::string save_loc, std::string file_name);
void write_part_data_to_hdf5(Data_manager& p_rep, hid_t obj_id, std::vector<std::string>& extra_data_type,
                             std::vector<std::string>& extra_data_name, int flag_type, int req_size);

#endif