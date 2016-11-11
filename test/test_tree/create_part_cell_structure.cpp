/////////////////////
//
//  Loads in and creates test datasets
//
//
/////////////////

#include "create_part_cell_structure.hpp"



void CreateSphereTest::SetUp(){

    std::string name = "files/partcell_files/test_sphere1_pcstruct_part.h5";

    create_test_dataset_from_hdf5(particle_map,pc_struct,name);
    
}

void CreateMembraneTest::SetUp(){
    
    std::string name = "files/partcell_files/membrane_pcstruct_part.h5";
    
    create_test_dataset_from_hdf5(particle_map,pc_struct,name);
    
}
