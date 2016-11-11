//
// Created by Bevan Cheeseman 3.11.2016
//
#include "../utils.h"
#include "tree_fixtures.hpp"
#include "create_part_cell_structure.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"

TEST_F(CreateSphereTest, SPARSE_STRUCTURE_SPHERE_TEST)
{
    
    //
    //  Sparse Particle Structure Test Cases
    //
    //
    
   
    
    //test general structure
    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pc_struct,false));
    //test neighbour cell search
    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pc_struct));
    //test y_coordinate offsets
    ASSERT_TRUE(compare_y_coords(pc_struct));
    //test part neighbour search
    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pc_struct));
    //test io
    //ASSERT_TRUE(read_write_structure_test(pcell_test));
    
    ASSERT_TRUE(true);
    
}





TEST_F(CreateSmallTreeTest, SPARSE_STRUCTURE_SMALL_TEST)
{
    
    //
    //  Sparse Particle Structure Test Cases
    //
    //
    
    // Set the intensities
    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
        
        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
            particle_map.downsampled[depth].mesh[i] = i;
        }
        
    }

    
    PartCellStructure<float,uint64_t> pcell_test(particle_map);
    
    //test general structure
    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pcell_test,true));
    //test neighbour cell search
    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pcell_test));
    //test y_coordinate offsets
    ASSERT_TRUE(compare_y_coords(pcell_test));
    //test part neighbour search
    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pcell_test));
    //test io
    ASSERT_TRUE(read_write_structure_test(pcell_test));
    
}


TEST_F(CreateBigTreeTest, SPARSE_STRUCTURE_BIG_TEST)
{
    
    
    // Set the intensities
    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
        
        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
            particle_map.downsampled[depth].mesh[i] = i;
        }
        
    }
    
    
    PartCellStructure<float,uint64_t> pcell_test(particle_map);
    
    //test general structure
    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pcell_test,true));
    //test neighbour cell search
    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pcell_test));
    //test y_coordinate offsets
    ASSERT_TRUE(compare_y_coords(pcell_test));
    //test part neighbour search
    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pcell_test));
    //test io
    ASSERT_TRUE(read_write_structure_test(pcell_test));
    
}

TEST_F(CreateNarrowTreeTest, SPARSE_STRUCTURE_NARROW_TEST)
{
    
    // Set the intensities
    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
        
        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
            particle_map.downsampled[depth].mesh[i] = i;
        }
        
    }

    
    
    PartCellStructure<float,uint64_t> pcell_test(particle_map);
    
    //test general structure
    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pcell_test,true));
    //test neighbour cell search
    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pcell_test));
    //test y_coordinate offsets
    ASSERT_TRUE(compare_y_coords(pcell_test));
    //test part neighbour search
    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pcell_test));
    //test io
    ASSERT_TRUE(read_write_structure_test(pcell_test));
    
    
}

int main(int argc, char **argv) {
    
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    
}