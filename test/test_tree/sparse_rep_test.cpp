//
// Created by Bevan Cheeseman 3.11.2016
//

#include "tree_fixtures.hpp"
#include "create_part_cell_structure.hpp"

TEST_F(CreateSphereTest, COMPARE_PART_MAP) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //


    //test general structure
    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map, pc_struct, false));

}
TEST_F(CreateSphereTest, COMPARE_PART_MAP_NEIGH_CELL) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //


    //test general structure
    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pc_struct));

}

TEST_F(CreateSphereTest, TEST_Y_COORDS) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test y_coordinate offsets
    ASSERT_TRUE(compare_y_coords(pc_struct));

}

TEST_F(CreateSphereTest, PART_NEIGH) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test part neighbour search
    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pc_struct));

}
TEST_F(CreateSphereTest, READ_WRITE) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(read_write_structure_test(pc_struct, "1"));
}

TEST_F(CreateSphereTest, PARENT_STRUCTURE) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(parent_structure_test(pc_struct));

}
TEST_F(CreateSphereTest, FIND_CELL) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(find_part_cell_test(pc_struct));

}

TEST_F(CreateSphereTest, NEIGH_CELL_TEST) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_neigh_cells(pc_struct));

}

TEST_F(CreateSphereTest, NEIGH_PART_TEST) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_neigh_parts(pc_struct));

}

TEST_F(CreateSphereTest, NEW_STRUCTURE) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_alt_part_struct(pc_struct));

}

//TEST_F(CreateSphereTest, MOORE_NEIGHBOURHOOD) {
//
//    //test neighbour cell search
//
//
//    std::cout << "moore neighbourhood test off" << std::endl;
//
//    //ASSERT_TRUE(utest_moore_neighbours(pc_struct));
//
//    ASSERT_TRUE(true);
//
//    //utest_moore_neighbours(pc_struct);
//
//}

TEST_F(CreateMembraneTest, COMPARE_PART_MAP) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //


    //test general structure
    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map, pc_struct, false));

}
TEST_F(CreateMembraneTest, COMPARE_PART_MAP_NEIGH_CELL) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //


    //test general structure
    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pc_struct));

}

TEST_F(CreateMembraneTest, TEST_Y_COORDS) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test y_coordinate offsets
    ASSERT_TRUE(compare_y_coords(pc_struct));

}

TEST_F(CreateMembraneTest, PART_NEIGH) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test part neighbour search
    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pc_struct));

}
TEST_F(CreateMembraneTest, READ_WRITE) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(read_write_structure_test(pc_struct, "2"));
}

TEST_F(CreateMembraneTest, PARENT_STRUCTURE) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(parent_structure_test(pc_struct));

}
TEST_F(CreateMembraneTest, FIND_CELL) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(find_part_cell_test(pc_struct));

}

TEST_F(CreateMembraneTest, NEIGH_CELL_TEST) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_neigh_cells(pc_struct));

}

TEST_F(CreateMembraneTest, NEIGH_PART_TEST) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_neigh_parts(pc_struct));

}

TEST_F(CreateMembraneTest, NEW_STRUCTURE) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_alt_part_struct(pc_struct));

}


TEST_F(CreateMembraneTest, APR_SERIAL_ITERATOR) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test iteration
    ASSERT_TRUE(utest_apr_serial_iterate(pc_struct));

}

TEST_F(CreateMembraneTest, APR_SERIAL_NEIGH) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test get face neighbours
    ASSERT_TRUE(utest_apr_serial_neigh(pc_struct));

}


TEST_F(CreateMembraneTest, APR_PARALELL_ITERATOR) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_apr_parallel_iterate(pc_struct));

}

TEST_F(CreateMembraneTest, APR_PARALELL_NEIGH) {

    //
    //  Sparse Particle Structure Test Cases
    //
    //

    //test io
    ASSERT_TRUE(utest_apr_parallel_neigh(pc_struct));

}


//TEST_F(CreateMembraneTest, MOORE_NEIGHBOURHOOD) {
//
//    //test neighbour cell search
//
//    std::cout << "moore test turned off" << std::endl;
//    //ASSERT_TRUE(utest_moore_neighbours(pc_struct));
//    ASSERT_TRUE(true);
//
//}

//TEST_F(CreateMembraneTest, SPARSE_STRUCTURE_MEMBRANE_LARGE_TEST)
//{
//
//    //
//    //  Sparse Particle Structure Test Cases
//    //
//    //
//
//    //test general structure
//    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pc_struct,false));
//    //test neighbour cell search
//    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pc_struct));
//    //test y_coordinate offsets
//    ASSERT_TRUE(compare_y_coords(pc_struct));
//    //test part neighbour search
//    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pc_struct));
//    //test io
//    ASSERT_TRUE(read_write_structure_test(pc_struct));
//
//    ASSERT_TRUE(parent_structure_test(pc_struct));
//
//    ASSERT_TRUE(find_part_cell_test(pc_struct));
//
//    ASSERT_TRUE(utest_neigh_cells(pc_struct));
//
//    ASSERT_TRUE(utest_neigh_parts(pc_struct));
//
//    ASSERT_TRUE(utest_alt_part_struct(pc_struct));
//
//}





//TEST_F(CreateSmallTreeTest, SPARSE_STRUCTURE_SMALL_TEST)
//{
//    
//    //
//    //  Sparse Particle Structure Test Cases
//    //
//    //
//    
//    // Set the intensities
//    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
//        
//        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
//            particle_map.downsampled[depth].mesh[i] = (uint16_t) i;
//        }
//        
//    }
//
//    
//    PartCellStructure<float,uint64_t> pcell_test(particle_map);
//    
//    //test general structure
//    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pcell_test,true));
//    //test neighbour cell search
//    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pcell_test));
//    //test y_coordinate offsets
//    ASSERT_TRUE(compare_y_coords(pcell_test));
//    //test part neighbour search
//    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pcell_test));
//    //test io
//    ASSERT_TRUE(read_write_structure_test(pcell_test));
//    
//}
//
//
//TEST_F(CreateBigTreeTest, SPARSE_STRUCTURE_BIG_TEST)
//{
//    
//    
//    // Set the intensities
//    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
//        
//        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
//            particle_map.downsampled[depth].mesh[i] = (uint16_t) i;
//        }
//        
//    }
//    
//    
//    PartCellStructure<float,uint64_t> pcell_test(particle_map);
//    
//    //test general structure
//    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pcell_test,true));
//    //test neighbour cell search
//    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pcell_test));
//    //test y_coordinate offsets
//    ASSERT_TRUE(compare_y_coords(pcell_test));
//    //test part neighbour search
//    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pcell_test));
//    //test io
//    ASSERT_TRUE(read_write_structure_test(pcell_test));
//    
//}
//
//TEST_F(CreateNarrowTreeTest, SPARSE_STRUCTURE_NARROW_TEST)
//{
//    
//    // Set the intensities
//    for(int depth = particle_map.k_min; depth <= (particle_map.k_max+1);depth++){
//        
//        for(int i = 0; i < particle_map.downsampled[depth].mesh.size();i++){
//            particle_map.downsampled[depth].mesh[i] = (uint16_t) i;
//        }
//        
//    }
//
//    
//    
//    PartCellStructure<float,uint64_t> pcell_test(particle_map);
//    
//    //test general structure
//    ASSERT_TRUE(compare_sparse_rep_with_part_map(particle_map,pcell_test,true));
//    //test neighbour cell search
//    ASSERT_TRUE(compare_sparse_rep_neighcell_with_part_map(particle_map,pcell_test));
//    //test y_coordinate offsets
//    ASSERT_TRUE(compare_y_coords(pcell_test));
//    //test part neighbour search
//    ASSERT_TRUE(compare_sparse_rep_neighpart_with_part_map(particle_map,pcell_test));
//    //test io
//    ASSERT_TRUE(read_write_structure_test(pcell_test));
//    
//    
//}

int main(int argc, char **argv) {



    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
    
}