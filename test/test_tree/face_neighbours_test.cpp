


#include "tree_fixtures.hpp"
#include "benchmarks/development/Tree/Tree.hpp"
#include "benchmarks/development/Tree/LevelIterator.hpp"

#define MAX_NUM_OF_CHILDREN 4


TEST_F(CreateSmallTreeTest, NEIGHBOURS_SMALL_TEST)
{
    uint8_t expected_res_1[1] = {41};
    uint8_t expected_res_2[1] = {103};

    Tree<float> tree(particle_map, tree_mem, contents_mem);
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();

    LevelIterator<float> it(tree, 3);
    std::vector<uint64_t> neighbours(MAX_NUM_OF_CHILDREN);

    tree.get_face_neighbours(*it, 4, it.get_current_coords(), it.level_multiplier,
                        it.child_index, neighbours);

    ASSERT_EQ(neighbours.size(), 1);

    for(int i = 0; i < neighbours.size(); i++)
    {
        ASSERT_EQ(neighbours[i], expected_res_1[i]);
    }

    it++; it++;

    neighbours.resize(MAX_NUM_OF_CHILDREN);
    tree.get_face_neighbours(*it, 4, it.get_current_coords(), it.level_multiplier,
                        it.child_index, neighbours);

    ASSERT_EQ(neighbours.size(), 1);

    for(int i = 0; i < neighbours.size(); i++)
    {
        ASSERT_EQ(neighbours[i], expected_res_2[i]);
    }
}


TEST_F(CreateBigTreeTest, NEIGHBOURS_BIG_TEST)
{
    uint8_t expected_res_1[1] = {78};

    Tree<float> tree(particle_map, tree_mem, contents_mem);
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();

    LevelIterator<float> it(tree, 2);
    std::vector<uint64_t> neighbours(MAX_NUM_OF_CHILDREN);

    tree.get_face_neighbours(*it, 4, it.get_current_coords(), it.level_multiplier,
                        it.child_index, neighbours);

    ASSERT_EQ(neighbours.size(), 1);

    for(int i = 0; i < neighbours.size(); i++)
    {
        ASSERT_EQ(neighbours[i], expected_res_1[i]);
    }

    //there should be 5 of them
}


TEST_F(CreateNarrowTreeTest, NEIGHBOURS_NARROW_TEST_DOWN)
{
    uint8_t expected_res_1[2] = {174,177};

    Tree<float> tree(particle_map, tree_mem, contents_mem);
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();

    LevelIterator<float> it(tree, 5);

    for(; it != it.end(); it++)
    {
        // loop over to the last cell in the level (z == 68)

        if(it.get_current_coords().z == 68)
        {
            break;
        }
    }

    std::vector<uint64_t> neighbours(MAX_NUM_OF_CHILDREN);
    tree.get_face_neighbours(*it, 4, it.get_current_coords(), it.level_multiplier,
                        it.child_index, neighbours);

    ASSERT_EQ(neighbours.size(), 0);

}


TEST_F(CreateNarrowTreeTest, NEIGHBOURS_NARROW_TEST_UP)
{
    uint8_t expected_res_1[3] = {210};

    Tree<float> tree(particle_map, tree_mem, contents_mem);
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();

    LevelIterator<float> it(tree, 6);

    for(; it != it.end(); it++)
    {
        // loop over to the last but one cell in the level (z == 62, y == 2)

        if(it.get_current_coords().z == 62 && it.get_current_coords().y == 2)
        {
            break;
        }
    }

    std::vector<uint64_t> neighbours(MAX_NUM_OF_CHILDREN);
    tree.get_face_neighbours(*it, 5, it.get_current_coords(), it.level_multiplier,
                        it.child_index, neighbours);

    ASSERT_EQ(neighbours.size(), 1);

    for(int i = 0 ; i < neighbours.size(); i++)
    {
        ASSERT_EQ(neighbours[i], expected_res_1[i]);
    }

}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}