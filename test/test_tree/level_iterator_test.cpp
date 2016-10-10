//
// Created by msusik on 04.10.16.
//

#include "tree_fixtures.hpp"
#include "../../src/data_structures/Tree/Tree.hpp"
#include "../../src/data_structures/Tree/LevelIterator.hpp"


TEST_F(CreateSmallTreeTest, LEVEL_ITERATOR_SMALL_TEST)
{

    Tree<float> tree(particle_map, tree_mem, contents_mem);
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();

    int counter = 0;

    for(LevelIterator<float> it(tree, 2); it != it.end(); it++)
    {
        counter++;
        ASSERT_EQ(it.get_current_coords().y % 4, 0);
        ASSERT_EQ(it.get_current_coords().x % 4, 0);
        ASSERT_EQ(it.get_current_coords().z % 4, 0);
        ASSERT_EQ(it.get_current_coords().y % 8, 4);
        ASSERT_EQ(it.get_current_coords().x % 8, 4);
        ASSERT_EQ(it.get_current_coords().z % 8, 4);
    }

    ASSERT_EQ(counter, 8);

    bool _406 = false;
    bool _509 = false;

    for(LevelIterator<float> it(tree, 3); it != it.end(); it++)
    {
        ASSERT_EQ(it.get_current_coords().y % 2, 0);
        ASSERT_EQ(it.get_current_coords().x % 2, 0);
        ASSERT_EQ(it.get_current_coords().z % 2, 0);
        counter++;

        if(raw_content[raw_tree[*it + 2]].intensity == 406)
        {
            // taken cell, coordinates(x,y,z) : 2,6,6
            _406 = true;
        }

        if(raw_content[raw_tree[*it + 2] + 8].intensity == 509)
        {
            // taken cell, coordinates(x,y,z) : 7,5,7
            _509 = true;
        }

    }

    ASSERT_TRUE(_406);
    ASSERT_TRUE(_509);
    ASSERT_EQ(counter, 32);
}


TEST_F(CreateBigTreeTest, LEVEL_ITERATOR_BIG_TEST)
{
    Tree<float> tree(particle_map, tree_mem, contents_mem);
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();

    int counter = 0;

    for(LevelIterator<float> it(tree, 2); it != it.end(); it++)
    {
        counter++;
        ASSERT_EQ(it.get_current_coords().y % 8, 0);
        ASSERT_EQ(it.get_current_coords().x % 8, 0);
        ASSERT_EQ(it.get_current_coords().z % 8, 0);
        ASSERT_EQ(it.get_current_coords().y % 16, 8);
        ASSERT_EQ(it.get_current_coords().x % 16, 8);
        ASSERT_EQ(it.get_current_coords().z % 16, 8);
    }

    ASSERT_EQ(counter, 8);

    for(LevelIterator<float> it(tree, 1); it != it.end(); it++)
    {
        ASSERT_EQ(it.get_current_coords().y, 16);
        ASSERT_EQ(it.get_current_coords().x, 16);
        ASSERT_EQ(it.get_current_coords().z, 16);
    }

    for(LevelIterator<float> it(tree, 3); it != it.end(); it++)
    {
        counter++;
    }

    ASSERT_EQ(counter, 32);

    bool _8 = false;
    bool _4087 = false;
    bool _286 = false;

    for(LevelIterator<float> it(tree, 4); it != it.end(); it++)
    {
        if(raw_content[raw_tree[*it + 2]].intensity == 8)
        {
            _8 = true;
        }

        if(raw_content[raw_tree[*it + 7]].intensity == 286)
        {
            _286 = true;
        }

        if(raw_content[raw_tree[*it + 8]].intensity == 4087)
        {
            _4087 = true;
        }
    }

}

TEST_F(CreateNarrowTreeTest, LEVEL_ITERATOR_NARROW_TEST)
{
    Tree<float> tree(particle_map, tree_mem, contents_mem);
    Content *raw_content = tree.get_raw_content();
    uint64_t *raw_tree = tree.get_raw_tree();

    int counter = 0;

    for(LevelIterator<float> it(tree, 5); it != it.end(); it++)
    {
        counter++;
        ASSERT_EQ(it.get_current_coords().x, 4);
        ASSERT_EQ(it.get_current_coords().y, 4);
        ASSERT_LT(3, it.get_current_coords().z);
        ASSERT_LT(it.get_current_coords().z, 69);
    }

    ASSERT_EQ(counter, 9);

    bool _71 = false;

    for(LevelIterator<float> it(tree, 6); it != it.end(); it++)
    {
        if(raw_content[raw_tree[*it + 2] + 3].intensity == 71)
        {
            _71 = true;
        }
    }

    ASSERT_TRUE(_71);
}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}