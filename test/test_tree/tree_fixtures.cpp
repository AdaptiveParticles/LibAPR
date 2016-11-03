//
// Created by msusik on 04.10.16.
//

#include "../../src/data_structures/Tree/Tree.hpp"
#include "tree_fixtures.hpp"


void CreateSmallTreeTest::SetUp()
{

    auto &layers = particle_map.layers;
    auto &downsampled = particle_map.downsampled;

    layers.resize(3);
    layers[2].x_num = layers[2].y_num = layers[2].z_num = 4;
    layers[1].x_num = layers[1].y_num = layers[1].z_num = 2;

    layers[1].mesh = {SLOPESTATUS, SLOPESTATUS, SLOPESTATUS, SLOPESTATUS, SLOPESTATUS,
                                   PARENTSTATUS, PARENTSTATUS, PARENTSTATUS};

    // warning: some fields will not be accessed, and setting Neighbourstatus to them has an undefined behaviour
    layers[2].mesh.resize(64, NEIGHBOURSTATUS);
    layers[2].mesh[40] = SLOPESTATUS;
    layers[2].mesh[44] = SLOPESTATUS;
    layers[2].mesh[55] = TAKENSTATUS;
    layers[2].mesh[56] = SLOPESTATUS;
    layers[2].mesh[60] = SLOPESTATUS;
    layers[2].mesh[62] = TAKENSTATUS;

    downsampled.resize(4);

    downsampled[3].x_num = downsampled[3].y_num = downsampled[3].z_num = 8;
    downsampled[2].x_num = downsampled[2].y_num = downsampled[2].z_num = 4;
    downsampled[1].x_num = downsampled[1].y_num = downsampled[1].z_num = 2;
    downsampled[1].mesh = {1,2,3,4,5,0,0,0};
    downsampled[2].mesh.resize(64, 1);
    downsampled[3].mesh.resize(512, 0);

    int i = 0;
    for(auto it=downsampled[3].mesh.begin(); downsampled[3].mesh.end() != it; it++)
    {
        *it = i++;
    }
    
    particle_map.k_min = 1;
    particle_map.k_max = 2;
    
    tree_mem.resize(1000);
    contents_mem.resize(500);
}

void CreateBigTreeTest::SetUp()
{
    auto &layers = particle_map.layers;
    auto &downsampled = particle_map.downsampled;

    layers.resize(4);
    layers[3].x_num = layers[3].y_num = layers[3].z_num = 8;
    layers[2].x_num = layers[2].y_num = layers[2].z_num = 4;
    layers[1].x_num = layers[1].y_num = layers[1].z_num = 2;

    layers[1].mesh = {SLOPESTATUS, PARENTSTATUS, SLOPESTATUS, SLOPESTATUS, SLOPESTATUS,
                      SLOPESTATUS, PARENTSTATUS, PARENTSTATUS};

    layers[2].mesh.resize(64, SLOPESTATUS);
    layers[2].mesh[3] = PARENTSTATUS;
    layers[2].mesh[61] = PARENTSTATUS;
    layers[2].mesh[62] = PARENTSTATUS;

    layers[3].mesh.resize(512, NEIGHBOURSTATUS);
    layers[3].mesh[7] = TAKENSTATUS;
    layers[3].mesh[507] = TAKENSTATUS;

    downsampled.resize(5);

    downsampled[4].x_num = downsampled[4].y_num = downsampled[4].z_num = 16;
    downsampled[3].x_num = downsampled[3].y_num = downsampled[3].z_num = 8;
    downsampled[2].x_num = downsampled[2].y_num = downsampled[2].z_num = 4;
    downsampled[1].x_num = downsampled[1].y_num = downsampled[1].z_num = 2;
    downsampled[1].mesh = {1,0,3,4,5,6,0,8};
    downsampled[2].mesh.resize(64, 1);
    downsampled[3].mesh.resize(512, 2);
    downsampled[4].mesh.resize(4096, 3);

    int i = 0;
    for(auto it=downsampled[4].mesh.begin(); downsampled[4].mesh.end() != it; it++)
    {
        *it = i++;
    }
    particle_map.k_min = 1;
    particle_map.k_max = 3;
    

    tree_mem.resize(10000);
    contents_mem.resize(500);
}

void CreateNarrowTreeTest::SetUp()
{
    auto &layers = particle_map.layers;
    auto &downsampled = particle_map.downsampled;

    // here, the sizes of the model are not powers of 2

    layers.resize(6);
    layers[1].x_num = layers[1].y_num = layers[1].z_num = 0;
    layers[2].x_num = layers[2].y_num = layers[2].z_num = 0;
    layers[3].x_num = layers[3].y_num = layers[3].z_num = 0;
    layers[4].x_num = layers[4].y_num = 1;
    layers[4].z_num = 9;
    layers[5].x_num = 1;
    layers[5].y_num = 2;
    layers[5].z_num = 17;

    layers[4].mesh = {SLOPESTATUS, PARENTSTATUS, PARENTSTATUS, SLOPESTATUS,
                      SLOPESTATUS, SLOPESTATUS, PARENTSTATUS, PARENTSTATUS, SLOPESTATUS};
    layers[5].mesh.resize(34, NEIGHBOURSTATUS);

    layers[5].mesh[4] = SLOPESTATUS;
    layers[5].mesh[5] = SLOPESTATUS;
    layers[5].mesh[9] = TAKENSTATUS;
    layers[5].mesh[24] = SLOPESTATUS;
    layers[5].mesh[25] = SLOPESTATUS;
    layers[5].mesh[29] = TAKENSTATUS;

    downsampled.resize(7);
    downsampled[4].mesh.resize(9,4);
    downsampled[4].x_num = downsampled[4].y_num = 1;
    downsampled[4].z_num = 9;

    downsampled[5].mesh.resize(34,5);
    downsampled[5].x_num = 1;
    downsampled[5].y_num = 2;
    downsampled[5].z_num = 17;

    downsampled[6].mesh.resize(272);
    downsampled[6].x_num = 2;
    downsampled[6].y_num = 4;
    downsampled[6].z_num = 34;
    
    particle_map.k_min = 4;
    particle_map.k_max = 5;

    int i = 0;
    for(auto it=downsampled[6].mesh.begin(); downsampled[6].mesh.end() != it; it++)
    {
        *it = i++;
    }
    tree_mem.resize(5000);
    contents_mem.resize(500);
    
}
