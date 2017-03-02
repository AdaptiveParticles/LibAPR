//
// Created by cheesema on 21/02/17.
//

#ifndef PARTPLAY_CREATE_SLICE_EPS_HPP
#define PARTPLAY_CREATE_SLICE_EPS_HPP

#include <algorithm>
#include <iostream>

#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellBase.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/data_structures/Tree/ParticleDataNew.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <utility>
#include "Board.h"
#include <sstream>
#include <string>

#include "create_eps/get_part_eps.hpp"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    unsigned int slice = 0;
    bool stats_file = false;
    std::string name = "name";
    float min = 100;
    float max = 2000;
    std::string input_slice = "";
};


#endif //PARTPLAY_CREATE_SLICE_EPS_HPP
