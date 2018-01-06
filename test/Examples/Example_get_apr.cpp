
#include <algorithm>
#include <iostream>

#include "Example_get_apr.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"
#include "../utils.h"
#include "../../src/numerics/misc_numerics.hpp"
#include "../../src/algorithm/apr_pipeline.hpp"

int main(int argc, char **argv) {

    //input parsing
    cmdLineOptions options;
    //init structure

    APR<float> apr;

    if(get_apr(argc,argv,apr,options)){

        //output
        std::string save_loc = options.output_dir;
        std::string file_name = options.output;

        Part_timer timer;

        timer.verbose_flag = true;

        timer.start_timer("writing output");

        apr.write_apr(save_loc,file_name);

        timer.stop_timer();

    } else {
        std::cout << "Oops, something went wrong. APR not computed :(." << std::endl;
    }

}


