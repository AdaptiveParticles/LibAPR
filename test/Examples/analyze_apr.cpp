//
// Created by cheesema on 23/01/17.
//

#include "analyze_apr.h"

#include "../../src/algorithm/apr_pipeline.hpp"

#include "../../src/analysis/apr_analysis.h"


int main(int argc, char **argv) {
    //input parsing
    cmdLineOptions options;

    //init structure
    PartCellStructure<float,uint64_t> pc_struct;

    get_apr(argc,argv,pc_struct,options);

    //read in original image
    Mesh_data<uint16_t> input_image;

    load_image_tiff(input_image, options.directory + options.input);

    compare_reconstruction_to_original(input_image,pc_struct,options);


}

