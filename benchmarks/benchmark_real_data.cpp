//
// Created by cheesema on 08/02/17.
//

#include "benchmark_real_data.hpp"

int main(int argc, char **argv) {

    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    std::cout << "Real Data Benchmark" << std::endl;

    cmdLineOptionsBench options = read_command_line_options(argc, argv);


    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////

    AnalysisData analysis_data(options.description, "real data benchmarks", argc, argv);

    int N_repeats = 1;

    Part_timer b_timer;
    b_timer.verbose_flag = true;

    int N_par1 = 0;

    std::vector<std::string> file_list = listFiles( options.directory,".tif");

    for (int j = 0; j < N_par1; j++) {

        for (int i = 0; i < N_repeats; i++) {

            //init structure
            PartCellStructure<float,uint64_t> pc_struct;

            get_apr(argc,argv,pc_struct,options);


        }
    }

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;
}
