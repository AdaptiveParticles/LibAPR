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

    std::cout << "Find Files" << std::endl;

    benchmark_settings bs;
    SynImage syn_image;

    AnalysisData analysis_data(options.description, "real data benchmarks", argc, argv);


    process_input(options,syn_image,analysis_data,bs);

    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////



    int N_repeats = 1;

    Part_timer b_timer;
    b_timer.verbose_flag = true;

    std::vector<std::string> file_list = listFiles( options.directory,".tif");

    std::string path_parts = get_path("IMAGE_GEN_PATH");
    std::string path_image = get_path("IMAGE_GEN_PATH");

    analysis_data.create_string_dataset("file_name",0);

    int N_par1 = file_list.size();

    for (int j = 0; j < N_par1; j++) {

        for (int i = 0; i < N_repeats; i++) {

            //init structure
            PartCellStructure<float,uint64_t> pc_struct;

            Part_rep part_rep;

            Mesh_data<uint16_t> input_image;

            std::string image_file_name = options.directory + file_list[j];

            std::string image_name  = file_list[j];
            image_name.erase(image_name.find_last_of("."), std::string::npos);

            std::string stats_file_name = image_name + "_stats.txt";

            if(check_file_exists(options.directory + stats_file_name)) {
                //check that the stats file exists

                load_image_tiff(input_image, image_file_name);

                get_image_stats(part_rep.pars, options.directory, stats_file_name);
                std::cout << "Stats file exists" << std::endl;

                get_apr(input_image,part_rep,pc_struct,analysis_data);

                produce_apr_analysis(input_image, analysis_data, pc_struct, part_rep.pars);

                pc_struct.name = image_name;

                Mesh_data<uint16_t> interp_img;
                // save pc reconstruction
                pc_struct.interp_parts_to_pc(interp_img,pc_struct.part_data.particle_data);
                write_image_tiff(input_image, path_image  + image_name + ".tif");
                // save APR
                write_apr_pc_struct(pc_struct,path_parts,image_name);
                // save APR Full
                write_apr_full_format(pc_struct,path_parts ,image_name);

                analysis_data.get_data_ref<std::string>("file_name")->data.push_back(
                        image_name);
                analysis_data.part_data_list["file_name"].print_flag = true;

            } else {
                std::cout << "Stats file doesn't exist" << std::endl;
            }

        }
    }

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;
}
