//
// Created by cheesema on 14/03/17.
//

#include <algorithm>
#include <iostream>

#include "Example_recon_img.h"

bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_recon_img -i input_apr_file -d directory [-o outputfile]\"" << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;

}

int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    Part_timer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR<float> apr;

    //read file
    apr.read_apr(file_name);

    //create mesh data structure for reconstruction
    MeshData<uint16_t> recon_pc;

    timer.start_timer("pc interp");
    //perform piece-wise constant interpolation
    apr.interp_img(recon_pc,apr.particles_int);

    timer.stop_timer();

    std::cout << "PC recon " << (recon_pc.x_num*recon_pc.y_num*recon_pc.z_num)/((timer.t2 - timer.t1)*1000000.0) << " million pixels per second"  <<  std::endl;

    std::string output_path = options.directory + apr.name + "_pc.tif";

    //write output as tiff
    recon_pc.write_image_tiff(output_path);

    //////////////////////////
    ///
    /// Create a particle dataset with the particle type and pc construct it
    ///
    ////////////////////////////

    //initialization of the iteration structures
    APRIterator<float> apr_it(apr); //this is required for parallel access
    uint64_t part;

    //create particle dataset
    ExtraPartCellData<float> type(apr);
    ExtraPartCellData<float> level(apr);

    timer.start_timer("APR parallel iterator loop");

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it)
    for (part = 0; part < apr.num_parts_total; ++part) {
        //needed step for any parallel loop (update to the next part)
        apr_it.set_iterator_to_particle_by_number(part);

        apr_it(type) = apr_it.type();
        apr_it(level) = apr_it.depth();
    }

    timer.stop_timer();

    MeshData<uint16_t> type_recon;

    apr.interp_img(type_recon,type);

    output_path = options.directory + apr.name + "_type.tif";

    //write output as tiff
    type_recon.write_image_tiff(output_path);

    //pc interp
    apr.interp_img(type_recon,level);

    output_path = options.directory + apr.name + "_level.tif";

    //write output as tiff
    type_recon.write_image_tiff(output_path);

    //smooth reconstruction - requires float
    MeshData<float> recon_smooth;

    std::vector<float> scale_d = {2,2,2};

    timer.start_timer("smooth reconstrution");

    apr.interp_parts_smooth(recon_smooth,apr.particles_int,scale_d);

    timer.stop_timer();

    std::cout << "Smooth recon " << (recon_smooth.x_num*recon_smooth.y_num*recon_smooth.z_num)/((timer.t2 - timer.t1)*1000000.0) << " million pixels per second"  <<  std::endl;

    output_path = options.directory + apr.name + "_smooth.tif";

    //write to tiff casting to unsigned 16 bit integer
    recon_smooth.write_image_tiff_uint16(output_path);


}