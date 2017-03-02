//
// Created by cheesema on 21/02/17.
//
//
//  Creates eps files of particular image slices from APR
//
//
//
//

#include "create_slice_eps.hpp"






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

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile -d directory [-t] [-o outputfile]\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-name"))
    {
        result.name = std::string(get_command_option(argv, argv + argc, "-name"));
    }


    if(command_option_exists(argv, argv + argc, "-t"))
    {
        part_rep.timer.verbose_flag = true;
    }

    if(command_option_exists(argv, argv + argc, "-slice"))
    {
        result.slice = std::stoi(std::string(get_command_option(argv, argv + argc, "-slice")));
    }

    if(command_option_exists(argv, argv + argc, "-min"))
    {
        result.min = std::stof(std::string(get_command_option(argv, argv + argc, "-min")));
    }

    if(command_option_exists(argv, argv + argc, "-max"))
    {
        result.max = std::stof(std::string(get_command_option(argv, argv + argc, "-max")));
    }

    if(command_option_exists(argv, argv + argc, "-input_slice_file"))
    {
        result.input_slice = (std::string(get_command_option(argv, argv + argc, "-input_slice_file")));
    }


    return result;

}

int main(int argc, char **argv) {

    Part_rep part_rep;

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);

    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);

    ParticlesFull parts_slice =  get_full_parts_slice(pc_struct,options.slice);

    std::vector<unsigned int> crange = {(unsigned int)options.min,(unsigned int)options.max};

    std::string save_loc = get_path("PARTGEN_OUTPUT_PATH");
    std::string name = options.name;

    create_part_eps(parts_slice,save_loc,name,crange);

    create_part_eps_type(parts_slice,save_loc,name);
    create_part_eps_depth(parts_slice,save_loc,name);

    Mesh_data<uint16_t> interp_img;

    pc_struct.interp_parts_to_pc(interp_img,pc_struct.part_data.particle_data);

    //single slice
    Mesh_data<uint16_t> slice;
    slice.initialize(interp_img.y_num,interp_img.x_num,1,0);

    for (int i = 0; i < interp_img.x_num; ++i) {
        for (int j = 0; j < interp_img.y_num; ++j) {
            slice(i,j,0) = interp_img(i,j,options.slice);
        }
    }

    //write the slice
    write_image_tiff(slice,save_loc + name + ".tif");


    if(options.input_slice != ""){

        Mesh_data<uint16_t> input_image;
        std::string input_file_name = options.input_slice;

        load_image_tiff(input_image, input_file_name);

        int num_pixels = input_image.y_num*input_image.x_num;

        ParticlesFull img_slice;

        img_slice.x_c.resize(num_pixels,0);
        img_slice.y_c.resize(num_pixels,0);
        img_slice.z_c.resize(num_pixels,0);

        img_slice.Ip.resize(num_pixels,0);
        img_slice.k_vec.resize(num_pixels,0);

        img_slice.depth_max = parts_slice.depth_max;
        img_slice.depth_min = parts_slice.depth_min;

        int counter = 0;

        int in;


        for (int i = 0; i < input_image.x_num; ++i) {
            for (int j = 0; j < input_image.y_num; ++j) {
                img_slice.x_c[counter] = 2*i;
                img_slice.y_c[counter] = 2*j;
                img_slice.Ip[counter] = input_image.mesh[counter];
                img_slice.k_vec[counter] = img_slice.depth_max+1;

                counter++;
            }
        }

        create_part_eps(img_slice,save_loc,name + "_mesh",crange);


    }


}


