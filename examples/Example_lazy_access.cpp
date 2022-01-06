//
// Created by joel on 06.01.22.
//
const char* usage = R"(
Example showing iteration with lazy access to both particle coordinates and values

Usage:

(using *.apr output of Example_get_apr)

Example_lazy_access -i input_apr_file -d directory

Note: There is no output, this file is best utilized by looking at the source code for an example of how to lazily access APR files.
)";

#include "Example_lazy_access.hpp"


int main(int argc, char **argv) {

    // INPUT PARSING
    cmdLineOptions options = read_command_line_options(argc, argv);
    std::string file_name = options.directory + options.input;

    APRTimer timer(true);

    timer.start_timer("initialization");

    // open APR file for reading
    APRFile aprFile;
    aprFile.set_read_write_tree(false);
    aprFile.open(file_name, "READ");

    // initialize lazy access and iterator for spatial information
    LazyAccess access;
    access.init(aprFile);
    access.open();
    LazyIterator lazy_it(access);

    // initialize lazy particle data
    LazyData<uint16_t> lazy_parts;
    lazy_parts.init_file(aprFile, "particles", true);
    lazy_parts.open();

    timer.stop_timer();


    timer.start_timer("lazy iteration (load by slice)");

    // (optional) preallocate buffers for lazily loaded data
    const uint64_t xy_num = lazy_it.x_num(lazy_it.level_max()) * lazy_it.y_num(lazy_it.level_max());
    lazy_it.set_buffer_size(xy_num);
    lazy_parts.set_buffer_size(xy_num);

    // store the result in memory
    ParticleData<uint16_t> result(lazy_it.total_number_particles());

    for(int level = lazy_it.level_max(); level > lazy_it.level_min(); --level) {
        for(int z = 0; z < lazy_it.z_num(level); ++z) {

            // load slice data
            auto slice_range = lazy_it.get_slice_range(level, z);
            lazy_it.load_range(slice_range.begin, slice_range.end);
            lazy_parts.load_range(slice_range.begin, slice_range.end);

            for(int x = 0; x < lazy_it.x_num(level); ++x) {
                for(lazy_it.begin(level, z, x); lazy_it < lazy_it.end(); ++lazy_it) {
                    result[lazy_it] = lazy_parts[lazy_it] + 3;
                }
            }
        }
    }

    timer.stop_timer();


    timer.start_timer("lazy iteration (load by row)");

    for(int level = lazy_it.level_max(); level > lazy_it.level_min(); --level) {
        for(int z = 0; z < lazy_it.z_num(level); ++z) {
            for(int x = 0; x < lazy_it.x_num(level); ++x) {

                // load row data
                auto row_range = lazy_it.get_row_range(level, z, x);
                lazy_it.load_range(row_range.begin, row_range.end);
                lazy_parts.load_range(row_range.begin, row_range.end);

                for(lazy_it.begin(level, z, x); lazy_it < lazy_it.end(); ++lazy_it) {
                    result[lazy_it] = lazy_parts[lazy_it] + 3;
                }
            }
        }
    }

    timer.stop_timer();


    // close files
    access.close();
    lazy_parts.close();
    aprFile.close();

    return 0;
}



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
        std::cerr << "Usage: \"Example_lazy_access -i input_apr_file -d directory\"" << std::endl;
        std::cerr << usage << std::endl;
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

    return result;
}