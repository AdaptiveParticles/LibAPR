//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
const char* usage = R"(
Example setting the APR iterator using random access

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_random_accesss -i input_apr_hdf5 -d input_directory

Note: There is no output, this file is best utilized by looking at the source code for example (test/Examples/Example_random_access.cpp) of how to code different
random access strategies on the APR.

)";


#include <algorithm>
#include <iostream>

#include "Example_random_access.hpp"
#include "io/APRFile.hpp"


int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR apr;

    timer.start_timer("full read");
    //read file
    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(apr);

    ParticleData<uint16_t> parts;
    aprFile.read_particles(apr,"particle_intensities",parts);

    aprFile.close();


    timer.stop_timer();

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end() - 3, name.end());

    auto apr_iterator = apr.iterator();

    ///////////////////////
    ///
    /// Set the iterator using random access by particle cell spatial index (x,y,z) and particle cell level
    ///
    ////////////////////////

    srand (time(NULL));

    std::cout << "Search for a Particle Cell that may not exist at random (x,y,z) and level = level_max" << std::endl;
    std::cout << "--------------------" << std::endl;

    ParticleCell random_particle_cell;
    random_particle_cell.x = (uint16_t)(apr.org_dims(1)-1)*((rand() % 10000)/10000.0f);
    random_particle_cell.y = (uint16_t)(apr.org_dims(0)-1)*((rand() % 10000)/10000.0f);
    random_particle_cell.z = (uint16_t)(apr.org_dims(2)-1)*((rand() % 10000)/10000.0f);
    random_particle_cell.level = apr.level_max();

    bool found = apr_iterator.set_iterator_by_particle_cell(random_particle_cell);

    if(!found){
        std::cout << "Particle Cell doesn't exist!" << std::endl;
    } else {
        std::cout << "Particle Cell exists with global index (particle number): " << random_particle_cell.global_index << " and has intensity value: " << parts[apr_iterator] <<  std::endl;
    }

    ///////////////////////
    ///
    /// Set the iterator using random access by using a global co-ordinate (in original pixels), and setting the iterator, to the Particle Cell that contains the point in its spatial domain.
    ///
    ////////////////////////

    for (int i = 0; i < 10; ++i) {

        float x = (apr.org_dims(1) - 1) * ((rand() % 10000) / 10000.0f);
        float y = (apr.org_dims(0) - 1) * ((rand() % 10000) / 10000.0f);
        float z = (apr.org_dims(2) - 1) * ((rand() % 10000) / 10000.0f);


        found = apr_iterator.set_iterator_by_global_coordinate(x, y, z);

        std::cout << std::endl;
        std::cout << "Searching for Particle Cell thats spatial domain contains (x,y,z)=(" << x << "," << y << "," << z << ") " << std::endl;
        std::cout << "--------------------" << std::endl;

        if(!found){
            std::cout << "out of bounds" << std::endl;
        } else {
            std::cout << "Particle Cell found is at level: " << apr_iterator.level() << " with x: " << apr_iterator.x()
                      << " y: " << apr_iterator.y() << " z: " << apr_iterator.z() << std::endl;
            std::cout << " with global index: " << apr_iterator << " and intensity "
                      << parts[apr_iterator] << std::endl;
        }
    }



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
        std::cerr << "Usage: \"Example_random_access -i input_apr_file -d directory\"" << std::endl;
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

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;

}
