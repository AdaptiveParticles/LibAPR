//
// Created by cheesema on 21.01.18.
//
//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Example setting the APR iterator using random access
///
/// Usage:
///
/// (using output of Example_compute_gradient)
///
/// Example_random_accesss -i input_apr_hdf5 -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "Example_random_access.hpp"

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
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR <uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end() - 3, name.end());

    APRIterator<uint16_t> apr_iterator(apr);

    ///////////////////////
    ///
    /// Set the iterator using random access by particle cell spatial index (x,y,z) and particle cell level
    ///
    ////////////////////////

    std::cout << "Search for a Particle Cell that may not exist at (x,y,z,l) = (10,10,10,4)" << std::endl;
    std::cout << "--------------------" << std::endl;

    ParticleCell random_particle_cell;
    random_particle_cell.x = 10;
    random_particle_cell.y = 10;
    random_particle_cell.z = 10;
    random_particle_cell.level = 4;

    bool found = apr_iterator.set_iterator_by_particle_cell(random_particle_cell);

    if(!found){
        std::cout << "Particle Cell doesn't exist!" << std::endl;
    } else {
        std::cout << "Particle Cell exists with global index (particle number): " << random_particle_cell.global_index << " and has intensity value: " << apr.particles_intensities[apr_iterator] <<  std::endl;
    }

    std::cout << std::endl;
    std::cout << "Search for a Particle Cell that we know exists, and has global index of 1000" << std::endl;
    std::cout << "--------------------" << std::endl;

    //now lets find a Particle Cell we know exits by setting the Iterator to the 1000th particle
    apr_iterator.set_iterator_to_particle_by_number(1000);

    random_particle_cell.x = apr_iterator.x();
    random_particle_cell.y = apr_iterator.y();
    random_particle_cell.z = apr_iterator.z();
    random_particle_cell.level = apr_iterator.level();

    found = apr_iterator.set_iterator_by_particle_cell(random_particle_cell);

    if(!found){
        std::cout << "Particle Cell doesn't exist!" << std::endl;
    } else {
        std::cout << "Particle Cell exists with global index (particle number): " << random_particle_cell.global_index << " and has intensity value: " << apr.particles_intensities[apr_iterator] <<  std::endl;
    }

    ///////////////////////
    ///
    /// Set the iterator using random access by using a global co-ordinate (in original pixels), and setting the iterator, to the Particle Cell that contains the point in its spatial domain.
    ///
    ////////////////////////

    srand (time(NULL));

    float x = apr.orginal_dimensions(1)*((rand() % 10000)/10000.0f);
    float y = apr.orginal_dimensions(0)*((rand() % 10000)/10000.0f);
    float z = apr.orginal_dimensions(2)*((rand() % 10000)/10000.0f);

    found = apr_iterator.set_iterator_by_global_coordinate(x,y,z);

    std::cout << std::endl;
    std::cout << "Searching for Particle Cell thats spatial domain contains (x,y,z)=(" << x << "," << y << "," << z << ") " << std::endl;
    std::cout << "--------------------" << std::endl;

    if(!found){
        std::cout << "out of bounds" << std::endl;
    } else {
        std::cout << "Particle Cell found is at level: " << apr_iterator.level() << " with x: " << apr_iterator.x() << " y: " << apr_iterator.y() << " z: " << apr_iterator.z() << std::endl;
        std::cout << "type: " << std::to_string((uint16_t)apr_iterator.type()) << " with global index: " << apr_iterator.global_index() << " and intensity " << apr.particles_intensities[apr_iterator] << std::endl;
    }
}
