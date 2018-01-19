//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// Examples of simple iteration an access to Particle Cell, and particle information. (See Example_neigh, for neighbor access)
///
/// Usage:
///
/// (using output of Example_get_apr)
///
/// Example_apr_iterate -i input_image_tiff -d input_directory
///
/////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>

#include "benchmarks/development/Example_newstructures.h"





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
        std::cerr << "Usage: \"Example_apr_iterate -i input_apr_file -d directory\"" << std::endl;
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
template<typename T>
void compare_two_maps(APR<T>& apr,APRAccess& aa1,APRAccess& aa2){


    uint64_t z_;
    uint64_t x_;

    //first compare the size

    for(uint64_t i = (apr.level_min());i <= apr.level_max();i++) {

        const unsigned int x_num_ = aa1.x_num[i];
        const unsigned int z_num_ = aa1.z_num[i];
        const unsigned int y_num_ = aa1.y_num[i];

        for (z_ = 0; z_ < z_num_; z_++) {
            for (x_ = 0; x_ < x_num_; x_++) {
                const uint64_t offset_pc_data = x_num_ * z_ + x_;


                if(aa1.gap_map.data[i][offset_pc_data].size()!=aa2.gap_map.data[i][offset_pc_data].size()) {
                    std::cout << "number of maps size mismatch" << std::endl;
                }

                if(aa1.gap_map.data[i][offset_pc_data].size()>0){
                    if(aa1.gap_map.data[i][offset_pc_data][0].map.size()!=aa2.gap_map.data[i][offset_pc_data][0].map.size()) {
                        std::cout << "number of gaps size mismatch" << std::endl;
                    }
                }


                if(aa1.gap_map.data[i][offset_pc_data].size()>0) {

                    std::vector<uint16_t> y_begin;
                    std::vector<uint16_t> y_end;
                    std::vector<uint64_t> global_index;

                    for (auto const &element : aa1.gap_map.data[i][offset_pc_data][0].map) {
                        y_begin.push_back(element.first);
                        y_end.push_back(element.second.y_end);
                        global_index.push_back(element.second.global_index_begin);
                    }

                    std::vector<uint16_t> y_begin2;
                    std::vector<uint16_t> y_end2;
                    std::vector<uint64_t> global_index2;

                    for (auto const &element : aa2.gap_map.data[i][offset_pc_data][0].map) {
                        y_begin2.push_back(element.first);
                        y_end2.push_back(element.second.y_end);
                        global_index2.push_back(element.second.global_index_begin);
                    }

                    for (int j = 0; j < y_begin.size(); ++j) {

                        if(y_begin[j]!=y_begin2[j]){
                            std::cout << "ybegin broke" << std::endl;
                        }

                        if(y_end[j]!=y_end2[j]){
                            std::cout << "ybegin broke" << std::endl;
                        }

                        if(global_index[j]!=global_index2[j]){
                            std::cout << "index broke" << std::endl;
                        }

                    }


                }

            }
        }
    }






}


template<typename T>
void create_neighbour_checker(APR<T>& apr,std::vector<MeshData<uint64_t>>& tree_rep){

    tree_rep.resize((apr.level_max()+1));

    for (int j = apr.level_min(); j <= apr.level_max(); ++j) {
            tree_rep[j].initialize(apr.spatial_index_y_max(j),apr.spatial_index_x_max(j),apr.spatial_index_z_max(j),0);
    }

    uint64_t counter = 0;

    for (apr.begin(); apr.end()!=0;apr.it_forward()) {

        tree_rep[apr.level()].access_no_protection(apr.y(),apr.x(),apr.z())=counter;

        uint64_t temp = tree_rep[apr.level()].access_no_protection(apr.y(),apr.x(),apr.z());

        counter++;

    }

}

template<typename T>
bool check_neighbours(APR<T>& apr,APRIteratorNew<T>& current,APRIteratorNew<T>& neigh){


    bool success = true;

    if(abs((float)neigh.level() - (float)current.level())>1){
        success = false;
    }

    float delta_x = current.x_global() - neigh.x_global();
    float delta_y = current.y_global() - neigh.y_global();
    float delta_z = current.z_global() - neigh.z_global();

    float resolution_max = 1.11*(0.5*pow(2,current.level_max()-current.level()) + 0.5*pow(2,neigh.level_max()-neigh.level()));

    float distance = sqrt(pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2));

    if(distance > resolution_max){
        success = false;
    }

    return success;
}

template<typename T>
bool check_neighbour_out_of_bounds(APRIteratorNew<T>& current,uint8_t face){


    uint64_t num_neigh = current.number_neighbours_in_direction(face);

    if(num_neigh ==0){
        ParticleCell neigh = current.get_neigh_particle_cell();

        if( (neigh.x >= current.spatial_index_x_max(neigh.level) ) | (neigh.y >= current.spatial_index_y_max(neigh.level) ) | (neigh.z >= current.spatial_index_z_max(neigh.level) )  ){
            return true;
        } else {
            return false;
        }
    }

    return true;
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
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    apr.parameters.input_dir = options.directory;

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end()-3,name.end());

    //APRAccess apr_access;


    //just run old code and initialize it there
    //apr_access.test_method(apr);

    /////
    //
    //  Now new data-structures
    //
    /////

    APRAccess apr_access2;
    std::vector<std::vector<uint8_t>> p_map;

    timer.start_timer("generate pmap");
    apr_access2.generate_pmap(apr,p_map);
    timer.stop_timer();

    timer.start_timer("generate map structure");
    apr_access2.initialize_structure_from_particle_cell_tree(apr,p_map);
    timer.stop_timer();

    //compare_two_maps(apr,apr_access,apr_access2);

    APRIteratorNew<uint16_t> apr_iterator(apr_access2);

    ExtraParticleData<uint16_t> particles_int;
    ExtraParticleData<uint16_t> x;
    ExtraParticleData<uint16_t> y;
    ExtraParticleData<uint16_t> z;
    ExtraParticleData<uint16_t> level;
    ExtraParticleData<uint64_t> indexd;

    uint64_t counter = 0;

    for (apr.begin(); apr.end()!=0;apr.it_forward()) {
        particles_int.data.push_back(apr(apr.particles_int));
        x.data.push_back(apr.x());
        y.data.push_back(apr.y());
        z.data.push_back(apr.z());
        level.data.push_back(apr.level());
        indexd.data.push_back(counter);
        counter++;
    }

    counter = 0;


    timer.start_timer("normal");

    std::vector<MeshData<uint64_t>> tree_rep;
    create_neighbour_checker(apr,tree_rep);

    std::cout << counter << std::endl;

    APRIteratorNew<uint16_t> neighbour_iterator(apr_access2);

    bool success = true;

    counter = 0;
    uint64_t particle_number;
    for (particle_number = 0; particle_number < apr_iterator.total_number_parts(); ++particle_number) {

        apr_iterator.set_iterator_to_particle_by_number(particle_number);
        //counter++;

        float temp = 0;

       //loop over all the neighbours and set the neighbour iterator to it
        for (int direction = 0; direction < 6; ++direction) {
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
            apr_iterator.find_neighbours_in_direction(direction);

            success = check_neighbour_out_of_bounds(apr_iterator,direction);

            for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                uint64_t this_index = tree_rep[apr_iterator.level()].access_no_protection(apr_iterator.y(),apr_iterator.x(),apr_iterator.z());

                // on each face, there can be 0-4 neighbours accessed by index
                if(neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)){
                    //will return true if there is a neighbour defined

                    uint64_t check_index = tree_rep[neighbour_iterator.level()].access_no_protection(neighbour_iterator.y(),neighbour_iterator.x(),neighbour_iterator.z());

                    uint64_t check_2 = neighbour_iterator(indexd);

                    uint16_t x_n = neighbour_iterator(x);
                    uint16_t y_n = neighbour_iterator(y);
                    uint16_t z_n = neighbour_iterator(z);
                    uint16_t level_n = neighbour_iterator(level);

                    if(check_index!=neighbour_iterator.global_index()){
                        success = false;
                    }

                    if(!check_neighbours(apr,apr_iterator,neighbour_iterator)){
                        success = false;
                    }

                    counter++;
                }

            }

        }
    }

    if(!success){
        std::cout << "BROKEN" << std::endl;
    } else {
        std::cout << "PASSING" << std::endl;
    }

    std::cout << counter << std::endl;

    timer.stop_timer();


//    timer.start_timer("parallel");
//
//    uint64_t particle_number;
//#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator) reduction(+:counter)
//    for (particle_number = 0; particle_number < apr_iterator.total_number_parts(); ++particle_number) {
//
//        apr_iterator.set_iterator_to_particle_by_number(particle_number);
//        counter++;
//
//        if(apr_iterator.y() != apr_iterator(y)){
//            std::cout << "broken y" << std::endl;
//        }
//
//        if(apr_iterator.x() != apr_iterator(x)){
//            std::cout << "broken x" << std::endl;
//        }
//
//        if(apr_iterator.z() != apr_iterator(z)){
//            std::cout << "broken z" << std::endl;
//        }
//
//        if(apr_iterator.level() != apr_iterator(level)){
//            std::cout << "broken level" << std::endl;
//        }
//
//    }
//
//    std::cout << counter << std::endl;
//
//    timer.stop_timer();
//
//
//
//    timer.start_timer("by level");
//
//    counter = 0;
//
//    for (uint64_t level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
//
//#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator) reduction(+:counter)
//        for (particle_number = apr_iterator.particles_level_begin(level); particle_number <  apr_iterator.particles_level_end(level); ++particle_number) {
//            //
//            //  Parallel loop over level
//            //
//            apr_iterator.set_iterator_to_particle_by_number(particle_number);
//
//            counter++;
//
//            if(apr_iterator.level() == level){
//
//            } else{
//                std::cout << "broken" << std::endl;
//            }
//        }
//    }
//
//    timer.stop_timer();
//
//    std::cout << counter << std::endl;
//
//    timer.start_timer("by level and z");
//
//    counter = 0;
//    uint64_t x_,z_;
//
//    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
//#pragma omp parallel for schedule(static) private(particle_number,z_) firstprivate(apr_iterator) reduction(+:counter)
//        for ( z_ = 0; z_ < apr.spatial_index_z_max(level); ++z_) {
//
//            for (particle_number = apr_iterator.particles_z_begin(level,z_);
//                 particle_number < apr_iterator.particles_z_end(level,z_); ++particle_number) {
//                //
//                //  Parallel loop over level
//                //
//                apr_iterator.set_iterator_to_particle_by_number(particle_number);
//
//                counter++;
//
//                if (apr_iterator.z() == z_) {
//
//                } else {
//                    std::cout << "broken" << std::endl;
//                }
//            }
//        }
//    }
//
//    timer.stop_timer();
//
//    std::cout << counter << std::endl;
//
//    timer.start_timer("by level the z then x");
//
//    counter = 0;
//
//
//    for (uint16_t level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
//#pragma omp parallel for schedule(static) private(particle_number,z_,x_) firstprivate(apr_iterator) reduction(+:counter)
//        for ( z_ = 0; z_ < apr.spatial_index_z_max(level); ++z_) {
//            for ( x_ = 0; x_ < apr.spatial_index_x_max(level); ++x_) {
//
//                for (particle_number = apr_iterator.particles_zx_begin(level, z_, x_);
//                     particle_number < apr_iterator.particles_zx_end(level, z_, x_); ++particle_number) {
//                    //
//                    //  Parallel loop over level
//                    //
//                    apr_iterator.set_iterator_to_particle_by_number(particle_number);
//
//                    counter++;
//
//                    if (apr_iterator.x() == x_) {
//
//                    } else {
//                        std::cout << "broken" << std::endl;
//                    }
//                }
//            }
//        }
//    }
//
//    std::cout << counter << std::endl;
//
//    timer.stop_timer();




//    MapStorageData map_data;
//
//    compare_two_maps(apr,apr_access,apr_access2);
//
//    timer.start_timer("flatten");
//    apr_access2.flatten_structure(apr,map_data);
//    timer.stop_timer();
//
//    std::cout << apr_access2.total_number_parts << std::endl;
//    std::cout << apr_access2.total_number_gaps << std::endl;
//    std::cout << apr_access2.total_number_non_empty_rows << std::endl;
//
//    APRAccess apr_access3;
//
//    apr_access3.total_number_non_empty_rows = apr_access2.total_number_non_empty_rows;
//    apr_access3.total_number_gaps = apr_access2.total_number_gaps;
//    apr_access3.total_number_parts = apr_access2.total_number_parts;
//
//    apr_access3.x_num = apr_access2.x_num;
//    apr_access3.y_num = apr_access2.y_num;
//    apr_access3.z_num = apr_access2.z_num;
//
//    apr_access3.rebuild_map(apr,map_data);
//
//    compare_two_maps(apr,apr_access,apr_access3);

}

