//
// Created by cheesema on 28.02.18.
//

//
// Created by cheesema on 28.02.18.
//

//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(


)";


#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"

#include "algorithm/APRConverter.hpp"
#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include <numerics/APRNumerics.hpp>
#include <numerics/APRComputeHelper.hpp>

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

void create_test_particles(APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,APRTreeIterator<uint16_t>& apr_tree_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,ExtraParticleData<float>& part_tree,std::vector<double>& stencil, const int stencil_size, const int stencil_half);

template<typename T,typename ParticleDataType>
void update_dense_array(const uint64_t level,const uint64_t z,APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator, APRIterator<uint16_t>& treeIterator, ExtraParticleData<float> &tree_data,MeshData<T>& temp_vec,ExtraParticleData<ParticleDataType>& particleData, const int stencil_size, const int stencil_half) {

    uint64_t x;

    const uint64_t x_num_m = temp_vec.x_num;
    const uint64_t y_num_m = temp_vec.y_num;


    uint64_t parent_number;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
    for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

        //
        //  This loop recreates particles at the current level, using a simple copy
        //

        uint64_t mesh_offset = (x + stencil_half) * y_num_m + x_num_m * y_num_m * (z % stencil_size);

        apr_iterator.set_new_lzx(level, z, x);
        for (unsigned long gap = 0;
             gap < apr_iterator.number_gaps(); apr_iterator.move_gap(gap)) {

            uint64_t y_begin = apr_iterator.current_gap_y_begin() ;
            uint64_t y_end = apr_iterator.current_gap_y_end() ;
            uint64_t index = apr_iterator.current_gap_index();

            std::copy(particleData.data.begin() + index, particleData.data.begin() + index + (y_end - y_begin) +1,
                      temp_vec.mesh.begin() + mesh_offset + y_begin + stencil_half);


        }

    }

    if (level > apr_iterator.level_min()) {
        const int y_num = apr_iterator.spatial_index_y_max(level);

        //
        //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
        //

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
        for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

            for (apr_iterator.set_new_lzx(level - 1, z / 2, x / 2);
                 apr_iterator.global_index() < apr_iterator.particles_zx_end(level - 1, z / 2,
                                                                             x /
                                                                             2); apr_iterator.set_iterator_to_particle_next_particle()) {

                int y_m = std::min(2 * apr_iterator.y() + 1, y_num-1);	// 2y+1+offset

                temp_vec.at(2 * apr_iterator.y() + stencil_half, x + stencil_half, z % stencil_size) = particleData[apr_iterator];
                temp_vec.at(y_m + stencil_half, x + stencil_half, z % stencil_size) = particleData[apr_iterator];


            }

        }
    }

    /******** start of using the tree iterator for downsampling ************/

    if (level < apr_iterator.level_max()) {
        for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
            for (treeIterator.set_new_lzx(level, z , x );
                 treeIterator.global_index() < treeIterator.particles_zx_end(level, z ,
                                                                             x ); treeIterator.set_iterator_to_particle_next_particle()) {

                temp_vec.at(treeIterator.y() + stencil_half, x +stencil_half, z % stencil_size) = tree_data[treeIterator];
            }
        }
    }
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

    ///////////////////////////
    ///
    /// Serial Neighbour Iteration (Only Von Neumann (Face) neighbours)
    ///
    /////////////////////////////////

    APRIterator<uint16_t> neighbour_iterator(apr);
    APRIterator<uint16_t> apr_iterator(apr);

    int num_rep = 1;

    timer.start_timer("APR serial iterator neighbours loop");

    //Basic serial iteration over all particles
    uint64_t particle_number;
    //Basic serial iteration over all particles


    ExtraParticleData<float> part_sum_standard(apr);

    APRTree<uint16_t> apr_tree(apr);

    ExtraParticleData<float> tree_intensity(apr_tree);
    ExtraParticleData<uint8_t> tree_counter(apr_tree);

    APRTreeIterator<uint16_t> treeIterator(apr_tree);

    APRTreeIterator<uint16_t> parentIterator(apr_tree);

  	/**** Filling the inside ********/
	uint64_t parent_number;

	ExtraParticleData<uint8_t> child_counter(apr_tree);
	ExtraParticleData<float> tree_data(apr_tree);

        for (particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            //set parent
            parentIterator.set_iterator_to_parent(apr_iterator);

            tree_data[parentIterator] = apr.particles_intensities[apr_iterator] +  tree_data[parentIterator];
            child_counter[parentIterator]++;
        }

        //then do the rest of the tree where order matters
        for (unsigned int level = treeIterator.level_max(); level >= treeIterator.level_min(); --level) {
            for(parent_number = treeIterator.particles_level_begin(level); parent_number < treeIterator.particles_level_end(level); ++parent_number) {
                treeIterator.set_iterator_to_particle_by_number(parent_number);
                tree_data[treeIterator]/=(1.0*child_counter[treeIterator]);
            }

            for (parent_number = treeIterator.particles_level_begin(level);
                 parent_number < treeIterator.particles_level_end(level); ++parent_number) {

                treeIterator.set_iterator_to_particle_by_number(parent_number);
                if(parentIterator.set_iterator_to_parent(treeIterator)) {
                    tree_data[parentIterator] = tree_data[treeIterator] + tree_data[parentIterator];
                    child_counter[parentIterator]++;
                }
            }
        }

//        for (unsigned int level = treeIterator.level_max(); level >= treeIterator.level_min(); --level) {
//               for(parent_number = treeIterator.particles_level_begin(level); parent_number < treeIterator.particles_level_end(level); ++parent_number) {
//                    treeIterator.set_iterator_to_particle_by_number(parent_number);
//                    tree_data[treeIterator]/=(1.0*child_counter[treeIterator]);
//                }
//        }

	// treeIterator only stores the Inside of the tree
   /****** End of filling **********/

    ExtraParticleData<float> part_sum(apr);


    const int stencil_half = 2;
    const int stencil_size = 2*stencil_half + 1;

    std::vector<double>  stencil;
    float stencil_value = 1.0f/(1.0f*pow(stencil_half*2 + 1,stencil_size));

    stencil.resize(pow(stencil_half*2 + 1,stencil_size),stencil_value);

    ExtraParticleData<float> part_sum_dense(apr);

    timer.start_timer("Dense neighbour access");

    for (int j = 0; j < num_rep; ++j) {

        for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

            unsigned int z = 0;
            unsigned int x = 0;

            const int y_num = apr_iterator.spatial_index_y_max(level);
            const int x_num = apr_iterator.spatial_index_x_max(level);
            const int z_num = apr_iterator.spatial_index_z_max(level);

            MeshData<float> temp_vec;
            temp_vec.init(apr_iterator.spatial_index_y_max(level) + (stencil_size-1),
                          apr_iterator.spatial_index_x_max(level) + (stencil_size-1),
                          stencil_size,
                          0); //padded boundaries

            z = 0;

            //initial condition
            for (int padd = 0; padd < stencil_half; ++padd) {
                update_dense_array(level,
                                   padd,
                                   apr,
                                   apr_iterator,
                                   treeIterator,
                                   tree_data,
                                   temp_vec,
                                   apr.particles_intensities,
                                   stencil_size,
                                   stencil_half);
            }

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - (stencil_half))) {
                    //update the next z plane for the access
                    update_dense_array(level, z + stencil_half, apr, apr_iterator, treeIterator, tree_data, temp_vec,apr.particles_intensities, stencil_size, stencil_half);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z+stencil_half)%stencil_size);

                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num ,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num , 0);
                    }
                }



#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.particles_zx_end(level, z,
                                                                                     x); apr_iterator.set_iterator_to_particle_next_particle()) {
                        double neigh_sum = 0;
                        int counter = 0;

                        const int k = apr_iterator.y() + stencil_half; // offset to allow for boundary padding
                        const int i = x + stencil_half;

                        //compute the stencil
                        for (int l = -stencil_half; l < stencil_half+1; ++l) {
                            for (int q = -stencil_half; q < stencil_half+1; ++q) {
                                for (int w = -stencil_half; w < stencil_half+1; ++w) {
                                    neigh_sum += stencil[counter]*temp_vec.at(k+w, i+q, (z+stencil_size+l)%stencil_size);
                                    counter++;
                                }
                            }
                        }

                        part_sum_dense[apr_iterator] = neigh_sum;

                    }//y, pixels/columns
                }//x , rows


            }//z
        }//levels
    }//reps


    timer.stop_timer();


    //check the result



    bool success = true;
    uint64_t f_c=0;

    ExtraParticleData<float> utest_particles(apr);

    apr.parameters.input_dir = options.directory;

    create_test_particles(apr,apr_iterator,treeIterator,utest_particles,apr.particles_intensities,tree_data,stencil,stencil_size, stencil_half);

//    MeshData<uint16_t> check_mesh;
//
//    apr.interp_img(check_mesh,part_sum_dense);
//
//    std::string image_file_name = options.directory +  "check.tif";
//    TiffUtils::saveMeshAsTiff(image_file_name, check_mesh);
//
//    apr.interp_img(check_mesh,utest_particles);
//
//    image_file_name = options.directory +  "check_standard.tif";
//    TiffUtils::saveMeshAsTiff(image_file_name, check_mesh);

    //Basic serial iteration over all particles
    for (particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        if(round(part_sum_dense.data[particle_number]) != round(utest_particles.data[particle_number])){

            float dense = part_sum_dense.data[particle_number];

            float standard = utest_particles.data[particle_number];

            //std::cout << apr_iterator.x()<< " "  << apr_iterator.y()<< " "  << apr_iterator.z() << " " << apr_iterator.level() << " " << dense << " " << standard << " " << (int)(apr_iterator.type()) << std::endl;

            success = false;
            f_c++;
        }

    }

    if(success){
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << " " << f_c <<  std::endl;
    }




}


void create_test_particles(APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,APRTreeIterator<uint16_t>& apr_tree_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,ExtraParticleData<float>& part_tree,std::vector<double>& stencil, const int stencil_size, const int stencil_half){

    for (uint64_t level_local = apr_iterator.level_max(); level_local >= apr_iterator.level_min(); --level_local) {


        MeshData<float> by_level_recon;
        by_level_recon.init(apr_iterator.spatial_index_y_max(level_local),apr_iterator.spatial_index_x_max(level_local),apr_iterator.spatial_index_z_max(level_local),0);

        for (uint64_t level = std::max((uint64_t)(level_local-1),(uint64_t)apr_iterator.level_min()); level <= level_local; ++level) {


            const float step_size = pow(2, level_local - level);

            uint64_t particle_number;

            for (particle_number = apr_iterator.particles_level_begin(level);
                 particle_number < apr_iterator.particles_level_end(level); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                int dim1 = apr_iterator.y() * step_size;
                int dim2 = apr_iterator.x() * step_size;
                int dim3 = apr_iterator.z() * step_size;

                float temp_int;
                //add to all the required rays

                temp_int = particles[apr_iterator];

                const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                    for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                        for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                            by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                        }
                    }
                }
            }
        }


        if(level_local < apr_iterator.level_max()){

            uint64_t level = level_local;

            const float step_size = 1;

            uint64_t particle_number;

            for (particle_number = apr_tree_iterator.particles_level_begin(level);
                 particle_number < apr_tree_iterator.particles_level_end(level); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_tree_iterator.set_iterator_to_particle_by_number(particle_number);

                int dim1 = apr_tree_iterator.y() * step_size;
                int dim2 = apr_tree_iterator.x() * step_size;
                int dim3 = apr_tree_iterator.z() * step_size;

                float temp_int;
                //add to all the required rays

                temp_int = part_tree[apr_tree_iterator];


                const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                    for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                        for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                            by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                        }
                    }
                }
            }

        }


        int x = 0;
        int z = 0;
        uint64_t level = level_local;

        for (z = 0; z < apr.spatial_index_z_max(level); ++z) {
            //lastly loop over particle locations and compute filter.
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.particles_zx_end(level, z,
                                                                                 x); apr_iterator.set_iterator_to_particle_next_particle()) {
                    double neigh_sum = 0;
                    float counter = 0;

                    const int k = apr_iterator.y(); // offset to allow for boundary padding
                    const int i = x;

                    for (int l = -stencil_half; l < stencil_half+1; ++l) {
                        for (int q = -stencil_half; q < stencil_half+1; ++q) {
                            for (int w = -stencil_half; w < stencil_half+1; ++w) {

                                if((k+w)>=0 & (k+w) < (apr.spatial_index_y_max(level))){
                                    if((i+q)>=0 & (i+q) < (apr.spatial_index_x_max(level))){
                                        if((z+l)>=0 & (z+l) < (apr.spatial_index_z_max(level))){
                                            neigh_sum += stencil[counter] * by_level_recon.at(k + w, i + q, z+l);
                                        }
                                    }
                                }


                                counter++;
                            }
                        }
                    }

                    test_particles[apr_iterator] = neigh_sum;

                }
            }
        }




       // std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
        //TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);

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
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
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

