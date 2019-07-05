//
// Created by cheesema on 21.01.18.
//

#ifndef PARTPLAY_BENCHAPR_HPP
#define PARTPLAY_BENCHAPR_HPP

#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRConverter.hpp"
#include <utility>
#include <cmath>
#include "../test/TestTools.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include "io/APRWriter.hpp"
#include "io/APRFile.hpp"

#include "data_structures/APR/particles/LazyData.hpp"

#include "AnalysisData.hpp"

struct cmdLineBenchOptions{
    float CR_min = 0;
    float CR_max = 9999;
    int image_size = 512;
    int number_reps = 1;
    int dimension = 3;

    std::string analysis_file_name = "analysis";
    std::string output_dir = "";

};

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

cmdLineBenchOptions read_bench_command_line_options(int argc, char **argv){

    cmdLineBenchOptions result;

    if(command_option_exists(argv, argv + argc, "-CR_min"))
    {
        result.CR_min = std::stof(std::string(get_command_option(argv, argv + argc, "-CR_min")));
    }

    if(command_option_exists(argv, argv + argc, "-CR_max"))
    {
        result.CR_max = std::stof(std::string(get_command_option(argv, argv + argc, "-CR_max")));
    }

    if(command_option_exists(argv, argv + argc, "-image_size"))
    {
        result.image_size = std::stoi(std::string(get_command_option(argv, argv + argc, "-image_size")));
    }

    if(command_option_exists(argv, argv + argc, "-number_reps"))
    {
        result.number_reps = std::stoi(std::string(get_command_option(argv, argv + argc, "-number_reps")));
    }

    if(command_option_exists(argv, argv + argc, "-file_name"))
    {
        result.analysis_file_name = std::string(get_command_option(argv, argv + argc, "-file_name"));
    }

    if(command_option_exists(argv, argv + argc, "-output_dir"))
    {
        result.output_dir = std::string(get_command_option(argv, argv + argc, "-output_dir"));
    }

    return result;

}

struct BenchmarkData{

    std::vector<APR> aprs;
    std::vector<ParticleData<uint16_t>> parts;

};

class BenchAPRHelper {

    float CR_min = 0;
    float CR_max = 9999;
    int image_size = 0;
    int number_reps = 1;

    int actual_image_size = 0;

    BenchmarkData basic_data;

    void load_benchmark_data(){

        for (int i = 0; i < CR_names.size(); ++i) {
            if((CR_names[i] >= CR_min) && (CR_names[i] <= CR_max)){
                std::string name = get_source_directory_apr() + benchmark_file_location + "/cr_" + std::to_string(CR_names[i]) + ".apr";
                APR apr;
                ParticleData<uint16_t> parts;

                APRFile aprFile;
                aprFile.open(name,"READ");
                aprFile.read_apr(apr);
                aprFile.read_particles(apr,"particle_intensities",parts);
                aprFile.close();

                basic_data.aprs.push_back(apr);
                basic_data.parts.push_back(parts);
            }
        }

    }

public:

    std::string benchmark_file_location = "files";
    std::vector<int> CR_names = {1,3,5,10,15,20,30,54,124,1000};

    AnalysisData analysisData;

    int number_datsets(){
        return basic_data.parts.size();
    }

    int get_number_reps(){
        return number_reps;
    }

    template<typename PartsType>
    bool generate_dataset(int num,APR& apr,ParticleData<PartsType>& parts){

        int dim_sz = actual_image_size/basic_data.aprs[num].org_dims(0);

        std::vector<int> tile_dims = {dim_sz,dim_sz,dim_sz};

        tileAPR(tile_dims,basic_data.aprs[num], basic_data.parts[num],apr,parts);

        analysisData.add_float_data("num_reps",number_reps);
        analysisData.add_float_data("image_size",actual_image_size);

        analysisData.add_apr_info(apr);

        return true;

    }

    void initialize_benchmark(cmdLineBenchOptions ops){

        CR_min = ops.CR_min;
        CR_max = ops.CR_max;
        image_size = ops.image_size;
        number_reps = ops.number_reps;

        actual_image_size = std::max((int) pow(2,std::ceil(log2(image_size))),(int) 256);

        load_benchmark_data();
    }

    std::string get_source_directory_apr(){
        // returns path to the directory where utils.cpp is stored

        std::string tests_directory = std::string(__FILE__);
        tests_directory = tests_directory.substr(0, tests_directory.find_last_of("\\/") + 1);

        return tests_directory;
    }


    /**
     * tileAPR - tiles an APR to generate a larger APR for testing purposes.
     * @param tile_dims a 3 dimensional vector with the number of times in each direction to tile
     * @param apr_input The APR to be tiled
     * @tparam parts particles to be tiled
     * @param apr_tiled The tiled APR
     * @tparam tiled_parts The tiled particles
*/
    template<typename S,typename U>
    static void tileAPR(std::vector<int> tile_dims, APR& apr_input,ParticleData<S>& parts, APR& apr_tiled,ParticleData<U>& tiled_parts) {
        //
        // Note the data-set should be a power of 2 in its dimensons for this to work.
        //

        APRTimer timer(false);

        if (tile_dims.size() != 3) {
            std::cerr << "in-correct tilling dimensions" << std::endl;
        }

        auto it = apr_input.iterator();

        float check_y = log2(1.0f*it.org_dims(0));
        float check_x = log2(1.0f*it.org_dims(1));
        float check_z = log2(1.0f*it.org_dims(2));

        //this function only works for datasets that are powers of 2.
        bool pow_2y = (check_y - std::floor(check_y)) == 0;
        bool pow_2x = (check_x - std::floor(check_x)) == 0;
        bool pow_2z = (check_z - std::floor(check_z)) == 0;

        if(!pow_2y || !pow_2x || !pow_2z){
            std::cerr << "The dimensions must be a power of 2!" << std::endl;
        }

        //round up to nearest even dimension



        int org_dims_y = std::ceil(it.org_dims(0)/2.0f)*2;
        int org_dims_x = std::ceil(it.org_dims(1)/2.0f)*2;
        int org_dims_z = std::ceil(it.org_dims(2)/2.0f)*2;


        //now to tile the APR
        auto new_y_num = org_dims_y * tile_dims[0];
        auto new_x_num = org_dims_x * tile_dims[1];
        auto new_z_num = org_dims_z * tile_dims[2];


        apr_tiled.aprInfo.init(new_y_num, new_x_num, new_z_num);
        apr_tiled.linearAccess.genInfo = &apr_tiled.aprInfo;

        PullingSchemeSparse ps;
        ps.initialize_particle_cell_tree(apr_tiled.aprInfo);


        timer.start_timer("init sparse tree");

        const int level_offset = apr_tiled.level_max() - apr_input.level_max();

        for (int z_tile = 0; z_tile < tile_dims[2]; ++z_tile) {
            for (int x_tile = 0; x_tile < tile_dims[1]; ++x_tile) {
                for (int y_tile = 0; y_tile < tile_dims[0]; ++y_tile) {

                    for (int level = it.level_min(); level <= it.level_max(); level++) {
                        int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it)
#endif
                        for (z = 0; z < it.z_num(level); z++) {
                            for (int x = 0; x < it.x_num(level); ++x) {

                                int new_level = level_offset + level;

                                //transform the co-ordinates to the new frame of reference..

                                int z_offset = (z_tile * org_dims_z) / apr_tiled.aprInfo.level_size[new_level];
                                int x_offset = (x_tile * org_dims_x) / apr_tiled.aprInfo.level_size[new_level];
                                int y_offset = (y_tile * org_dims_y) / apr_tiled.aprInfo.level_size[new_level];


                                if (level == it.level_max()) {
                                    if ((x % 2 == 0) && (z % 2 == 0)) {
                                        const size_t offset_pc =
                                                apr_tiled.aprInfo.x_num[new_level - 1] * ((z + z_offset) / 2) +
                                                ((x + x_offset) / 2);

                                        auto &mesh = ps.particle_cell_tree.data[new_level - 1][offset_pc][0].mesh;

                                        for (it.begin(level, z, x); it < it.end();
                                             it++) {
                                            //insert
                                            int y = (it.y() + y_offset) / 2;
                                            mesh[y] = 1;
                                        }
                                    }
                                } else {

                                    const size_t offset_pc =
                                            apr_tiled.aprInfo.x_num[new_level] * (z + z_offset) + (x + x_offset);

                                    auto &mesh = ps.particle_cell_tree.data[new_level][offset_pc][0].mesh;

                                    for (it.begin(level, z, x); it < it.end();
                                         it++) {
                                        //insert
                                        mesh[it.y() + y_offset] = 4;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();

        timer.start_timer("pulling scheme");

        //ps.pulling_scheme_main();

        timer.stop_timer();

        timer.start_timer("initialize structures");

        apr_tiled.linearAccess.initialize_linear_structure_sparse(apr_input.parameters, ps.particle_cell_tree);

        apr_tiled.apr_initialized = true;

        timer.stop_timer();

        tiled_parts.init(apr_tiled.total_number_particles());

        auto it_tile = apr_tiled.iterator();


        //auto it_tree = apr_input.tree_iterator();

        ParticleData<float> tparts_input;

        APRTreeNumerics::fill_tree_mean(apr_input, parts, tparts_input);

        //
        // Now we need to sample the particles, with the fact that some particles with have increased resolution in the merge
        // potentially due to boundary effects
        //

        int org_y_num = org_dims_y;
        int org_x_num = org_dims_x;
        int org_z_num = org_dims_z;

        //now compute the particles
        for (int level = (it.level_min() + level_offset); level <= it_tile.level_max(); level++) {

            auto level_org = level - level_offset;
            int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it,it_tile)
#endif
            for (z = 0; z < it_tile.z_num(level); z++) {

                int z_org =
                        (((z) * apr_tiled.aprInfo.level_size[level]) % org_z_num) / apr_tiled.aprInfo.level_size[level];

                for (int x = 0; x < it_tile.x_num(level); ++x) {

                    int x_org = (((x) * apr_tiled.aprInfo.level_size[level]) % org_x_num) /
                                apr_tiled.aprInfo.level_size[level];


                    it.begin(level_org, z_org, x_org);


                    for (it_tile.begin(level, z, x); it_tile < it_tile.end(); ++it_tile) {

                        int y_org = (((it_tile.y()) * apr_tiled.aprInfo.level_size[level]) % org_y_num) /
                                    apr_tiled.aprInfo.level_size[level];


                        while ((it.y() < y_org) && (it < it.end())) {
                            it++;
                        }


                        if ((it.y() == y_org) && (it != it.end())) {
                            tiled_parts[it_tile] = parts[it];
                        }

                        if (it >= (it.end() - 1)) {
                            it.begin(level_org, z_org, x_org);
                        }

                    }
                }
            }
        }

    };

};




#endif