//
// Created by cheesema on 24.01.18.
//

#ifndef PARTPLAY_APRBENCHMARK_HPP
#define PARTPLAY_APRBENCHMARK_HPP


#include <benchmarks/analysis/AnalysisData.hpp>
#include "src/data_structures/APR/APR.hpp"

class APRBenchmark {

public:

    APRBenchmark(){

    }

    AnalysisData analysis_data;
    //Linear neighbour access
    template<typename U,typename V>
    float pixels_linear_neighbour_access(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats);

    template<typename U,typename V>
    float apr_linear_neighbour_access(APR<U> apr,float num_repeats);

    //Random access
    template<typename U,typename V>
    float pixel_neighbour_random(uint64_t y_num, uint64_t x_num, uint64_t z_num, float num_repeats);

    template<typename U,typename V>
    float apr_random_access(APR<U>& apr, float num_repeats);

    template<typename ImageType>
    void benchmark_dataset(APRConverter<ImageType>& apr_converter);

};

template<typename ImageType>
void APRBenchmark::benchmark_dataset(APRConverter<ImageType>& apr_converter){

    APR<uint16_t> apr;

    std::string name = apr_converter.par.input_image_name;

    APRTimer timer;

    TiffUtils::TiffInfo inputTiff(apr_converter.par.input_dir + apr_converter.par.input_image_name);
    MeshData<uint16_t> input_image = TiffUtils::getMesh<uint16_t>(inputTiff);

    apr_converter.total_timer.verbose_flag = true;

    apr_converter.get_apr_method(apr, input_image);

    float num_repeats = 10;

    float linear_pixel_pm = pixels_linear_neighbour_access<uint16_t,float>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),num_repeats);
    float linear_apr_pm = apr_linear_neighbour_access<uint16_t,float>(apr,num_repeats);

    float pp_ratio_linear = linear_pixel_pm/linear_apr_pm;
    analysis_data.add_float_data("pp_ratio_linear",pp_ratio_linear);

    float num_repeats_random = 10000000;

    float random_pixel_pm = pixel_neighbour_random<uint16_t,float>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2), num_repeats_random);
    float random_apr_pm = apr_random_access<uint16_t,float>(apr,num_repeats_random);

    float pp_ratio_random = random_pixel_pm/random_apr_pm;
    analysis_data.add_float_data("pp_ratio_random",pp_ratio_random);

    APRCompress<uint16_t> apr_compress;

    APRWriter apr_writer;

    ExtraParticleData<uint16_t> intensities;
    intensities.copy_parts(apr,apr.particles_intensities);

    apr_compress.set_compression_type(1);

    timer.verbose_flag = false;

    timer.start_timer("write_compress_wnl");
    float apr_wnl_in_mb = apr_writer.write_apr(apr,apr_converter.par.input_dir ,name + "_compress",apr_compress,BLOSC_ZSTD,3,2);
    timer.stop_timer();


    apr.particles_intensities.copy_parts(apr,intensities);
    apr_compress.set_compression_type(2);

    timer.start_timer("write_compress_predict_only");
    float apr_predict_in_mb = apr_writer.write_apr(apr,apr_converter.par.input_dir ,name + "_compress1",apr_compress,BLOSC_ZSTD,3,2);
    timer.stop_timer();

    apr.particles_intensities.copy_parts(apr,intensities);
    apr_compress.set_compression_type(0);

    timer.start_timer("write_no_compress");
    float apr_direct_in_mb = apr_writer.write_apr(apr,apr_converter.par.input_dir ,name + "_compress2",apr_compress,BLOSC_ZSTD,3,2);
    timer.stop_timer();

    timer.start_timer("write_particles_only");
    float apr_parts_only_mb = apr_writer.write_particles_only(apr_converter.par.input_dir ,name + "_parts_only",intensities);

    timer.stop_timer();

    analysis_data.add_float_data("storage_normal", apr_direct_in_mb );
    analysis_data.add_float_data("storage_wnl", apr_wnl_in_mb );
    analysis_data.add_float_data("storage_predict", apr_predict_in_mb );
    analysis_data.add_float_data("storage_only_particles", apr_parts_only_mb );

    analysis_data.add_float_data("number_particles",apr.apr_access.total_number_particles);
    analysis_data.add_float_data("total_number_gaps",apr.apr_access.total_number_gaps);
    analysis_data.add_float_data("total_number_non_empty_rows",apr.apr_access.total_number_non_empty_rows);

    analysis_data.add_float_data("total_number_type_stored",apr.apr_access.global_index_by_level_end[apr.level_max()-1]);

    analysis_data.add_float_data("ratio_access_storage", (apr_direct_in_mb - apr_parts_only_mb)/apr_direct_in_mb );

    analysis_data.add_timer(timer);

    analysis_data.add_timer(apr_converter.fine_grained_timer);
    analysis_data.add_timer(apr_converter.computation_timer);
    analysis_data.add_timer(apr_converter.method_timer);
    analysis_data.add_timer(apr_converter.allocation_timer);
    analysis_data.add_timer(apr_converter.total_timer);

    float estimated_storage_size_gaps_access  = (3.0727*pow(10.0,-5)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2)) + (2.125*pow(10.0,-6))*apr.apr_access.total_number_gaps*log2(1.0*apr.apr_access.total_number_gaps/(apr.apr_access.total_number_non_empty_rows*1.0));

    float particles_storage_cost = 2*apr.apr_access.total_number_particles/(1000000.0);

    float type_storage_cost = 2*apr.apr_access.global_index_by_level_end[apr.level_max()-1]/1000000.0;

    float apr_access_bits_per_particle = 8*1000000.0*estimated_storage_size_gaps_access/(1.0*apr.apr_access.total_number_particles);

    float total_in_memory_cost_apr = particles_storage_cost + estimated_storage_size_gaps_access;

    float memory_cost_neighbour_pixels = ((2+4)*(apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2)))/1000000.0;

    float memory_cost_neighbour_apr = estimated_storage_size_gaps_access + 6*apr.apr_access.total_number_particles/(1000000.0);

    analysis_data.add_float_data("memory_cost_neighbour_apr",memory_cost_neighbour_apr);
    analysis_data.add_float_data("memory_cost_neighbour_pixels",memory_cost_neighbour_pixels);
    analysis_data.add_float_data("total_in_memory_cost_apr",total_in_memory_cost_apr);
    analysis_data.add_float_data("apr_access_bits_per_particle",apr_access_bits_per_particle);
    analysis_data.add_float_data("type_storage_cost",type_storage_cost);
    analysis_data.add_float_data("estimated_storage_size_gaps_access",estimated_storage_size_gaps_access);
    analysis_data.add_float_data("particles_storage_cost",particles_storage_cost);

    // #TODO CR/MCR PP

    float total_image_size = ((apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2))*2)/1000000.0;

    float computational_ratio = (apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2))/(apr.total_number_particles()*1.0);

    float memory_reduction_neighbour_access = memory_cost_neighbour_pixels/memory_cost_neighbour_apr;

    float MCR_normal = total_image_size/apr_direct_in_mb;

    float MCR_predict = total_image_size/apr_predict_in_mb;

    float MCR_winl = total_image_size/apr_wnl_in_mb;

    analysis_data.add_float_data("tota_image_size",total_image_size);
    analysis_data.add_float_data("computational_ratio",computational_ratio);
    analysis_data.add_float_data("memory_reduction_neighbour_access",memory_reduction_neighbour_access);
    analysis_data.add_float_data("MCR_normal",MCR_normal);
    analysis_data.add_float_data("MCR_predict",MCR_predict);
    analysis_data.add_float_data("MCR_winl",MCR_winl);


}


template<typename U,typename V>
float APRBenchmark::apr_linear_neighbour_access(APR<U> apr,float num_repeats){

    APRTimer timer;

    ExtraParticleData<V> output(apr);
    APRIterator<U> apr_iterator(apr);
    APRIterator<U> neighbour_iterator(apr);

    uint64_t particle_number;

    timer.start_timer("APR parallel iterator neighbour loop");

    for(int r = 0;r < num_repeats;r++) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator, neighbour_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)

            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            U neigh_sum = 0;
            U counter = 0;

            //loop over all the neighbours and set the neighbour iterator to it
            for (int direction = 0; direction < 6; ++direction) {
                apr_iterator.find_neighbours_in_direction(direction);
                // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                    if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                        //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
                        neigh_sum += apr.particles_intensities[neighbour_iterator];
                        counter++;
                    }

                }
            }

            output[apr_iterator] = neigh_sum/(counter*1.0);

        }
    }


    timer.stop_timer();

    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds/num_repeats;

    std::cout << "APR Linear Neigh: " << (apr.total_number_particles()) << " took: " << time << std::endl;
    std::cout << "per 1000000 particles took: " << (time)/((1.0*apr.total_number_particles())/1000000.0) << std::endl;

    analysis_data.add_float_data("neigh_apr_linear_total",time);
    analysis_data.add_float_data("neigh_apr_linear_perm",(time)/((1.0*apr.total_number_particles())/1000000.0));

    return (time)/((1.0*apr.total_number_particles())/1000000.0);

}

template<typename U,typename V>
float APRBenchmark::pixels_linear_neighbour_access(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    MeshData<U> input_data;
    MeshData<V> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

    APRTimer timer;
    timer.verbose_flag = false;
    timer.start_timer("full pixel neighbour access");

    int j = 0;
    int k = 0;
    int i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    //float neigh_sum = 0;

    for(int r = 0;r < num_repeats;r++){

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n)
#endif
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                for(k = 0;k < y_num;k++){
                    U neigh_sum = 0;
                    U counter = 0;

                    for(int  d  = 0;d < 6;d++){

                        i_n = i + dir_x[d];
                        k_n = k + dir_y[d];
                        j_n = j + dir_z[d];

                        //check boundary conditions
                        if((i_n >=0) & (i_n < x_num) ){
                            if((j_n >=0) & (j_n < z_num) ){
                                if((k_n >=0) & (k_n < y_num) ){
                                    neigh_sum += input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                                    counter++;
                                }
                            }
                        }
                    }

                    output_data.mesh[j*x_num*y_num + i*y_num + k] = neigh_sum/(counter*1.0);

                }
            }
        }

    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds/num_repeats;

    std::cout << "Pixel Linear Neigh: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    std::cout << "per 1000000 pixel took: " << (time)/((1.0*x_num*y_num*z_num)/1000000.0) << std::endl;

    analysis_data.add_float_data("neigh_pixel_linear_total",time);
    analysis_data.add_float_data("neigh_pixel_linear_perm",(time)/((1.0*x_num*y_num*z_num)/1000000.0));

    return (time)/((1.0*x_num*y_num*z_num)/1000000.0);

}

template<typename U,typename V>
float APRBenchmark::pixel_neighbour_random(uint64_t y_num, uint64_t x_num, uint64_t z_num, float num_repeats){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};

    MeshData<U> input_data;
    MeshData<V> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);

    APRTimer timer;


    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    std::vector<uint16_t> x_random;
    std::vector<uint16_t> y_random;
    std::vector<uint16_t> z_random;

    x_random.reserve(num_repeats);
    y_random.reserve(num_repeats);
    z_random.reserve(num_repeats);

    for(int r = 0;r < num_repeats;r++) {

        x_random.push_back((uint16_t)(std::rand() % x_num));
        y_random.push_back((uint16_t)(std::rand() % y_num));
        z_random.push_back((uint16_t)(std::rand() % z_num));
    }


    timer.verbose_flag = false;
    timer.start_timer("full previous filter");

    int r;

    //Generate random access numbers
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n,r)
#endif
    for(r = 0;r < num_repeats;r++){

        i = x_random[r];
        j = z_random[r];
        k = y_random[r];

        U neigh_sum = 0;
        U counter = 0;

        for(int  d  = 0;d < 6;d++){

            i_n = (int)(i + dir_x[d]);
            k_n =  (int)(k + dir_y[d]);
            j_n =  (int)(j + dir_z[d]);

            //check boundary conditions
            if((i_n >=0) & (i_n < x_num) ){
                if((j_n >=0) & (j_n < z_num) ){
                    if((k_n >=0) & (k_n < y_num) ){
                        neigh_sum += input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                        counter++;
                    }
                }
            }
        }

        output_data.mesh[j*x_num*y_num + i*y_num + k] = neigh_sum/(counter*1.0);

    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds;

    float est_full_time = (time)*(1.0*x_num*y_num*z_num)/num_repeats;

    std::cout << "Random Access Pixel: Size: " << num_repeats << " took: " << (est_full_time) << std::endl;
    std::cout << "per 1000000 pixel took: " << (time*1000000.0)/((1.0*num_repeats)) << std::endl;

    analysis_data.add_float_data("random_access_pixel_neigh_total",est_full_time);
    analysis_data.add_float_data("random_access_pixel_neigh_perm",(time*1000000.0)/((1.0*num_repeats)));

    return (time*1000000.0)/((1.0*num_repeats));

}

template<typename U,typename V>
float APRBenchmark::apr_random_access(APR<U>& apr, float num_repeats){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //


    APRTimer timer;

    ExtraParticleData<V> output(apr);
    APRIterator<U> apr_iterator(apr);
    APRIterator<U> neighbour_iterator(apr);

    std::vector<uint16_t> x_;
    std::vector<uint16_t> y_;
    std::vector<uint16_t> z_;
    std::vector<uint8_t> level_;

    x_.reserve(apr.total_number_particles());
    y_.reserve(apr.total_number_particles());
    z_.reserve(apr.total_number_particles());
    level_.reserve(apr.total_number_particles());


    uint64_t particle_number;

    for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
        //needed step for any parallel loop (update to the next part)

        apr_iterator.set_iterator_to_particle_by_number(particle_number);

        x_.push_back(apr_iterator.x());
        y_.push_back(apr_iterator.y());
        z_.push_back(apr_iterator.z());
        level_.push_back((uint8_t)apr_iterator.level());

    }


    std::vector<uint16_t> x_random;
    std::vector<uint16_t> y_random;
    std::vector<uint16_t> z_random;
    std::vector<uint8_t> level_random;

    x_random.reserve(num_repeats);
    y_random.reserve(num_repeats);
    z_random.reserve(num_repeats);
    level_random.reserve(num_repeats);

    for(int r = 0;r < num_repeats;r++) {
        //needed step for any parallel loop (update to the next part)
        particle_number = (std::rand() % apr.total_number_particles());

        x_random.push_back(x_[particle_number]);
        y_random.push_back(y_[particle_number]);
        z_random.push_back(z_[particle_number]);
        level_random.push_back(level_[particle_number]);
    }

    ParticleCell random_particle_cell;


    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    //Generate random access numbers
    int r;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(r) firstprivate(apr_iterator, neighbour_iterator,random_particle_cell)
#endif
    for(r = 0;r < num_repeats;r++){

        random_particle_cell.x = x_random[r];
        random_particle_cell.y = y_random[r];
        random_particle_cell.z = z_random[r];
        random_particle_cell.level = level_random[r];

        U neigh_sum = 0;
        U counter = 0;

        apr_iterator.set_iterator_by_particle_cell(random_particle_cell);

        //loop over all the neighbours and set the neighbour iterator to it
        for (int direction = 0; direction < 6; ++direction) {
            apr_iterator.find_neighbours_in_direction(direction);
            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
            for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                    //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
                    neigh_sum += apr.particles_intensities[neighbour_iterator];
                    counter++;
                }

            }
        }

        output[apr_iterator] = neigh_sum/(counter*1.0);

    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds;

    float est_full_time = (time)*(1.0*apr.total_number_particles())/num_repeats;

    std::cout << "Random Access Particle TOTAL : " << num_repeats << " took: " << (est_full_time) << std::endl;
    std::cout << "per 1000000 Particles took: " << (time*1000000.0)/((1.0*num_repeats)) << std::endl;

    analysis_data.add_float_data("random_access_apr_neigh_total",est_full_time);
    analysis_data.add_float_data("random_access_apr_neigh_perm",(time*1000000.0)/((1.0*num_repeats)));

    return (time*1000000.0)/((1.0*num_repeats));

}



#endif //PARTPLAY_APRBENCHMARK_HPP
