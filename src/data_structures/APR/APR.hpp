//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "src/io/hdf5functions_blosc.h"
#include "src/misc/APRTimer.hpp"
#include "src/algorithm/APRParameters.hpp"
#include "src/numerics/APRCompress.hpp"
#include "src/data_structures/APR/APRAccess.hpp"
#include "src/io/APRWriter.hpp"
#include "src/numerics/APRReconstruction.hpp"
#include "src/data_structures/APR/ExtraParticleData.hpp"

#include <map>
#include <unordered_map>


class APRParameters;
class OldAPRConverter;

template<typename ImageType>
class APR {

    template<typename S>
    friend class APRConverter;

    friend class APRWriter;

    friend class PullingScheme;

    template<typename S>
    friend class APRIterator;

    template<typename S>
    friend class APRIteratorOld;

    template<typename S>
    friend class ExtraPartCellData;

    friend class APRAccess;

    friend class APRReconstruction;

    friend class APRBenchmark;

    friend class OldAPRConverter;

private:

    APRWriter apr_writer;
    APRReconstruction apr_recon;
    APRAccess apr_access;

    //deprecated - old access paradigm
    std::vector<uint64_t> num_parts;
    std::vector<uint64_t> num_elements;
    ExtraPartCellData<uint64_t> num_parts_xy;
    uint64_t num_elements_total;
    uint64_t num_parts_total;

public:

    //APR Particle Intensities
    ExtraParticleData<ImageType> particles_intensities;

    //Main internal datastructures
    std::string name;
    APRParameters parameters;

    //old parameters (depreciated)

    APR(){
    }

    //deprecitated
    ExtraPartCellData<ImageType> particles_int_old; // holds the particles intenisty information

    unsigned int orginal_dimensions(int dim){
        return apr_access.org_dims[dim];
    }

    uint64_t level_max(){
        return apr_access.level_max;
    }

    uint64_t level_min(){
        return apr_access.level_min;
    }

    inline uint64_t spatial_index_x_max(const unsigned int level){
        return (apr_access).x_num[level];
    }

    inline uint64_t spatial_index_y_max(const unsigned int level){
        return (apr_access).y_num[level];
    }

    inline uint64_t spatial_index_z_max(const unsigned int level){
        return (apr_access).z_num[level];
    }

    inline uint64_t total_number_particles(){
        return (apr_access).total_number_particles;
    }

    ///////////////////////////////////
    ///
    /// APR IO Methods (Calls members of the APRWriter class)
    ///
    //////////////////////////////////

    //basic IO
    void read_apr(std::string file_name){
        //
        apr_writer.read_apr(*this,file_name);
    }

    void write_apr(std::string save_loc,std::string file_name){
        apr_writer.write_apr(*this, save_loc,file_name);
    }

    void write_apr(std::string save_loc,std::string file_name,APRCompress<ImageType>& apr_compressor,unsigned int blosc_comp_type,unsigned int blosc_comp_level,unsigned int blosc_shuffle){
        apr_writer.write_apr((*this),save_loc, file_name, apr_compressor,blosc_comp_type ,blosc_comp_level,blosc_shuffle);
    }

    //generate APR that can be read by paraview
    template<typename T>
    void write_apr_paraview(std::string save_loc,std::string file_name,ExtraParticleData<T>& parts){
        apr_writer.write_apr_paraview((*this), save_loc,file_name,parts);
    }

    //write out ExtraPartCellData
    template< typename S>
    void write_particles_only( std::string save_loc,std::string file_name,ExtraParticleData<S>& parts_extra){
        apr_writer.write_particles_only(save_loc, file_name, parts_extra);
    };

    //read in ExtraPartCellData
    template<typename T>
    void read_parts_only(std::string file_name,ExtraParticleData<T>& extra_parts){
        apr_writer.read_parts_only(file_name,extra_parts);
    };

    ////////////////////////
    ///
    ///  APR Reconstruction Methods (Calls APRReconstruction methods)
    ///
    //////////////////////////

    template<typename U,typename V>
    void interp_img(MeshData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        apr_recon.interp_img((*this),img, parts);

    }


    template<typename U>
    void interp_depth_ds(MeshData<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //


        apr_recon.interp_depth_ds((*this),img);

    }

    template<typename U>
    void interp_depth(MeshData<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        apr_recon.interp_level((*this), img);

    }

    template<typename U>
    void interp_type(MeshData<U>& img){
        //
        //  Interpolates the APR
        //

        apr_recon.interp_type((*this),img);

    }

    template<typename U,typename V>
    void interp_parts_smooth(MeshData<U>& out_image,ExtraParticleData<V>& interp_data,std::vector<float> scale_d = {2,2,2}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        apr_recon.interp_parts_smooth((*this),out_image,interp_data,scale_d);

    }

    template<typename U,typename V>
    void get_parts_from_img(MeshData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using the nearest pixel (rounded up, i.e. next pixel after particles that sit on off pixel locations)
        //

        //re-write this.



        //initialization of the iteration structures
        APRIterator<ImageType> apr_iterator(*this); //this is required for parallel access
        uint64_t particle_number;
        parts.data.resize(apr_iterator.total_number_particles());


#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            apr_iterator(parts) = img.access_no_protection(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

        }

    }

    template<typename U,typename V>
    void get_parts_from_img(std::vector<MeshData<U>>& img_by_level,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using an image tree (img_by_level is a vector of images)
        //


        //initialization of the iteration structures
        APRIterator<ImageType> apr_iterator(*this); //this is required for parallel access
        uint64_t particle_number;

        parts.data.resize(apr_iterator.total_number_particles());

#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            parts[apr_iterator] = img_by_level[apr_iterator.level()].access_no_protection(apr_iterator.y(),apr_iterator.x(),apr_iterator.z());

        }


    }
};


#endif //PARTPLAY_APR_HPP
