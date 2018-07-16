//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP


#include "algorithm/APRParameters.hpp"
#include "numerics/APRCompress.hpp"
#include "numerics/APRReconstruction.hpp"
#include "algorithm/APRConverter.hpp"


#include "io/APRWriter.hpp"
#include "APRAccess.hpp"
#include "ExtraParticleData.hpp"
#include "APRTree.hpp"

#include "APRIterator.hpp"

template<typename ImageType>
class APR {

    APRWriter apr_writer;
    APRReconstruction apr_recon;

public:

    APRAccess apr_access;
    APRConverter<ImageType> apr_converter;
    APRCompress<ImageType> apr_compress;
    APRTree<ImageType> apr_tree;

    ExtraParticleData<ImageType> particles_intensities;
    std::string name;
    APRParameters parameters;

    unsigned int orginal_dimensions(int dim) const { return apr_access.org_dims[dim]; }

    uint64_t level_max() const { return apr_access.l_max; }
    uint64_t level_min() const { return apr_access.l_min; }

    inline uint64_t spatial_index_x_max(const unsigned int level) const { return apr_access.x_num[level]; }
    inline uint64_t spatial_index_y_max(const unsigned int level) const { return apr_access.y_num[level]; }
    inline uint64_t spatial_index_z_max(const unsigned int level) const { return apr_access.z_num[level]; }

    inline uint64_t total_number_particles() const { return (apr_access).total_number_particles; }

    ///////////////////////////////////
    ///
    /// APR Generation Methods (Calls members of the APRConverter class)
    ///
    //////////////////////////////////

    APRIterator iterator() {
        return APRIterator(apr_access);
    }

    bool get_apr(){
        //copy across parameters
        this->apr_converter.par = parameters;
        return this->apr_converter.get_apr(*this);
    }

    template<typename T>
    bool get_apr(PixelData<T>& input_img){
        this->apr_converter.par = parameters;
        return this->apr_converter.get_apr_method(*this, input_img);
    }

    ///////////////////////////////////
    ///
    /// APR IO Methods (Calls members of the APRWriter class)
    ///
    //////////////////////////////////

    //basic IO
    void read_apr(std::string file_name){
        apr_writer.read_apr(*this,file_name);
    }

    void read_apr(std::string file_name,bool read_tree,unsigned int max_level_delta){
        apr_writer.read_apr(*this,file_name,read_tree,max_level_delta);
    }

    FileSizeInfo write_apr(std::string save_loc,std::string file_name){
        return apr_writer.write_apr(*this, save_loc,file_name);
    }

    FileSizeInfo write_apr(std::string save_loc,std::string file_name,APRCompress<ImageType>& apr_compressor,unsigned int blosc_comp_type,unsigned int blosc_comp_level,unsigned int blosc_shuffle,bool write_tree = false){
        return apr_writer.write_apr((*this),save_loc, file_name, apr_compressor,blosc_comp_type ,blosc_comp_level,blosc_shuffle,write_tree);
    }

    FileSizeInfo write_apr(std::string save_loc,std::string file_name,unsigned int blosc_comp_type,unsigned int blosc_comp_level,unsigned int blosc_shuffle,bool write_tree = false){

        return apr_writer.write_apr((*this),save_loc, file_name, this->apr_compress,blosc_comp_type ,blosc_comp_level,blosc_shuffle,write_tree);
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
    }

    //read in ExtraPartCellData
    template<typename T>
    void read_parts_only(std::string file_name,ExtraParticleData<T>& extra_parts){
        apr_writer.read_parts_only(file_name,extra_parts);
    }

    ////////////////////////
    ///
    ///  APR Reconstruction Methods (Calls APRReconstruction methods)
    ///
    //////////////////////////

    template<typename U,typename V>
    void interp_img(PixelData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        apr_recon.interp_img((*this),img, parts);
    }

    template<typename U>
    void interp_level_ds(PixelData<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        apr_recon.interp_depth_ds((*this),img);
    }

    template<typename U>
    void interp_level(PixelData<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        apr_recon.interp_level((*this), img);
    }


    template<typename U,typename V>
    void interp_parts_smooth(PixelData<U>& out_image,ExtraParticleData<V>& interp_data,std::vector<float> scale_d = {2,2,2}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        apr_recon.interp_parts_smooth((*this),out_image,interp_data,scale_d);
    }

    template<typename U,typename V>
    void get_parts_from_img(std::vector<PixelData<U>>& img_by_level,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using an image tree (img_by_level is a vector of images)

        //initialization of the iteration structures
         //this is required for parallel access
        auto apr_iterator = iterator();
        parts.data.resize(apr_iterator.total_number_particles());

        std::cout << "Total number of particles: " << apr_iterator.total_number_particles() << std::endl;

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        parts[apr_iterator] = img_by_level[apr_iterator.level()].at(apr_iterator.y(),apr_iterator.x(),apr_iterator.z());

                    }
                }
            }
        }

    }

    template<typename U,typename V>
    void get_parts_from_img(PixelData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using the nearest pixel (rounded up, i.e. next pixel after particles that sit on off pixel locations)
        //

        //initialization of the iteration structures
        //this is required for parallel access
        auto apr_iterator = iterator();
        uint64_t particle_number;
        parts.data.resize(apr_iterator.total_number_particles());


        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        parts[apr_iterator] = img.at(apr_iterator.y_nearest_pixel(), apr_iterator.x_nearest_pixel(),
                                                     apr_iterator.z_nearest_pixel());

                    }
                }
            }
        }

    }

};


#endif //PARTPLAY_APR_HPP
