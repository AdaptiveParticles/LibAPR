//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "benchmarks/development/Tree/PartCellStructure.hpp"

//#include "benchmarks/development/old_numerics/filter_numerics.hpp"
//#include "benchmarks/development/old_numerics/misc_numerics.hpp"

#include "benchmarks/development/Tree/PartCellData.hpp"

//#include "benchmarks/development/Tree/PartCellStructure.hpp"

#include "benchmarks/development/Tree/CurrLevel.hpp"

#include "src/io/hdf5functions_blosc.h"

#include "benchmarks/development/Tree/APRIteratorOld.hpp"

#include "src/misc/APRTimer.hpp"

#include "src/algorithm/APRParameters.hpp"

#include "src/numerics/APRCompress.hpp"

#include "src/io/APRWriter.hpp"

#include "src/data_structures/APR/APRAccess.hpp"

#include "src/numerics/APRReconstruction.hpp"

#include "src/data_structures/APR/ExtraParticleData.hpp"

#include <map>
#include <unordered_map>


class APRParameters;

template<typename ImageType>
class APR {

    template<typename S>
    friend class APRConverter;

    friend class old::APRWriter;

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

    friend void create_pc_data_new(APR<float>& apr,PartCellStructure<float,uint64_t>& pc_struct); //for testing

private:

    APRWriter apr_writer;
    APRReconstruction apr_recon;
    APRAccess apr_access;

    //deprecated - old access paradigm
    PartCellData<uint64_t> pc_data;
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
    Proc_par pars;

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


#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
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

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            apr_iterator(parts) = img_by_level[apr_iterator.level()].access_no_protection(apr_iterator.y(),apr_iterator.x(),apr_iterator.z());

        }


    }

private:

    void get_part_numbers() {
        //
        //  Computes totals of total number of particles, and the total number of elements (PC and gap nodes)
        //

        this->num_parts.resize(pc_data.depth_max + 1);
        this->num_elements.resize(pc_data.depth_max + 1);

        int z_, x_, j_, y_;

        uint64_t counter_parts = 0;
        uint64_t counter_elements = 0;

        for (uint64_t depth = (pc_data.depth_min); depth <= pc_data.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[depth];
            const unsigned int z_num_ = pc_data.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            CurrentLevel<ImageType, uint64_t> curr_level_l(pc_data);
            curr_level_l.set_new_depth(depth, pc_data);

            const float step_size = pow(2, curr_level_l.depth_max - curr_level_l.depth);

#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(curr_level_l) reduction(+:counter_parts) reduction(+:counter_elements)  if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    curr_level_l.set_new_xz(x_, z_, pc_data);

                    for (j_ = 0; j_ < curr_level_l.j_num; j_++) {

                        bool iscell = curr_level_l.new_j(j_, pc_data);

                        if (iscell) {
                            //Indicates this is a particle cell node
                            curr_level_l.update_cell(pc_data);

                            counter_parts++;

                        } else {

                            curr_level_l.update_gap(pc_data);

                        }

                        counter_elements++;
                    }
                }
            }

            this->num_parts[depth] = counter_parts;
            this->num_elements[depth] = counter_elements;

        }

        this->num_parts_total = 0;
        num_elements_total = 0;

        for (int i = 0; i <= pc_data.depth_max; ++i) {

            num_elements_total += num_elements[i];
        }

        this->num_parts_total += num_parts[pc_data.depth_max];

    }

    void set_part_numbers_xz() {
        //
        //  Computes totals of total number of particles in each xz
        //

        num_parts_xy.initialize_structure_parts_empty(particles_int_old);

        int z_, x_, j_, y_;

        uint64_t counter_parts = 0;

        for (uint64_t depth = (pc_data.depth_min); depth <= pc_data.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[depth];
            const unsigned int z_num_ = pc_data.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            CurrentLevel<ImageType, uint64_t> curr_level_l(pc_data);
            curr_level_l.set_new_depth(depth, pc_data);

            const float step_size = pow(2, curr_level_l.depth_max - curr_level_l.depth);
#pragma omp parallel for default(shared) private(z_,x_,j_,counter_parts) firstprivate(curr_level_l)  if(z_num_*x_num_ > 100)
            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {

                    counter_parts=0;

                    curr_level_l.set_new_xz(x_, z_, pc_data);

                    for (j_ = 0; j_ < curr_level_l.j_num; j_++) {

                        bool iscell = curr_level_l.new_j(j_, pc_data);

                        if (iscell) {
                            //Indicates this is a particle cell node
                            curr_level_l.update_cell(pc_data);

                            counter_parts++;

                        } else {

                            curr_level_l.update_gap(pc_data);

                        }
                    }
                    num_parts_xy.data[curr_level_l.depth][curr_level_l.pc_offset].push_back(counter_parts);
                }
            }

        }

        counter_parts = 0;

        //it needs to be a cumulative sum

        for (uint64_t depth = (pc_data.depth_min); depth <= pc_data.depth_max; depth++) {
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_data.x_num[depth];
            const unsigned int z_num_ = pc_data.z_num[depth];

            const unsigned int x_num_min_ = 0;
            const unsigned int z_num_min_ = 0;

            for (z_ = z_num_min_; z_ < z_num_; z_++) {
                //both z and x are explicitly accessed in the structure

                for (x_ = x_num_min_; x_ < x_num_; x_++) {
                    size_t pc_offset = x_num_*z_ + x_;
                    counter_parts += num_parts_xy.data[depth][pc_offset][0];
                    num_parts_xy.data[depth][pc_offset][0] = counter_parts;
                }
            }

        }


    }



};




#endif //PARTPLAY_APR_HPP
