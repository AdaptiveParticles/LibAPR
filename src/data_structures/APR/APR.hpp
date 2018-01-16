//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP

#include "benchmarks/development/Tree/PartCellStructure.hpp"

//#include "benchmarks/development/old_numerics/filter_numerics.hpp"
//#include "benchmarks/development/old_numerics/misc_numerics.hpp"

#include "src/data_structures/APR/PartCellData.hpp"

//#include "benchmarks/development/Tree/PartCellStructure.hpp"

#include "CurrLevel.hpp"

#include "src/io/hdf5functions_blosc.h"

#include "src/data_structures/APR/APR_iterator.hpp"

#include "src/misc/APR_timer.hpp"

#include "src/algorithm/APRParameters.hpp"

#include "src/numerics/APRCompress.hpp"

#include "src/io/APRWriter.hpp"

#include "src/data_structures/APR/APRAccess.hpp"

#include "src/numerics/APRReconstruction.hpp"

#include <map>
#include <unordered_map>


class APR_parameters;

typedef std::unordered_map<uint16_t,uint16_t> hash_map;
//typedef std::map<uint16_t,uint16_t> hash_map;

template<typename ImageType>
class APR : public APR_iterator<ImageType>{

    template<typename S>
    friend class APR_converter;

    friend class APRWriter;

    friend class PullingScheme;

    template<typename S>
    friend class APR_iterator;

    template<typename S>
    friend class ExtraPartCellData;

    friend class APRAccess;

    friend class APRReconstruction;

    friend void create_pc_data_new(APR<float>& apr,PartCellStructure<float,uint64_t>& pc_struct); //for testing

private:

    APRWriter apr_writer;

    APRReconstruction apr_recon;

    PartCellData<uint64_t> pc_data;

    std::vector<uint64_t> num_parts;
    std::vector<uint64_t> num_elements;
    ExtraPartCellData<uint64_t> num_parts_xy;
    uint64_t num_elements_total;

    std::vector<unsigned int> org_dims;

    //Experimental
    ExtraPartCellData<hash_map> random_access;

public:

    //Main internal datastructures

    ExtraPartCellData<ImageType> particles_int; // holds the particles intenisty information

    // holds the spatial and neighbours access information and methods

    //used for storing number of paritcles and cells per level for parallel access iterators


    std::string name;
    APR_parameters parameters;

    //old parameters (depreciated)
    Proc_par pars;

    APR(){
        this->pc_data_pointer = &pc_data;
    }


    //deprecitated
    ExtraPartCellData<uint16> y_vec;

    unsigned int orginal_dimensions(int dim){
        return pc_data.org_dims[dim];
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
    void write_apr_paraview(std::string save_loc,std::string file_name,ExtraPartCellData<T>& parts){
        apr_writer.write_apr_paraview((*this), save_loc,file_name,parts);
    }

    //write out ExtraPartCellData
    template< typename S>
    void write_particles_only( std::string save_loc,std::string file_name,ExtraPartCellData<S>& parts_extra){
        apr_writer.write_particles_only( *this ,save_loc, file_name, parts_extra);
    };

    //read in ExtraPartCellData
    template<typename T>
    void read_parts_only(std::string file_name,ExtraPartCellData<T>& extra_parts){
        apr_writer.read_parts_only(*this,file_name,extra_parts);
    };

    ////////////////////////
    ///
    ///  APR Reconstruction Methods (Calls APRReconstruction methods)
    ///
    //////////////////////////

    template<typename U,typename V>
    void interp_img(Mesh_data<U>& img,ExtraPartCellData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        apr_recon.interp_img((*this),img, parts);

    }


    template<typename U>
    void interp_depth_ds(Mesh_data<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //


        apr_recon.interp_depth_ds((*this),img);

    }

    template<typename U>
    void interp_depth(Mesh_data<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        apr_recon.interp_depth((*this),img);

    }

    template<typename U>
    void interp_type(Mesh_data<U>& img){
        //
        //  Interpolates the APR
        //

        apr_recon.interp_type((*this),img);

    }

    template<typename U,typename V>
    void interp_parts_smooth(Mesh_data<U>& out_image,ExtraPartCellData<V>& interp_data,std::vector<float> scale_d = {2,2,2}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        apr_recon.interp_parts_smooth((*this),out_image,interp_data,scale_d);

    }

    template<typename U,typename V>
    void get_parts_from_img(Mesh_data<U>& img,ExtraPartCellData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using the nearest pixel (rounded up, i.e. next pixel after particles that sit on off pixel locations)
        //

        //re-write this.

        parts.init(*this);

        //initialization of the iteration structures
        APR_iterator<ImageType> apr_it(*this); //this is required for parallel access
        uint64_t part;

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it)
        for (part = 0; part < this->num_parts_total; ++part) {
            //needed step for any parallel loop (update to the next part)
            apr_it.set_iterator_to_particle_by_number(part);

            apr_it(parts) = img.access_no_protection(apr_it.y_nearest_pixel(),apr_it.x_nearest_pixel(),apr_it.z_nearest_pixel());

        }

    }

    template<typename U,typename V>
    void get_parts_from_img(std::vector<Mesh_data<U>>& img_by_level,ExtraPartCellData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using an image tree (img_by_level is a vector of images)
        //

        parts.init(*this);

        //initialization of the iteration structures
        APR_iterator<ImageType> apr_it(*this); //this is required for parallel access
        uint64_t part;

#pragma omp parallel for schedule(static) private(part) firstprivate(apr_it)
        for (part = 0; part < this->num_parts_total; ++part) {
            //needed step for any parallel loop (update to the next part)
            apr_it.set_iterator_to_particle_by_number(part);

            apr_it(parts) = img_by_level[apr_it.depth()].access_no_protection(apr_it.y(),apr_it.x(),apr_it.z());

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

        num_parts_xy.initialize_structure_parts_empty(particles_int);

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


    ///////////////////////
    ///
    /// Random Access Structures (Experimental) Cheeseman 2018
    ///
    ///
    ///////////////////////
//
//    int random_access_pc(uint64_t depth,uint16_t y,uint64_t x,uint64_t z){
//        //
//        //  Random access check for valid x,z, any given y, returns the index of the stored Particle Intensity.
//        //
//
//        int j;
//
//        uint64_t pc_offset = pc_data.x_num[depth]*z + x;
//
//        if(random_access.data[depth][pc_offset].size() > 0) {
//            hash_map::iterator pc = random_access.data[depth][pc_offset][0].find(y);
//
//            if(pc != random_access.data[depth][pc_offset][0].end()){
//                j = pc->second;
//            } else {
//                return -1;
//            }
//
//        } else {
//            return -1;
//
//        }
//
//        return j;
//
//    }
//
//    //////////////////////////
//    ///
//    /// Experimental random access neighbours.
//    ///
//    /// \tparam S data type of the particles
//    /// \param face the neighbour direction (+y,-y,+x,-x,+z,-z)
//    /// \param parts the particles data structure
//    /// \param neigh_val vector returning the particles values of the neighbours
//    ////////////////////////
//
//    template<typename S>
//    void get_neigh_random(unsigned int face,ExtraPartCellData<S>& parts,std::vector<S>& neigh_val){
//        //
//        //  Get APR face neighbours relying on random access through a map, or unordered map structure for y
//        //
//
//        const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
//        const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
//        const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};
//
//        constexpr uint8_t neigh_child_dir[6][3] = {{4,2,2},{4,2,2},{0,4,4},{0,4,4},{0,2,2},{0,2,2}};
//
//        constexpr uint8_t child_offsets[3][3] = {{0,1,1},{1,0,1},{1,1,0}};
//
//        //first try on same depth
//        int z_ = this->z() + dir_z[face];
//        int x_ = this->x() + dir_x[face];
//        int y_ = this->y() + dir_y[face];
//        int depth_ = this->depth();
//
//        uint16_t j=0;
//
//        uint64_t pc_offset = pc_data.x_num[depth_]*z_ + x_;
//        bool found = false;
//
//        neigh_val.resize(0);
//
//        if((x_ < 0) | (x_ >= pc_data.x_num[depth_]) | (z_ < 0) | (z_ >= pc_data.z_num[depth_]) ){
//            //out of bounds
//            return;
//        }
//
//        if(random_access.data[depth_][pc_offset].size() > 0) {
//            hash_map::iterator pc = random_access.data[depth_][pc_offset][0].find(y_);
//
//            if(pc != random_access.data[depth_][pc_offset][0].end()){
//                j = pc->second;
//                found = true;
//            }
//        }
//
//        if(!found){
//            //
//            //  Find parents
//            //
//
//            unsigned int depth_p = depth_ - 1;
//            unsigned int x_p = x_/2;
//            unsigned int y_p = y_/2;
//            unsigned int z_p = z_/2;
//
//            pc_offset = pc_data.x_num[depth_p]*z_p + x_p;
//
//            if(random_access.data[depth_p][pc_offset].size() > 0) {
//                hash_map::iterator pc = random_access.data[depth_p][pc_offset][0].find(y_p);
//
//                if(pc != random_access.data[depth_p][pc_offset][0].end()){
//                    j = pc->second;
//                    found = true;
//                }
//            }
//
//            if(!found) {
//
//                if(depth_ < pc_data.depth_max) {
//                    // get the potentially 4 children
//                    unsigned int depth_c = depth_ + 1;
//                    unsigned int x_c = (x_ + dir_x[face])*2 + (dir_x[face]<0);
//                    unsigned int y_c = (y_ + dir_y[face])*2 + (dir_y[face]<0);
//                    unsigned int z_c = (z_ + dir_z[face])*2 + (dir_z[face]<0);
//
//                    unsigned int dir = face/2;
//
//                    for (int i = 0; i < 2; ++i) {
//                        for (int k = 0; k < 2; ++k) {
//                            y_ = y_c + (child_offsets[dir][0])*i + (child_offsets[dir][0])*k;
//                            x_ = x_c + (child_offsets[dir][1])*i + (child_offsets[dir][1])*k;
//                            z_ = z_c + (child_offsets[dir][2])*i + (child_offsets[dir][2])*k;
//
//                            //add of they exist
//                            if((x_ < 0) | (x_ >= pc_data.x_num[depth_c]) | (z_ < 0) | (z_ >= pc_data.z_num[depth_]) ){
//                                //out of bounds
//
//                            } else {
//
//                                pc_offset = pc_data.x_num[depth_c]*z_ + x_;
//
//                                if (random_access.data[depth_c][pc_offset].size() > 0) {
//                                    hash_map::iterator pc = random_access.data[depth_c][pc_offset][0].find(y_);
//
//                                    if (pc != random_access.data[depth_c][pc_offset][0].end()) {
//                                        j = pc->second;
//                                        neigh_val.push_back(parts.data[depth_c][pc_offset][j]);
//                                    }
//                                }
//
//                            }
//
//
//
//                        }
//                    }
//
//
//
//                }
//
//            } else{
//                neigh_val.push_back(parts.data[depth_p][pc_offset][j]);
//            }
//
//        } else{
//
//            neigh_val.push_back(parts.data[depth_][pc_offset][j]);
//
//        }
//
//
//    }

//    void init_random_access(){
//
//
//        random_access.initialize_structure_parts_empty(particles_int);
//
//        ExtraPartCellData<std::pair<uint16_t,uint16_t>> hash_init;
//
//        hash_init.initialize_structure_parts_empty(particles_int);
//
//        int counter = 0;
//
//        //create the intiializer lists
//
//        //loop over all particles
//        for (this->begin(); this->end() == true; this->it_forward()) {
//
//            hash_init.data[this->depth()][this->curr_level.pc_offset].push_back({this->y(),this->j()});
//
//        }
//
//        //now create the actual hash tables
//        for(uint64_t i = pc_data.depth_min;i <= pc_data.depth_max;i++) {
//
//            const unsigned int x_num_ = pc_data.x_num[i];
//            const unsigned int z_num_ = pc_data.z_num[i];
//
//            for (uint64_t z_ = 0; z_ < z_num_; z_++) {
//
//                for (uint64_t x_ = 0; x_ < x_num_; x_++) {
//                    const uint64_t offset_pc_data = x_num_ * z_ + x_;
//                    if(hash_init.data[i][offset_pc_data].size() > 0) {
//                        random_access.data[i][offset_pc_data].resize(1);
//
//                        random_access.data[i][offset_pc_data][0].insert(hash_init.data[i][offset_pc_data].begin(),
//                                                                        hash_init.data[i][offset_pc_data].end());
//                    }
//
//                }
//            }
//        }
//
//    }



};




#endif //PARTPLAY_APR_HPP
