//
// Created by cheesema on 21.11.18.
//

#ifndef APR_TIME_APRTIMEIO_HPP
#define APR_TIME_APRTIMEIO_HPP

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/particles/PartCellData.hpp"
#include "io/APRWriter.hpp"




template<typename BuffType>
struct Buffer{
    std::vector<BuffType> data;
    uint64_t global_b=0;
    uint64_t global_e=0;
    uint64_t buffer_size=0;

    void init(uint64_t sz){
        data.resize(sz);
        buffer_size = sz;
        global_b = 0;
        global_e = sz - 1;
    }
};

template<typename ImageType>
struct TimeData{
    PartCellData<ImageType>* add_fp;
    PartCellData<ImageType>* update_fp;
    PartCellData<ImageType>* remove_fp;

    PartCellData<ImageType>* add_y;
    PartCellData<ImageType>* update_y;
    PartCellData<ImageType>* remove_y;

    unsigned int l_max;

};

struct ChangeTable{
    //stores what rows have additions or removals this time step using a pixel cache in 2D
    std::vector<PixelData<uint64_t>> add_b;
    std::vector<PixelData<uint64_t>> remove_b;

    std::vector<PixelData<uint64_t>> add_e;
    std::vector<PixelData<uint64_t>> remove_e;

    std::vector<PixelData<uint64_t>> update_b;
    std::vector<PixelData<uint64_t>> update_e;

    std::vector<PixelData<uint64_t>> particle_sum;

    void init(RandomAccess& init_access){
        add_b.resize(init_access.level_max()+1);
        add_e.resize(init_access.level_max()+1);
        remove_b.resize(init_access.level_max()+1);
        remove_e.resize(init_access.level_max()+1);

        update_b.resize(init_access.level_max()+1);
        update_e.resize(init_access.level_max()+1);

        particle_sum.resize(init_access.level_max()+1);

        for (int i = init_access.l_min; i <= init_access.level_max(); ++i) {
            add_b[i].init(init_access.x_num[i],init_access.z_num[i],1);
            remove_b[i].init(init_access.x_num[i],init_access.z_num[i],1);

            add_e[i].init(init_access.x_num[i],init_access.z_num[i],1);
            remove_e[i].init(init_access.x_num[i],init_access.z_num[i],1);

            particle_sum[i].initWithValue(init_access.gap_map.x_num[i],init_access.gap_map.z_num[i],1,0);

            update_b[i].init(init_access.x_num[i],init_access.z_num[i],1);
            update_e[i].init(init_access.x_num[i],init_access.z_num[i],1);
        }
    }

    void max(){
        for (int i = 0; i < add_b.size(); ++i) {
            std::fill(add_b[i].mesh.begin(),add_b[i].mesh.end(),UINT64_MAX);
            std::fill(add_e[i].mesh.begin(),add_e[i].mesh.end(),UINT64_MAX);
            std::fill(remove_b[i].mesh.begin(),remove_b[i].mesh.end(),UINT64_MAX);
            std::fill(remove_e[i].mesh.begin(),remove_e[i].mesh.end(),UINT64_MAX);
            std::fill(update_b[i].mesh.begin(),update_b[i].mesh.end(),UINT64_MAX);
            std::fill(update_e[i].mesh.begin(),update_e[i].mesh.end(),UINT64_MAX);
        }
    }
};

struct BufferPc{
    std::vector<uint16_t> y;
    std::vector<uint16_t> x;
    std::vector<uint16_t> z;
    std::vector<uint8_t> l;
    uint64_t global_b=0;
    uint64_t global_e=0;
    uint64_t buffer_size=0;

    void init(uint64_t sz){
        y.resize(sz);
        x.resize(sz);
        z.resize(sz);
        l.resize(sz);
        buffer_size = sz;
        global_b = 0;
        global_e = sz - 1;
    }
};

class TimeTests;

template<typename ImageType>
class APRTimeIO : public APRWriter{

    friend class TimeTests;

    bool add_pc_max(uint16_t y, uint16_t x, uint16_t z){
        return ((y%2 + x%2 + z%2) == 0);
    }

    APRParameters initial_parameters;
    RandomAccess initial_access_info;

    std::vector<uint64_t> update_f_totals;
    std::vector<uint64_t> update_totals;

    std::vector<uint64_t> add_f_totals;
    std::vector<uint64_t> add_totals;

    std::vector<uint64_t> remove_f_totals;
    std::vector<uint64_t> remove_totals;

    uint64_t current_time_step= UINT64_MAX;

    uint64_t direct_updated = UINT64_MAX;

    int64_t key_frame_delta = 0;
    std::string file_name;
    std::vector<uint64_t> key_frame_index_vector;
    uint64_t current_key_frame = UINT64_MAX;
    uint64_t current_delta_frame = UINT64_MAX;

    //keyframe vars (writing)
    uint64_t key_frame = 0;
    uint64_t key_frame_index = 0;
    uint64_t delta_num = 0;

    Buffer<ImageType> update_f_buff;
    Buffer<ImageType> add_f_buff;
    Buffer<ImageType> remove_f_buff;

    BufferPc update_buff;
    BufferPc add_buff;
    BufferPc remove_buff;

    std::vector<APR> APR_buffer;

    ChangeTable changeTable;

    std::vector<uint64_t> update_global_index;
    std::vector<uint64_t> remove_global_index;
    std::vector<uint64_t> add_global_index;

    ParticleData<uint16_t> updated_t_prev;


    bool current_direction = true;
    bool key_frame_loaded = true;

public:

    uint64_t get_num_updated(){
        uint64_t updated = 0;

        if(current_t > 0) {
            updated = add_totals[current_t] - add_totals[current_t-1];
        } else {
            updated = add_totals[current_t];
        }

        if(current_t > 0) {
            updated += update_totals[current_t] - update_totals[current_t-1];
        } else {
            updated += update_totals[current_t];
        }

        if(current_t > 0) {
            updated += remove_totals[current_t] - remove_totals[current_t-1];
        } else {
            updated += remove_totals[current_t];
        }
        return updated;
    }

    uint64_t updated_num = 0;
    uint64_t removed_num = 0;
    uint64_t added_num = 0;

    bool calculate_time_updated = false;

    APR prev_apr;

    PartCellData<ImageType> current_particles;
    ParticleData<uint16_t> updated_t;

    APR* current_APR;

    uint64_t number_time_steps=0;

    uint64_t get_dims(hid_t obj_id, const char *ds_name){

        hid_t dset_id =  H5Dopen2(obj_id, ds_name ,H5P_DEFAULT);

        hid_t dspace = H5Dget_space(dset_id);

        hsize_t current_dim;
        H5Sget_simple_extent_dims(dspace, &current_dim, NULL);

        H5Sclose(dspace);
        H5Dclose(dset_id);

        return current_dim;

    }

    template<typename R,typename S,typename T>
    void copy_parts_to_pcd(APR<R>& apr,ExtraParticleData<S>& parts2copy,PartCellData<T>& parts_pcd){

        auto apr_iterator = apr.iterator();
        parts_pcd.initialize_structure_parts_empty(apr);

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

                    const uint64_t offset_pc_data = apr_iterator.spatial_index_x_max(level) * z + x;

                    auto begin = apr_iterator.set_new_lzx(level, z, x);

                    if(begin != UINT64_MAX) {

                        auto b = apr_iterator.global_index();
                        auto e = apr_iterator.end_index;

                        parts_pcd.data[level][offset_pc_data].resize(e - b);

                        std::copy(parts2copy.data.begin() + b,
                                  parts2copy.data.begin() + e,
                                  parts_pcd.data[level][offset_pc_data].begin());
                    }

                }
            }
        }


    }

    template<typename R,typename S,typename T>
    void copy_pcd_to_parts(APR<R>& apr,ExtraParticleData<S>& partsDest,PartCellData<T>& parts_pcd2copy){

        auto apr_iterator = apr.iterator();
        partsDest.data.resize(apr_iterator.total_number_particles());

        auto total_parts = apr_iterator.total_number_particles();

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

                    const uint64_t offset_pc_data = apr_iterator.spatial_index_x_max(level) * z + x;

                    auto begin = apr_iterator.set_new_lzx(level, z, x);

                    if(begin != UINT64_MAX) {

                        auto b = apr_iterator.global_index();
                        auto e = apr_iterator.end_index;

                        auto sz = parts_pcd2copy.data[level][offset_pc_data];

                        std::copy(parts_pcd2copy.data[level][offset_pc_data].begin(),
                                  parts_pcd2copy.data[level][offset_pc_data].end(),
                                  partsDest.data.begin() + b);
                    }

                }
            }
        }


    }


    template<typename R>
    void transfer_particles_through_time(APR<ImageType>& apr_old,APR<ImageType>& apr_new,ExtraParticleData<R>& old_parts,ExtraParticleData<R>& new_parts,R set_value = 0){
        //
        //  Propogates a set of particles from one APR to another where the particles exist in both
        //
        //

        auto new_iterator = apr_new.iterator();
        auto old_iterator = apr_old.iterator();

        new_parts.data.resize(apr_new.total_number_particles(),0);
        std::fill(new_parts.data.begin(),new_parts.data.end(),0);

        for (unsigned int level = new_iterator.level_min(); level <= new_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(new_iterator,old_iterator)
#endif
            for (z = 0; z < new_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < new_iterator.spatial_index_x_max(level); ++x) {
                    old_iterator.set_new_lzx(level, z, x);
                    for (new_iterator.set_new_lzx(level, z, x); new_iterator.global_index() < new_iterator.end_index;
                         new_iterator.set_iterator_to_particle_next_particle()) {

                        while((old_iterator.y() < new_iterator.y()) && (old_iterator.global_index() < old_iterator.end_index)){
                            old_iterator.set_iterator_to_particle_next_particle();
                        }

                        if(new_iterator.y() == old_iterator.y()){
                            new_parts[new_iterator] = old_parts[old_iterator];
                        } else {
                            new_parts[new_iterator] = set_value;
                        }

                    }
                }
            }
        }

    }

    template<typename R,typename S,typename T,typename U>
    void interp_particles_apr_to_apr(APR<R>& apr_from,APR<S>& apr_to,ExtraParticleData<T>& parts_from,ExtraParticleData<U>& parts_to) {
        //
        //  APR-> APR interpolation using piecewise constant
        //
        //  {P_from,T_from} -> {P_to}, where P are the Particles sets
        //
        //  This is done by first direct assignment from {P_from,T_from} -> {P_to,T_to} then followed by upsample(T_to) -> {P_to}
        //

        APRTree<R> tree_from;
        APRTree<S> tree_to;

        // Initilize the two trees
        tree_from.init(apr_from); // access for T_from
        tree_to.init(apr_to); // access for T_to

        //  Down-sample the original particles

        ExtraParticleData<float> parts_from_tree;
        ExtraParticleData<U> parts_to_tree;

        parts_from_tree.data.resize(tree_from.total_number_parent_cells());
        std::fill(parts_from_tree.data.begin(),parts_from_tree.data.end(),0);

        // {P_from} ->(ds) {T_from}
        tree_from.fill_tree_mean(apr_from,tree_from,parts_from,parts_from_tree);

        // Init the target particles
        const U init_val = 0;
        parts_to.data.resize(apr_to.total_number_particles());
        std::fill(parts_to.data.begin(),parts_to.data.end(),init_val);

        parts_to_tree.data.resize(tree_to.total_number_parent_cells());
        std::fill(parts_to_tree.data.begin(),parts_to_tree.data.end(),init_val);

        // Now PC interp {P_from,T_from} -> {P_to,T_to}
        auto tree_to_it = tree_to.tree_iterator();
        auto tree_from_it = tree_from.tree_iterator();

        auto apr_to_it = apr_to.iterator();
        auto apr_from_it = apr_from.iterator();

        // First {P_from} -> {P_to,T_to}
        for (unsigned int level = apr_from_it.level_min(); level <= apr_from_it.level_max(); ++level) {
            int z = 0;
            int x = 0;

            const bool max_level = (level == apr_from_it.level_max());

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_from_it,apr_to_it,tree_to_it)
#endif
            for (z = 0; z < apr_from_it.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_from_it.spatial_index_x_max(level); ++x) {

                    // Tree Particles
                    if(!max_level){
                        tree_to_it.set_new_lzx(level, z, x);
                    }

                    // Particles
                    apr_to_it.set_new_lzx(level, z, x);

                    for (apr_from_it.set_new_lzx(level, z, x); apr_from_it.global_index() < apr_from_it.end_index;
                         apr_from_it.set_iterator_to_particle_next_particle()) {

                        while ((apr_to_it.y() < apr_from_it.y()) && (apr_to_it.global_index() < apr_to_it.end_index)) {
                            // {P_f->P_t}
                            apr_to_it.set_iterator_to_particle_next_particle();

                        }

                        if((apr_to_it.y() == apr_from_it.y())  ) {
                            parts_to[apr_to_it] = parts_from[apr_from_it];
                        }

                        if(!max_level){
                            // {P_f->T_t}

                            while ((tree_to_it.y() < apr_from_it.y()) && (tree_to_it.global_index() < tree_to_it.end_index)) {
                                tree_to_it.set_iterator_to_particle_next_particle();
                            }

                            if((tree_to_it.y() == apr_from_it.y()) ){

                                parts_to_tree[tree_to_it] = parts_from[apr_from_it];
                            }
                        }
                    }
                }
            }
        }

        // First {T_from} -> {P_to,T_to}
        for (unsigned int level = tree_from_it.level_min(); level <= tree_from_it.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_from_it,apr_to_it,tree_to_it)
#endif
            for (z = 0; z < apr_from_it.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_from_it.spatial_index_x_max(level); ++x) {

                    // Particles
                    apr_to_it.set_new_lzx(level, z, x);

                    // Tree Particles
                    tree_to_it.set_new_lzx(level, z, x);

                    for (tree_from_it.set_new_lzx(level, z, x); tree_from_it.global_index() < tree_from_it.end_index;
                         tree_from_it.set_iterator_to_particle_next_particle()) {

                        // {P_f->P_t}
                        while ((apr_to_it.y() < tree_from_it.y()) && (apr_to_it.global_index() < apr_to_it.end_index)) {
                            apr_to_it.set_iterator_to_particle_next_particle();
                        }

                        if(apr_to_it.y() == tree_from_it.y()){
                            parts_to[apr_to_it] = parts_from_tree[tree_from_it];
                        }


                        // {P_f->T_t}
                        while ((tree_to_it.y() < tree_from_it.y()) && (tree_to_it.global_index() < tree_to_it.end_index)) {
                            tree_to_it.set_iterator_to_particle_next_particle();
                        }
                        if(tree_to_it.y() == tree_from_it.y()){
                            parts_to_tree[tree_to_it] = parts_from_tree[tree_from_it];
                        }

                    }
                }
            }
        }

        //require additional iterator here for parent.
        auto tree_to_parent_it = tree_to.tree_iterator();

        // Final step, for those still empty do {T_to} (us)-> {T_to}
        for (unsigned int level = (tree_to_it.level_min()+1); level <= tree_to_it.level_max(); ++level) {
            // note in this case the order from min->max must be preserved

            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_to_parent_it, tree_to_it)
#endif
            for (z = 0; z < tree_to_it.spatial_index_z_max(level); z++) {
                for (x = 0; x < tree_to_it.spatial_index_x_max(level); ++x) {

                    // Parent Tree Particles
                    tree_to_parent_it.set_new_lzx(level-1, z/2, x/2);

                    for (tree_to_it.set_new_lzx(level, z, x); tree_to_it.global_index() < tree_to_it.end_index;
                         tree_to_it.set_iterator_to_particle_next_particle()) {

                        while ((tree_to_parent_it.y() < (tree_to_it.y()/2)) && (tree_to_parent_it.global_index() < tree_to_parent_it.end_index)) {
                            // {T_t(us)->T_t}
                            tree_to_parent_it.set_iterator_to_particle_next_particle();
                        }

                        if((tree_to_parent_it.y() == (tree_to_it.y()/2))){
                            if(parts_to_tree[tree_to_it]==0) {
                                parts_to_tree[tree_to_it] = parts_to_tree[tree_to_parent_it];
                            }
                        }
                    }

                }
            }
        }


        // Final step, for those still empty do {T_to} -> {P_to} (Since we are doing a read from parent to particle we don't need to worry about race conditions)
        for (unsigned int level = (apr_to_it.level_min()+1); level <= apr_to_it.level_max(); ++level) {
            // note in this case the order from min->max must be preserved

            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_to_it, tree_to_parent_it)
#endif
            for (z = 0; z < apr_to_it.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_to_it.spatial_index_x_max(level); ++x) {

                    // Tree Parent Particles
                    tree_to_parent_it.set_new_lzx(level-1, z/2, x/2);

                    for (apr_to_it.set_new_lzx(level, z, x); apr_to_it.global_index() < apr_to_it.end_index;
                         apr_to_it.set_iterator_to_particle_next_particle()) {

                        while ((tree_to_parent_it.y() < (apr_to_it.y()/2)) && (tree_to_parent_it.global_index() < tree_to_parent_it.end_index)) {
                            // {T_t(us)->P_t}
                            tree_to_parent_it.set_iterator_to_particle_next_particle();
                        }

                        if(tree_to_parent_it.y() == (apr_to_it.y()/2)){
                            if(parts_to[apr_to_it]==0) {
                                parts_to[apr_to_it] = parts_to_tree[tree_to_parent_it];
                            }
                        }
                    }
                }
            }

        }


    }




    template<typename R,typename S>
    void add_particles(APR<R>& apr,PartCellData<S>& parts_pcd,std::vector<PixelData<uint64_t>>& add_b,std::vector<PixelData<uint64_t>>& add_e,BufferPc& buff,std::vector<S>& buff_f){

        /*
         *
         * ADD!
         *
         *
         */



        auto& access = apr.apr_access;
        uint64_t x_;
        uint64_t z_;

        for (uint64_t level = (access.level_min()); level <= access.level_max(); level++) {

            auto x_num_ = (unsigned int) access.x_num[level];
            auto z_num_ = (unsigned int) access.z_num[level];
            auto y_num_ = (unsigned int) access.y_num[level];

            const bool max_level = (access.level_max() == level);

            const auto y_num_max = (unsigned int) access.y_num[level];
            const auto x_num_max = (unsigned int) access.x_num[level];
            const auto z_num_max = (unsigned int) access.z_num[level];

            auto it = apr.iterator();


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_, z_) firstprivate(it)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {
                for (x_ = 0; x_ < x_num_; x_++) {

                    const uint64_t offset_pc_data = x_num_ * z_ + x_;

                    const auto x_a = x_;
                    const auto z_a = z_;

                    if(add_b[level].at(x_a,z_a,0)!=UINT64_MAX){

                        it.set_new_lzx(level,z_,x_);

                        auto mit = it.get_current_gap();

                        auto pc = it.get_current_particle_cell();

                        auto pcdKey = it.get_pcd_key();

                        auto& part_vec = parts_pcd.data[level][offset_pc_data];

                        auto add_num = add_e[level].at(x_a,z_a,0) - add_b[level].at(x_a,z_a,0)+1;

                        std::vector<S> add_val;
                        add_val.resize(part_vec.size() + add_num,0);

                        for (auto i = add_b[level].at(x_a,z_a,0); i < (add_e[level].at(x_a,z_a,0)+1); ++i) {
                            pc.y = buff.y[i];

                            access.find_particle_cell_pcdKey(pc,mit,pcdKey);

                            add_val[pcdKey.local_ind] =  buff_f[i];

                        }

                        int offset = 0;
                        for (int j = 0; j < add_val.size(); ++j) {
                            auto val = add_val[j];

                            if(val > 0){
                                offset-=1;
                            } else {
                                add_val[j] = part_vec[j+offset];
                            }
                        }

                        std::swap(part_vec,add_val);

                    }

                }
            }
        }


    }



    template<typename R,typename S>
    void remove_particles(APR<R>& apr,PartCellData<S>& parts_pcd,std::vector<PixelData<uint64_t>>& remove_b,std::vector<PixelData<uint64_t>>& remove_e,BufferPc& buff){
        //
        //  Removes the aprticles between timesteps
        //

        /*
         *
        * REMOVE!
        *
        */

        auto& access = apr.apr_access;
        uint64_t x_;
        uint64_t z_;

        for (uint64_t level = (access.level_min()); level <= access.level_max(); level++) {

            auto x_num_ = (unsigned int) access.x_num[level];
            auto z_num_ = (unsigned int) access.z_num[level];
            auto y_num_ = (unsigned int) access.y_num[level];

            const bool max_level = (access.level_max() == level);

            const auto y_num_max = (unsigned int) access.y_num[level];
            const auto x_num_max = (unsigned int) access.x_num[level];
            const auto z_num_max = (unsigned int) access.z_num[level];

            auto it = apr.iterator();

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_, z_) firstprivate(it)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {
                for (x_ = 0; x_ < x_num_; x_++) {

                    const uint64_t offset_pc_data = x_num_ * z_ + x_;

                    const auto x_a = x_;
                    const auto z_a = z_;

                    if(remove_b[level].at(x_a,z_a,0)!=UINT64_MAX){

                        it.set_new_lzx(level,z_,x_);

                        auto mit = it.get_current_gap();

                        auto pcdKey = it.get_pcd_key();

                        auto& part_vec = parts_pcd.data[level][offset_pc_data];

                        auto pc = it.get_current_particle_cell();

                        for (int i = remove_b[level].at(x_a,z_a,0); i < (remove_e[level].at(x_a,z_a,0)+1); ++i) {
                            pc.y = buff.y[i];

                            access.find_particle_cell_pcdKey(pc,mit,pcdKey);

                            part_vec[pcdKey.local_ind]=(-1); // #FIXME how to deal with this?

                        }

                        int offset = 0;
                        for (int j = 0; j < part_vec.size(); ++j) {
                            auto val = part_vec[j];
                            part_vec[j+offset] = part_vec[j];

                            if(val == ((ImageType)-1)){
                                offset-=1;
                            }
                        }

                        int new_size = part_vec.size() + offset;

                        part_vec.resize(new_size);

                    }

                }
            }
        }
    }

    template<typename R,typename S>
    void update_particles_val(APR<R>& apr,ExtraParticleData<S>& particles,std::vector<PixelData<uint64_t>>& update_b,std::vector<PixelData<uint64_t>>& update_e,BufferPc& buff_pc,const S update_val){
        //
        //  Updates particles from the buffer (make more general) //do the change outside the function
        //
        //

        auto& access = apr.apr_access;
        uint64_t x_;
        uint64_t z_;

        for (uint64_t level = (access.level_min()); level <= access.level_max(); level++) {

            auto x_num_ = (unsigned int) access.x_num[level];
            auto z_num_ = (unsigned int) access.z_num[level];

            auto it = apr.iterator();

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_, z_) firstprivate(it)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {
                for (x_ = 0; x_ < x_num_; x_++) {

                    if(update_b[level].at(x_,z_,0)!=UINT64_MAX){

                        it.set_new_lzx(level,z_,x_);

                        auto mit = it.get_current_gap();
                        auto pc = it.get_current_particle_cell();

                        for (int i = update_b[level].at(x_,z_,0); i < (update_e[level].at(x_,z_,0)+1); ++i) {

                            pc.y = buff_pc.y[i];

                            access.find_particle_cell(pc,mit);

                            particles[pc.global_index] = update_val;

                        }

                    }

                }
            }
        }

    }


    template<typename R,typename S>
    void update_particles(APR<R>& apr,PartCellData<S>& parts_pcd,std::vector<S>& f_data,std::vector<PixelData<uint64_t>>& update_b,std::vector<PixelData<uint64_t>>& update_e,BufferPc& buff_pc,const bool forward){
        //
        //  Updates particles from the buffer (make more general) //do the change outside the function
        //
        //


        auto& access = apr.apr_access;
        uint64_t x_;
        uint64_t z_;

        for (uint64_t level = (access.level_min()); level <= access.level_max(); level++) {

            auto x_num_ = (unsigned int) access.x_num[level];
            auto z_num_ = (unsigned int) access.z_num[level];

            auto it = apr.iterator();

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_, z_) firstprivate(it)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {
                for (x_ = 0; x_ < x_num_; x_++) {

                    if(update_b[level].at(x_,z_,0)!=UINT64_MAX){

                        it.set_new_lzx(level,z_,x_);

                        auto mit = it.get_current_gap();
                        auto pcd_key = it.get_pcd_key();
                        auto pc = it.get_current_particle_cell();

                        for (int i = update_b[level].at(x_,z_,0); i < (update_e[level].at(x_,z_,0)+1); ++i) {

                            pc.y = buff_pc.y[i];

                            access.find_particle_cell_pcdKey(pc,mit,pcd_key);

                            float update_val = (f_data[i]/2)*(1 - 2*(f_data[i]%2));

                            if(forward) {
                                parts_pcd[pcd_key] += update_val;
                            } else {
                                parts_pcd[pcd_key] -= update_val;
                            }

                        }

                    }

                }
            }
        }

    }



    void read_pc(hid_t obj_id,std::string name,Buffer<ImageType>& parts,BufferPc& pcs){
        //
        // FIX ME NEEDS TO BE CHANGED TO START / END
        //

        hid_t type_parts = APRWriter::Hdf5Type<ImageType>::type();

        std::string d_name = name + "_fp";
        readData({type_parts, d_name.c_str()}, obj_id, parts.data.data());

        d_name = name + "_x";
        readData({H5T_NATIVE_UINT16, d_name.c_str()}, obj_id, pcs.x.data());

        d_name = name + "_z";
        readData({H5T_NATIVE_UINT16, d_name.c_str()}, obj_id, pcs.z.data());

        d_name = name + "_y";
        readData({H5T_NATIVE_UINT16, d_name.c_str()}, obj_id, pcs.y.data());

        d_name = name + "_l";
        readData({H5T_NATIVE_UINT8, d_name.c_str()}, obj_id, pcs.l.data());

    }


    void read_apr_init(const std::string &input_file_name){
        file_name = input_file_name;

        AprFile::Operation op;

        op = AprFile::Operation::READ;

        AprFile f(file_name, op, 0);

        uint64_t num_key_frames = get_dims(f.groupId,"keyframe_indices");
        key_frame_index_vector.resize(num_key_frames);

        readData({H5T_NATIVE_UINT64, "keyframe_indices"}, f.groupId, key_frame_index_vector.data());

        //how can I replace this?
        number_time_steps = key_frame_index_vector.back() + 1;

        read_apr_parameters(f.objectId,initial_parameters);
        read_access_info(f.objectId,initial_access_info);

        APR_buffer.resize(1);
    }

    void read_apr_delta_data(uint64_t key_frame){

        APRTimer timer(false);

        bool load_diff_data = true;

        if(key_frame == (key_frame_index_vector.size() - 1)){
            load_diff_data = false;
        } else if((key_frame_index_vector[key_frame+1] - key_frame_index_vector[key_frame]) == 1) {
            load_diff_data = false;
        }

        if(load_diff_data) {

            current_delta_frame = key_frame;

            timer.start_timer("init");

            AprFile::Operation op;

            op = AprFile::Operation::READ;

            AprFile f(file_name, op, key_frame, "dt");

            hid_t meta_data = f.objectId;

            /*
             * Get the cumulative sums for the various update variables for the updates throuhg time
             */

            hid_t type_totals = H5T_NATIVE_UINT64;

            uint64_t time_steps_to_next = get_dims(meta_data, "update_fp_num");


            update_f_totals.resize(time_steps_to_next);
            update_totals.resize(time_steps_to_next);

            add_f_totals.resize(time_steps_to_next);
            add_totals.resize(time_steps_to_next);

            remove_f_totals.resize(time_steps_to_next);
            remove_totals.resize(time_steps_to_next);

            readData({H5T_NATIVE_UINT64, "update_fp_num"}, meta_data, update_f_totals.data());
            readData({H5T_NATIVE_UINT64, "update_num"}, meta_data, update_totals.data());
            readData({H5T_NATIVE_UINT64, "add_fp_num"}, meta_data, add_f_totals.data());
            readData({H5T_NATIVE_UINT64, "add_num"}, meta_data, add_totals.data());
            readData({H5T_NATIVE_UINT64, "remove_fp_num"}, meta_data, remove_f_totals.data());
            readData({H5T_NATIVE_UINT64, "remove_num"}, meta_data, remove_totals.data());

            //for now lets read the whole dataset in.

            //first init
            update_buff.init(update_totals.back());
            update_f_buff.init(update_f_totals.back());

            add_buff.init(add_totals.back());
            add_f_buff.init(add_f_totals.back());

            remove_buff.init(remove_totals.back());
            remove_f_buff.init(remove_f_totals.back());

            timer.stop_timer();

            timer.start_timer("init read");

            //now read
            if (update_totals.back() > 0) {
                read_pc(meta_data, "update", update_f_buff, update_buff);
            }

            if (remove_totals.back() > 0) {
                read_pc(meta_data, "remove", remove_f_buff, remove_buff);
            }

            if (add_totals.back() > 0) {
                read_pc(meta_data, "add", add_f_buff, add_buff);
            }

            timer.stop_timer();

        }

    }

    void read_apr_key_frame(uint64_t key_frame){
        APRTimer timer;
        timer.verbose_flag = false;

        APR<ImageType> temp_apr;

        read_apr(temp_apr, file_name, false, 0, key_frame);
        current_time_step = key_frame_index_vector[key_frame];

        copy_parts_to_pcd(temp_apr,temp_apr.particles_intensities,current_particles);

        current_key_frame = key_frame;

        //setting things
        //copy_parts_to_pcd(org_apr, org_apr.particles_intensities, current_particles);

        APR_buffer[0].copy_from_APR(temp_apr);
        current_APR = &APR_buffer[0];

        changeTable.init(temp_apr.apr_access);

        auto &init_access = temp_apr.apr_access;

        uint64_t prev = 0;

        //Need to initialize the change table.
        for (uint64_t level = (init_access.level_min()); level <= init_access.level_max(); level++) {

            auto x_num_ = (unsigned int) init_access.x_num[level];
            auto z_num_ = (unsigned int) init_access.z_num[level];

            const bool max_level = (init_access.level_max() == level);

            auto &xz_init = init_access.global_index_by_level_and_zx_end[level];

            for (int i = 0; i < xz_init.size(); ++i) {
                changeTable.particle_sum[level].mesh[i] = xz_init[i] - prev;

                prev = xz_init[i];
            }

        }

        if(calculate_time_updated){
            //
            //  This maintains an additional particle property that stores when the particles value was last updated in terms of time point.
            //
            calculate_updated_time(true);

        }



    }

    uint64_t find_key_frame(uint64_t time_point){

        uint64_t key_frame_before=key_frame_index_vector.size()-1;

        for (int i = 0; i < key_frame_index_vector.size(); ++i) {
            if(key_frame_index_vector[i] > time_point){
                key_frame_before = i-1;
                break;
            }
        }

        return key_frame_before;
    }

    uint64_t find_key_frame_previous(uint64_t time_point){

        uint64_t key_frame_before=key_frame_index_vector.size()-1;

        for (int i = 0; i < key_frame_index_vector.size(); ++i) {
            if(key_frame_index_vector[i] >= time_point){
                key_frame_before = i;
                break;
            }
        }

        return key_frame_before;
    }

    void calculate_updated_time(bool key_frame,bool direction = true){

        if(key_frame){
            updated_t.data.resize(current_APR->total_number_particles());
            std::fill(updated_t.data.begin(),updated_t.data.end(),current_time_step);
        } else {

            update_particles_val(*current_APR,updated_t,changeTable.update_b,changeTable.update_e,update_buff,(uint16_t) current_time_step);
            if(current_direction) {
                update_particles_val(*current_APR, updated_t, changeTable.add_b, changeTable.add_e, add_buff,
                                     (uint16_t) current_time_step);
            } else{
                update_particles_val(*current_APR, updated_t, changeTable.remove_b, changeTable.remove_e, remove_buff,
                                     (uint16_t) current_time_step);
            }
        }

    }



    bool read_time_point(uint64_t time_point){
        //
        //
        //  Time stepping logic, moves forward or backwards and jumps to key frames depending on the current, and intended time point
        //
        //

        if(time_point >= number_time_steps){
            //out of range
            return false;
        }

        if(current_time_step == time_point){
            //do nothing
            return true;
        } else {

            if((time_point > current_time_step) || (current_key_frame >= key_frame_index_vector.size())) {
                current_direction = true;
                uint64_t before_key_frame = find_key_frame(time_point);

                if ((current_key_frame != before_key_frame) || (current_time_step < key_frame_index_vector[before_key_frame])) {
                    //load key frame
                    read_apr_key_frame(before_key_frame);
                    key_frame_loaded = true;

                } else{
                    key_frame_loaded = false;
                }

                if(current_time_step==time_point){
                    //loaded
                } else {

                    if (current_delta_frame != before_key_frame) {
                        read_apr_delta_data(before_key_frame);
                        key_frame_delta = 0;
                    }

                    //forward
                    while (current_time_step < time_point) {
                        read_next();
                    }
                }
            } else {
                current_direction = false;
                uint64_t after_key_frame = find_key_frame_previous(time_point);

                if (current_key_frame != (after_key_frame) || (current_time_step > key_frame_index_vector[after_key_frame])) {
                    //load key frame
                    read_apr_key_frame(after_key_frame);
                    key_frame_loaded = true;
                } else {
                    key_frame_loaded = false;
                }

                if(current_time_step==time_point){
                    //loaded
                } else {

                    if (current_delta_frame != (after_key_frame-1)) {
                        read_apr_delta_data((after_key_frame-1));
                    }

                    //forward
                    while (current_time_step > time_point) {
                        read_previous();
                    }
                }
            }
        }

        return false;

    }


    void read_previous(){
        //
        //  Moves APR to previous time-step by updated the -adds-removes-updates (reverses add and removes and updates)
        //

        current_time_step--;
        key_frame_delta = current_time_step - key_frame_index_vector[current_delta_frame] + 1;
        move(0);

        if(calculate_time_updated){
            //
            //  This maintains an additional particle property that stores when the particles value was last updated in terms of time point.
            //
            calculate_updated_time(false,false);

        }

    }

    void read_next() {
        //
        //  Moves APR to next time-step by updated the adds+removes+updates
        //

        current_time_step++;
        key_frame_delta = current_time_step - key_frame_index_vector[current_key_frame];
        move(1);

        if(calculate_time_updated){
            //
            //  This maintains an additional particle property that stores when the particles value was last updated in terms of time point.
            //
            calculate_updated_time(false,true);

        }

    }


    void update_access(BufferPc& r_buff,BufferPc& a_buff,std::vector<PixelData<uint64_t>>& remove_b,std::vector<PixelData<uint64_t>>& remove_e,std::vector<PixelData<uint64_t>>& add_b,std::vector<PixelData<uint64_t>>& add_e){

        int current_index = 0;

        //copy
        //APR_buffer[current_index].apr_access = APR_buffer[previous_index].apr_access;

        RandomAccess &new_access = APR_buffer[current_index].apr_access;
        RandomAccess &old_access = APR_buffer[current_index].apr_access; //redundant, #FIXME is there any logic to keeping this in for future extensions?

        uint64_t z_;
        uint64_t x_;

        const bool odd_end = (new_access.y_num.back()%2) != 0; //odd y-num;

        for (uint64_t level = (new_access.level_min()); level <= new_access.level_max(); level++) {

            auto x_num_ = (unsigned int) new_access.x_num[level];
            auto z_num_ = (unsigned int) new_access.z_num[level];
            auto y_num_ = (unsigned int) new_access.y_num[level];

            const bool max_level = (new_access.level_max() ==level );

            if (max_level) {
                //account for the APR, where the maximum level is down-sampled one
                x_num_ = (unsigned int) new_access.x_num[level - 1];
                z_num_ = (unsigned int) new_access.z_num[level - 1];
                y_num_ = (unsigned int) new_access.y_num[level - 1];
            }

            const auto y_num_max = (unsigned int) new_access.y_num[level];
            const auto x_num_max = (unsigned int) new_access.x_num[level];
            const auto z_num_max = (unsigned int) new_access.z_num[level];

            std::vector<uint8_t> y_temp;
            y_temp.resize(y_num_,0);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_,z_) firstprivate(y_temp)
#endif
            for (z_ = 0; z_ < z_num_; z_++) {
                for (x_ = 0; x_ < x_num_; x_++) {

                    const uint64_t offset_pc_data = x_num_ * z_ + x_;

                    auto x_a = x_;
                    auto z_a = z_;

                    if(max_level){
                        x_a = 2*x_;
                        z_a = 2*z_;
                    }

                    if ((add_b[level].at(x_a,z_a,0)!=UINT64_MAX) || (remove_b[level].at(x_a,z_a,0)!=UINT64_MAX) ) {

                        std::fill(y_temp.begin(),y_temp.end(),0);

                        if (old_access.gap_map.data[level][offset_pc_data].size() > 0) {

                            //so the loops are over y/2, however the y stored in the map are the true y. hence the need for the max level conditions (This is utilizing the Equivalence Optimization).
                            if(max_level) {
                                //requried change in indexing here.
                                for (auto const &element : old_access.gap_map.data[level][offset_pc_data][0].map) {
                                    for (int y = (element.first/2); y <= (element.second.y_end/2); ++y) {
                                        y_temp[y] = 1;
                                    }
                                }

                            } else {
                                for (auto const &element : old_access.gap_map.data[level][offset_pc_data][0].map) {
                                    for (int y = element.first; y <= element.second.y_end; ++y) {
                                        y_temp[y] = 1;
                                    }
                                }
                            }
                        } else {

                        }

                        if(max_level) {
                            for (int j = add_b[level].at(x_a, z_a, 0);
                                 j < (add_e[level].at(x_a, z_a, 0) + 1); ++j) {
                                y_temp[a_buff.y[j] / 2] = 1;
                            }

                            for (int j = remove_b[level].at(x_a, z_a, 0);
                                 j < (remove_e[level].at(x_a, z_a, 0) + 1); ++j) {
                                y_temp[r_buff.y[j] / 2] = 0;
                            }
                        } else {
                            for (int j = add_b[level].at(x_a, z_a, 0);
                                 j < (add_e[level].at(x_a, z_a, 0) + 1); ++j) {
                                y_temp[a_buff.y[j]] = 1;
                            }

                            for (int j = remove_b[level].at(x_a, z_a, 0);
                                 j < (remove_e[level].at(x_a, z_a, 0) + 1); ++j) {
                                y_temp[r_buff.y[j]] = 0;
                            }



                        }

                        //now insert into the dataset
                        if(new_access.gap_map.data[level][offset_pc_data].size()==0){
                            new_access.gap_map.data[level][offset_pc_data].resize(1);
                        }

                        auto &new_map = new_access.gap_map.data[level][offset_pc_data][0].map;

                        new_map.clear();

                        uint16_t current = 0;
                        uint16_t previous = 0;

                        YGap_map gap;
                        gap.y_end = 0;
                        gap.global_index_begin_offset = 0;
                        uint16_t current_key=0;

                        uint16_t cumsum = 0;
                        uint16_t current_offset = 0;

                        for (uint16_t y = 0; y < y_num_; ++y) {
                            uint8_t status = y_temp[y];
                            if (status > 0) {
                                current = 1;
                                if (previous == 0) {
                                    if (max_level) {
                                        current_key = 2*y;
                                    } else {
                                        current_key = y;
                                    }
                                    current_offset = cumsum;
                                }
                                if(max_level){
                                    cumsum+=2;
                                } else{
                                    cumsum++;
                                }

                            } else {
                                current = 0;
                                if (previous == 1) {
                                    if (max_level) {
                                        gap.y_end = (2*y - 1);
                                    } else {
                                        gap.y_end = (y - 1);
                                    }
                                    gap.global_index_begin_offset = current_offset;
                                    new_map.insert(new_map.end(),{current_key,gap});
                                }
                            }

                            previous = current;
                        }
                        //end node
                        if (previous == 1) {
                            //insert
                            gap.global_index_begin_offset = current_offset;
                            gap.y_end = (uint16_t)(y_num_max-1);
                            new_map.insert(new_map.end(),{current_key,gap});

                            if(max_level){
                                if(odd_end){
                                    cumsum--;
                                }
                            }

                        }

                        if(max_level){
                            //unsigned int number_rows = ( unsigned int) 1*((2*x_+1) < x_num_max) + 1*((2*z_+1) < z_num_max) +  1*((2*z_+1) < z_num_max)*((2*x_+1) < x_num_max);

                            //changeTable.particle_sum[level].at(x_, z_, 0) = cumsum * number_rows;
                            const auto x2 = 2*x_+1;
                            const auto z2 = 2*z_+1;

                            const uint64_t v1 = (uint64_t) ( (x2) < x_num_max);
                            const uint64_t v2 = (uint64_t) (z2 < z_num_max) ;

                            changeTable.particle_sum[level].at(x_, z_, 0) = cumsum*(v1 + v2 + v1*v2 + 1);

                            //changeTable.particle_sum[level].at(x_, z_, 0) = cumsum;
                        } else {
                            changeTable.particle_sum[level].at(x_, z_, 0) = cumsum;
                        }

                        //removed
                        if( new_map.size()==0){
                            new_access.gap_map.data[level][offset_pc_data].resize(0);
                        }

                    } else {
                        //if the new map is empty, we need to copy across, this is for intialization, assumes that those existing in the new are the same as the old. (i.e. its from the prev timestep)

                        if (old_access.gap_map.data[level][offset_pc_data].size() > 0) {

                            if(new_access.gap_map.data[level][offset_pc_data].size() == 0) {
                                uint64_t begin = 0;
                                if(offset_pc_data == 0){
                                    if(level == new_access.level_min()){
                                        begin =  0;
                                    } else {
                                        begin = old_access.global_index_by_level_and_zx_end[level].back();
                                    }
                                } else {
                                    begin  = old_access.global_index_by_level_and_zx_end[level][offset_pc_data-1];
                                }

                                changeTable.particle_sum[level].at(x_,z_,0) = old_access.global_index_by_level_and_zx_end[level][offset_pc_data]-begin;

                                new_access.gap_map.data[level][offset_pc_data].resize(1);

                                new_access.gap_map.data[level][offset_pc_data][0].map = old_access.gap_map.data[level][offset_pc_data][0].map;
                            }
                        }
                    }



                }
            }
        }

        /*
         *
         * These cumumlative sum totals are required for iteration and storage of the global indices without the use of uint64 (the values are stored in a cache and updated each timestep))
         * Then the cumulative sum is done once over an array, allowing good performance for the serial task.
         */

        for (uint64_t level = (new_access.level_min()); level <= new_access.level_max(); level++) {

            auto x_num_ = (unsigned int) new_access.x_num(level);
            auto z_num_ = (unsigned int) new_access.z_num(level);

            const bool max_level = (new_access.level_max() == level);

            if (max_level) {
                //account for the APR, where the maximum level is down-sampled one
                x_num_ = (unsigned int) new_access.x_num(level - 1);
                z_num_ = (unsigned int) new_access.z_num(level - 1);
            }

            new_access.global_index_by_level_and_zx_end[level].resize(z_num_ * x_num_);

            if(level != new_access.level_min()) {
                //need to add it to have the cumumlative sum go across levels
                changeTable.particle_sum[level].mesh[0] += new_access.global_index_by_level_and_zx_end[level-1].back();
            }

            std::partial_sum (changeTable.particle_sum[level].mesh.begin(), changeTable.particle_sum[level].mesh.end(), new_access.global_index_by_level_and_zx_end[level].begin());

            if(level != new_access.level_min()) {
                //need to remove it to make the totals still valid for the next step.
                changeTable.particle_sum[level].mesh[0] -= new_access.global_index_by_level_and_zx_end[level-1].back();
            }

        }

        //update the total number of particles
        new_access.total_number_particles = new_access.global_index_by_level_and_zx_end[new_access.level_max()].back();

    }

    void move(const bool direction){
        //
        //  direction = 1, moves forward a frame, direction = -1 moves backwards
        //

        auto current_index = 0;

        current_APR = &APR_buffer[current_index];

        prev_apr.copy_from_APR(*current_APR);

        // add loop

        uint64_t add_begin = 0;
        uint64_t add_end = 0;

        uint64_t remove_begin = 0;
        uint64_t remove_end = 0;

        uint64_t update_begin = 0;
        uint64_t update_end = 0;

        uint64_t add_f_begin = 0;
        uint64_t add_f_end = 0;

        uint64_t remove_f_begin = 0;
        uint64_t remove_f_end = 0;

        uint64_t update_f_begin = 0;
        uint64_t update_f_end = 0;

        if (key_frame_delta == 1) {
            add_end = add_totals[key_frame_delta - 1];
            remove_end = remove_totals[key_frame_delta - 1];

            update_end = update_totals[key_frame_delta - 1];

            add_f_end = add_f_totals[key_frame_delta - 1];
            remove_f_end = remove_f_totals[key_frame_delta - 1];

            update_f_end = update_f_totals[key_frame_delta - 1];

        } else {
            add_begin = add_totals[key_frame_delta - 2];
            add_end = add_totals[key_frame_delta - 1];

            remove_begin = remove_totals[key_frame_delta - 2];
            remove_end = remove_totals[key_frame_delta - 1];

            update_begin = update_totals[key_frame_delta - 2];
            update_end = update_totals[key_frame_delta - 1];

            add_f_begin = add_f_totals[key_frame_delta - 2];
            add_f_end = add_f_totals[key_frame_delta - 1];

            remove_f_begin = remove_f_totals[key_frame_delta - 2];
            remove_f_end = remove_f_totals[key_frame_delta - 1];

            update_f_begin = update_f_totals[key_frame_delta - 2];
            update_f_end = update_f_totals[key_frame_delta - 1];

        }

        update_global_index.resize(update_f_end - update_f_begin);
        remove_global_index.resize(remove_f_end - remove_f_begin);
        add_global_index.resize(add_f_end - add_f_begin);

        changeTable.max();

        for (auto j = add_begin; j < add_end; ++j) {

            uint16_t x = add_buff.x[j];
            uint16_t z = add_buff.z[j];
            uint16_t y = add_buff.y[j];
            auto l = add_buff.l[j];

            if (changeTable.add_b[l].at(x, z, 0) == UINT64_MAX) {
                changeTable.add_b[l].at(x, z, 0) = j;
                changeTable.add_e[l].at(x, z, 0) = j;
            } else {
                changeTable.add_e[l].at(x, z, 0) = j;
            }



        }
        for (auto j = remove_begin; j < remove_end; ++j) {

            auto x = remove_buff.x[j];
            auto z = remove_buff.z[j];
            auto y = remove_buff.y[j];
            auto l = remove_buff.l[j];


            if (changeTable.remove_b[l].at(x, z, 0) == UINT64_MAX) {
                changeTable.remove_b[l].at(x, z, 0) = j;
                changeTable.remove_e[l].at(x, z, 0) = j;
            } else {
                changeTable.remove_e[l].at(x, z, 0) = j;
            }



        }

        for (auto j = update_begin; j < update_end; ++j) {

            auto x = update_buff.x[j];
            auto z = update_buff.z[j];
            auto y = update_buff.y[j];
            auto l = update_buff.l[j];

            if (changeTable.update_b[l].at(x, z, 0) == UINT64_MAX) {
                changeTable.update_b[l].at(x, z, 0) = j;
                changeTable.update_e[l].at(x, z, 0) = j;
            } else {
                changeTable.update_e[l].at(x, z, 0) = j;
            }
        }

        if(direction) {
            //forwards
            /*
            *  Remove particles, before the data-structures get updated
            *
            */

            remove_particles(APR_buffer[current_index], current_particles, changeTable.remove_b, changeTable.remove_e,
                             remove_buff);

            /*
            *  Update Access Datastructures
            *
            */

            update_access(remove_buff, add_buff, changeTable.remove_b, changeTable.remove_e, changeTable.add_b,
                          changeTable.add_e);

            /*
            * Add Particles
            *
            */

            add_particles(APR_buffer[current_index], current_particles, changeTable.add_b, changeTable.add_e, add_buff,
                          add_f_buff.data);

            /*
            * Update Particles
            *
            */

            update_particles(APR_buffer[current_index], current_particles,update_f_buff.data,changeTable.update_b, changeTable.update_e, update_buff, direction);
        } else {
            //backward (Note same as above, except the add and remove have been flipped, and update particles subtracts)
            /*
           *  Remove particles, before the data-structures get updated
           *
           */

            remove_particles(APR_buffer[current_index], current_particles, changeTable.add_b, changeTable.add_e,
                             add_buff);

            /*
            *  Update Access Datastructures
            *
            */

            update_access(add_buff, remove_buff, changeTable.add_b, changeTable.add_e, changeTable.remove_b,
                          changeTable.remove_e);

            /*
            * Add Particles
            *
            */

            add_particles(APR_buffer[current_index], current_particles, changeTable.remove_b, changeTable.remove_e, remove_buff,
                          remove_f_buff.data);


            /*
            * Update Particles
            *
            */

            update_particles(APR_buffer[current_index], current_particles,update_f_buff.data,changeTable.update_b, changeTable.update_e, update_buff, direction);
        }

    }




    template<typename T, typename R>
    void update_to_pixel_img(PixelData<T> &pixelImg, BufferPc &buff, std::vector<R> &buff_f,
                             std::vector<uint64_t> &totals_pc,const uint64_t delta,const uint64_t max_level,const int delta_update,const T const_val = 0) {

        uint64_t add_begin = 0;
        uint64_t add_end = 0;

        if (delta == 1) {
            add_end = totals_pc[delta - 1];

        } else {
            add_begin = totals_pc[delta - 2];
            add_end = totals_pc[delta - 1];
        }

        for (auto i = add_begin; i < add_end; ++i) {

            auto level = buff.l[i];
            auto x = buff.x[i];
            auto y = buff.y[i];
            auto z = buff.z[i];

            auto f = buff_f[i];

            if(delta_update > 0){
                f = (f/2)*(1 - 2*(f%2));
            }

            const int step_size = pow((int) 2 , max_level - level);

            //
            //  Parallel loop over level
            //
            if (level == max_level) {

                if(delta_update == 0){
                    pixelImg.at(y, x, z) = f;
                } else if (delta_update == 1){
                    pixelImg.at(y, x, z) += f;
                } else if (delta_update == 2){
                    pixelImg.at(y, x, z) -= f;
                }

            } else {
                int dim1 = y * step_size;
                int dim2 = x * step_size;
                int dim3 = z * step_size;

                const int offset_max_dim1 = std::min((int) pixelImg.y_num, (int) (dim1 + step_size));
                const int offset_max_dim2 = std::min((int) pixelImg.x_num, (int) (dim2 + step_size));
                const int offset_max_dim3 = std::min((int) pixelImg.z_num, (int) (dim3 + step_size));

                if(delta_update == 0){
                    for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                        for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                            for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                pixelImg.mesh[i + (k) * pixelImg.y_num + q * pixelImg.y_num * pixelImg.x_num] = f;
                            }
                        }
                    }
                } else if (delta_update == 1){
                    for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                        for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                            for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                pixelImg.mesh[i + (k) * pixelImg.y_num + q * pixelImg.y_num * pixelImg.x_num] += f;
                            }
                        }
                    }
                } else if (delta_update == 2){
                    for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                        for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                            for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                pixelImg.mesh[i + (k) * pixelImg.y_num + q * pixelImg.y_num * pixelImg.x_num] -= f;
                            }
                        }
                    }
                } else if (delta_update == 3){
                    for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                        for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                            for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                pixelImg.mesh[i + (k) * pixelImg.y_num + q * pixelImg.y_num * pixelImg.x_num] = const_val;
                            }
                        }
                    }
                }


            }
        }

    }

    template<typename T>
    void direct_update_interp(PixelData<T> &pixelImg) {


        if(key_frame_loaded || direct_updated == (UINT64_MAX)) {

            APR<ImageType> temp_apr;

            read_apr(temp_apr, file_name, false, 0, current_key_frame);

            temp_apr.interp_img(pixelImg, temp_apr.particles_intensities);

            direct_updated = key_frame_index_vector[current_key_frame];
        }


        if (pixelImg.mesh.size() !=
            (initial_access_info.org_dims[0] * initial_access_info.org_dims[1] * initial_access_info.org_dims[2])) {
            //check if the img needs to be initialized
            pixelImg.init(initial_access_info.org_dims[0], initial_access_info.org_dims[1],
                          initial_access_info.org_dims[2]);
        }

        //updates
        if (current_direction) {

            while (direct_updated < current_time_step) {

                direct_updated++;

                uint64_t current_delta = direct_updated - key_frame_index_vector[current_key_frame];

                update_to_pixel_img(pixelImg, update_buff, update_f_buff.data,
                                    update_totals, current_delta, initial_access_info.level_max(), 1);

                update_to_pixel_img(pixelImg, add_buff, add_f_buff.data,
                                    add_totals, current_delta, initial_access_info.level_max(), 0);


            }
        } else {
            while (direct_updated > current_time_step) {
                direct_updated--;

                uint64_t current_delta = direct_updated - key_frame_index_vector[current_delta_frame] + 1;

                update_to_pixel_img(pixelImg, update_buff, update_f_buff.data,
                                    update_totals, current_delta, initial_access_info.level_max(), 2);

                update_to_pixel_img(pixelImg, remove_buff, remove_f_buff.data,
                                    remove_totals, current_delta, initial_access_info.level_max(), 0);


            }
        }


    }




    template<typename T>
    void flatten_pcd(PartCellData<T>& partCellData,std::vector<T>& flatData){
        //
        //  Copy the data from partCellData to a contiguous array
        //

        std::vector<std::vector<uint64_t>> indx;
        indx.resize(partCellData.depth_max+1);

        uint64_t prev = 0;

        //Serial indexing loop, to enable the parrallel loop that follows.
        for (int i = partCellData.depth_min; i <= partCellData.depth_max; ++i) {

            indx[i].resize(partCellData.data[i].size());

            for (int j = 0; j < partCellData.data[i].size(); ++j) {
                indx[i][j] = prev;
                prev = partCellData.data[i][j].size() + prev;
            }
        }

        flatData.resize(prev);

        for (auto i = partCellData.depth_min; i <= partCellData.depth_max; ++i) {

            int j = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(j)
#endif
            for (j = 0; j < partCellData.data[i].size(); ++j) {
                uint64_t idx = indx[i][j];
                std::copy(partCellData.data[i][j].begin(),partCellData.data[i][j].end(),flatData.begin() + idx);
            }
        }

    }

    template<typename T,typename R>
    void flatten_pcd_gen_spatial_info(PartCellData<T>& partCellData,std::vector<R>& flatData,std::string pc_property) {
        //
        //  Copy the data from partCellData to a contiguous array
        //

        std::vector<std::vector<uint64_t>> indx;
        indx.resize(partCellData.depth_max + 1);

        uint64_t prev = 0;

        auto ind = 0;

        if (pc_property == "x") {
            ind = 1;
        } else if (pc_property == "z") {
            ind = 2;
        } else if (pc_property == "level") {
            ind = 3;
        } else {
            std::cerr << "Particle Property Selection Error" << std::endl;
        }

        const int selection = ind;

        //Serial indexing loop, to enable the parrallel loop that follows.
        for (auto level = partCellData.depth_min; level <= partCellData.depth_max; ++level) {

            indx[level].resize(partCellData.data[level].size());

            for (int j = 0; j < partCellData.data[level].size(); ++j) {
                indx[level][j] = prev;
                prev = partCellData.data[level][j].size() + prev;
            }
        }

        flatData.resize(prev);

        for (auto i = partCellData.depth_min; i <= partCellData.depth_max; ++i) {

            int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z)
#endif
            for (z = 0; z < partCellData.z_num[i]; ++z) {
                for (int x = 0; x < partCellData.x_num[i]; ++x) {

                    auto j = z * partCellData.x_num[i] + x;

                    auto idx_b = indx[i][j];
                    auto idx_e = indx[i][j] + partCellData.data[i][j].size();

                    if (selection == 1) {
                        std::fill(flatData.begin() + idx_b, flatData.begin() + idx_e, x);
                    } else if (selection == 2) {
                        std::fill(flatData.begin() + idx_b, flatData.begin() + idx_e, z);
                    } else if (selection == 3) {
                        std::fill(flatData.begin() + idx_b, flatData.begin() + idx_e, i);
                    }
                }
            }

        }
    }


    void write_particle_cells_append(TimeData<ImageType>& tdata,hid_t location,unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 4, unsigned int blosc_shuffle=1){


        std::vector<uint16_t> space_buffer;
        std::vector<uint8_t> level_buffer;

        std::vector<uint64_t> sz={0};

        hid_t type_index = H5T_NATIVE_UINT16;

        //REMOVE
        //y
        flatten_pcd(*tdata.remove_y,space_buffer);
        writeDataAppend({type_index, "remove_y"}, location, space_buffer, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);

        //x
        flatten_pcd_gen_spatial_info(*tdata.remove_y,space_buffer,"x");
        writeDataAppend({type_index, "remove_x"}, location,space_buffer, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        //z
        flatten_pcd_gen_spatial_info(*tdata.remove_y,space_buffer,"z");
        sz[0] = writeDataAppend({type_index, "remove_z"}, location, space_buffer, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        //level
        flatten_pcd_gen_spatial_info(*tdata.remove_y,level_buffer,"level");
        writeDataAppend({H5T_NATIVE_UINT8, "remove_l"}, location, level_buffer, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);

        writeDataAppend({H5T_NATIVE_UINT64, "remove_num"}, location, sz, blosc_comp_type,
                        blosc_comp_level,
                        blosc_shuffle);

        //ADD
        //y
        flatten_pcd(*tdata.add_y,space_buffer);
        writeDataAppend({type_index, "add_y"}, location, space_buffer, blosc_comp_type, blosc_comp_level,
                        blosc_shuffle);

        //x
        flatten_pcd_gen_spatial_info(*tdata.add_y,space_buffer,"x");
        writeDataAppend({type_index, "add_x"}, location,space_buffer, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        //z
        flatten_pcd_gen_spatial_info(*tdata.add_y,space_buffer,"z");
        sz[0] = writeDataAppend({type_index, "add_z"}, location, space_buffer, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        //level
        flatten_pcd_gen_spatial_info(*tdata.add_y,level_buffer,"level");
        writeDataAppend({H5T_NATIVE_UINT8, "add_l"}, location, level_buffer, blosc_comp_type, blosc_comp_level,
                        blosc_shuffle);

        writeDataAppend({H5T_NATIVE_UINT64, "add_num"}, location, sz, blosc_comp_type,
                        blosc_comp_level,
                        blosc_shuffle);

        //UPDATE
        //y
        flatten_pcd(*tdata.update_y,space_buffer);
        writeDataAppend({type_index, "update_y"}, location, space_buffer, blosc_comp_type, blosc_comp_level,
                        blosc_shuffle);

        //x
        flatten_pcd_gen_spatial_info(*tdata.update_y,space_buffer,"x");
        writeDataAppend({type_index, "update_x"}, location,space_buffer, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        //z
        flatten_pcd_gen_spatial_info(*tdata.update_y,space_buffer,"z");
        sz[0] = writeDataAppend({type_index, "update_z"}, location, space_buffer, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        //level
        flatten_pcd_gen_spatial_info(*tdata.update_y,level_buffer,"level");
        writeDataAppend({H5T_NATIVE_UINT8, "update_l"}, location, level_buffer, blosc_comp_type, blosc_comp_level,
                        blosc_shuffle);

        writeDataAppend({H5T_NATIVE_UINT64, "update_num"}, location, sz, blosc_comp_type,
                        blosc_comp_level,
                        blosc_shuffle);

    }

    FileSizeInfo write_key_frame(APR<ImageType>& apr, const std::string &save_loc, const std::string &file_name,uint64_t key_frame_num,uint64_t global_index) {
        APRCompress<ImageType> apr_compressor;
        apr_compressor.set_compression_type(0);
        delta_num = 0;
        key_frame=key_frame_num;
        key_frame_index = global_index;

        FileSizeInfo fileSizeInfo = write_apr(apr, save_loc, file_name, apr_compressor,BLOSC_ZSTD,4,1,false,true);

        AprFile::Operation op;
        std::string hdf5_file_name = save_loc + file_name + "_apr.h5";
        op = APRWriter::AprFile::Operation::WRITE_APPEND;
        APRWriter::AprFile f(hdf5_file_name, op, 0);

        //append value.
        std::vector<uint64_t> sz={key_frame_index};
        hid_t type_index = H5T_NATIVE_UINT64;
        writeDataAppend({type_index, "keyframe_indices"}, f.groupId, sz, BLOSC_ZSTD,
                        4,
                        1);

        return fileSizeInfo;
    }

    FileSizeInfoTime write_time_step(TimeData<ImageType>& timeData, const std::string &save_loc, const std::string &file_name, unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 4, unsigned int blosc_shuffle=1,bool write_tree = false) {
        APRTimer write_timer;
        write_timer.verbose_flag = true;

        std::string hdf5_file_name = save_loc + file_name + "_apr.h5";


        AprFile::Operation op;

        if (write_tree) {
            op = APRWriter::AprFile::Operation::WRITE_WITH_TREE;
        } else {
            op = APRWriter::AprFile::Operation::WRITE;
        }

        unsigned int t = delta_num;

        if(t>=1){
            op = APRWriter::AprFile::Operation::WRITE_APPEND;
        } else {

        }

        APRWriter::AprFile f(hdf5_file_name, op, key_frame,"dt");


        delta_num++;

        FileSizeInfo fileSizeInfo1;
        FileSizeInfoTime fzt;
        if (!f.isOpened()) return fzt;

        hid_t meta_location = f.groupId;

        meta_location = f.objectId;

        hid_t type = APRWriter::Hdf5Type<ImageType>::type();

        hid_t type_index = H5T_NATIVE_UINT64;

        auto o_size = f.getFileSize();

        std::vector<uint64_t> sz={0};

        std::vector<ImageType> part_buffer;
        flatten_pcd(*timeData.update_fp,part_buffer);

        sz[0] = writeDataAppend({type, "update_fp"}, meta_location, part_buffer, blosc_comp_type,
                            blosc_comp_level,
                            blosc_shuffle);

        writeDataAppend({type_index, "update_fp_num"}, meta_location, sz, blosc_comp_type,
                        blosc_comp_level,
                        blosc_shuffle);

        flatten_pcd(*timeData.add_fp,part_buffer);
        sz[0] =writeDataAppend({type, "add_fp"}, meta_location, part_buffer, blosc_comp_type, blosc_comp_level,
                          blosc_shuffle);

        writeDataAppend({type_index, "add_fp_num"}, meta_location, sz, blosc_comp_type,
                        blosc_comp_level,
                        blosc_shuffle);

        flatten_pcd(*timeData.remove_fp,part_buffer);
        sz[0] =writeDataAppend({type, "remove_fp"}, meta_location, part_buffer, blosc_comp_type,
                          blosc_comp_level, blosc_shuffle);

        writeDataAppend({type_index, "remove_fp_num"}, meta_location, sz, blosc_comp_type,
                        blosc_comp_level,
                        blosc_shuffle);

        write_particle_cells_append(timeData, meta_location);

        // ------------- output the file size -------------------
        auto file_size = f.getFileSize();
        double sizeMB = file_size / 1e6;

        std::cout << "HDF5 Total Filesize: " << sizeMB << " MB\n" << "Writing Complete" << std::endl;

        return fzt;
    }


};


#endif //APR_TIME_APRTIMEIO_HPP
