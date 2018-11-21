//
// Created by cheesema on 14.01.18.
//

#ifndef APRWRITER_HPP
#define APRWRITER_HPP

#include "hdf5functions_blosc.h"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRAccess.hpp"
#include "ConfigAPR.h"
#include <numeric>
#include <memory>
#include <data_structures/APR/APR.hpp>


struct FileSizeInfo {
    float total_file_size=0;
    float intensity_data=0;
    float access_data=0;
};

struct FileSizeInfoTime {
    float total_file_size=0;
    float update_fp = 0;
    float update_index = 0;
    float add_index = 0;
    float add_fp = 0;
    float remove_index = 0;
    float remove_fp = 0;
};


struct AprType {hid_t hdf5type; const char * const typeName;};
namespace AprTypes  {

    const AprType TotalNumberOfParticlesType = {H5T_NATIVE_UINT64, "total_number_particles"};
    const AprType TotalNumberOfGapsType = {H5T_NATIVE_UINT64, "total_number_gaps"};
    const AprType TotalNumberOfNonEmptyRowsType = {H5T_NATIVE_UINT64, "total_number_non_empty_rows"};
    const AprType VectorSizeType = {H5T_NATIVE_UINT64, "type_vector_size"};
    const AprType NumberOfXType = {H5T_NATIVE_UINT64, "x_num"};
    const AprType NumberOfYType = {H5T_NATIVE_UINT64, "y_num"};
    const AprType NumberOfZType = {H5T_NATIVE_UINT64, "z_num"};
    const AprType MinLevelType = {H5T_NATIVE_UINT64, "level_min"};
    const AprType MaxLevelType = {H5T_NATIVE_UINT64, "level_max"};
    const AprType LambdaType = {H5T_NATIVE_FLOAT, "lambda"};
    const AprType CompressionType = {H5T_NATIVE_INT, "compress_type"};
    const AprType QuantizationFactorType = {H5T_NATIVE_FLOAT, "quantization_factor"};
    const AprType SigmaThType = {H5T_NATIVE_FLOAT, "sigma_th"};
    const AprType SigmaThMaxType = {H5T_NATIVE_FLOAT, "sigma_th_max"};
    const AprType IthType = {H5T_NATIVE_FLOAT, "I_th"};
    const AprType DxType = {H5T_NATIVE_FLOAT, "dx"};
    const AprType DyType = {H5T_NATIVE_FLOAT, "dy"};
    const AprType DzType = {H5T_NATIVE_FLOAT, "dz"};
    const AprType PsfXType = {H5T_NATIVE_FLOAT, "psfx"};
    const AprType PsfYType = {H5T_NATIVE_FLOAT, "psfy"};
    const AprType PsfZType = {H5T_NATIVE_FLOAT, "psfz"};
    const AprType RelativeErrorType = {H5T_NATIVE_FLOAT, "rel_error"};
    const AprType BackgroundIntensityEstimateType = {H5T_NATIVE_FLOAT, "background_intensity_estimate"};
    const AprType NoiseSdEstimateType = {H5T_NATIVE_FLOAT, "noise_sd_estimate"};
    const AprType NumberOfLevelXType = {H5T_NATIVE_INT, "x_num_"};
    const AprType NumberOfLevelYType = {H5T_NATIVE_INT, "y_num_"};
    const AprType NumberOfLevelZType = {H5T_NATIVE_INT, "z_num_"};
    const AprType MapGlobalIndexType = {H5T_NATIVE_INT16, "map_global_index"};
    const AprType MapYendType = {H5T_NATIVE_INT16, "map_y_end"};
    const AprType MapYbeginType = {H5T_NATIVE_INT16, "map_y_begin"};
    const AprType MapNumberGapsType = {H5T_NATIVE_INT16, "map_number_gaps"};
    const AprType MapLevelType = {H5T_NATIVE_UINT8, "map_level"};
    const AprType MapXType = {H5T_NATIVE_INT16, "map_x"};
    const AprType MapZType = {H5T_NATIVE_INT16, "map_z"};
    const AprType ParticleCellType = {H5T_NATIVE_UINT8, "particle_cell_type"};
    const AprType NameType = {H5T_C_S1, "name"};
    const AprType GitType = {H5T_C_S1, "githash"};

    const char * const ParticleIntensitiesType = "particle_intensities"; // type read from file
    const char * const ExtraParticleDataType = "extra_particle_data"; // type read from file
    const char * const ParticlePropertyType = "particle property"; // user defined type

    // Paraview specific
    const AprType ParaviewXType = {H5T_NATIVE_UINT16, "x"};
    const AprType ParaviewYType = {H5T_NATIVE_UINT16, "y"};
    const AprType ParaviewZType = {H5T_NATIVE_UINT16, "z"};
    const AprType ParaviewLevelType = {H5T_NATIVE_UINT8, "level"};
    const AprType ParaviewTypeType = {H5T_NATIVE_UINT8, "type"};
}


class APRWriter {

protected:
    unsigned int current_t = 0;


public:

    size_t get_number_time_steps(const std::string &file_name){
        AprFile::Operation op;


        size_t number_time_steps = 0;

        op = AprFile::Operation::READ;

        AprFile f(file_name, op,0);

        if (!f.isOpened()) return 0;

        hsize_t num_obj;

        H5Gget_num_objs(f.groupId,&num_obj);

        number_time_steps = num_obj;

        return number_time_steps;

    }


    template<typename ImageType>
    void read_apr(APR<ImageType>& apr, const std::string &file_name,bool build_tree = false, int max_level_delta = 0,unsigned int time_point = UINT32_MAX) {

        APRTimer timer;
        timer.verbose_flag = false;

        APRTimer timer_f;
        timer_f.verbose_flag = false;

        AprFile::Operation op;

        if(build_tree){
            op = AprFile::Operation::READ_WITH_TREE;
        } else {
            op = AprFile::Operation::READ;
        }

        unsigned int t = 0;

        if(time_point!=UINT32_MAX){
            t = time_point;
        }

        AprFile f(file_name, op,t);

        hid_t meta_data = f.groupId;

        if(time_point!=UINT32_MAX){
            meta_data = f.objectId;
        }


        if (!f.isOpened()) return;

        // ------------- read metadata --------------------------
        char string_out[100] = {0};
        hid_t attr_id = H5Aopen(meta_data,"name",H5P_DEFAULT);
        hid_t atype = H5Aget_type(attr_id);
        hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);
        H5Aread(attr_id, atype_mem, string_out) ;
        H5Aclose(attr_id);
        apr.name= string_out;

        // check if the APR structure has already been read in, the case when a partial load has been done, now only the particles need to be read
        uint64_t old_particles = apr.apr_access.total_number_particles;
        uint64_t old_gaps = apr.apr_access.total_number_gaps;

        readAttr(AprTypes::TotalNumberOfParticlesType, meta_data, &apr.apr_access.total_number_particles);
        readAttr(AprTypes::TotalNumberOfGapsType, meta_data, &apr.apr_access.total_number_gaps);

        readAttr(AprTypes::MaxLevelType, meta_data, &apr.apr_access.l_max);
        readAttr(AprTypes::MinLevelType, meta_data, &apr.apr_access.l_min);

        int compress_type;
        readAttr(AprTypes::CompressionType, meta_data, &compress_type);
        float quantization_factor;
        readAttr(AprTypes::QuantizationFactorType, meta_data, &quantization_factor);


        bool read_structure = true;

        if((old_particles == apr.apr_access.total_number_particles) && (old_gaps == apr.apr_access.total_number_gaps) && (build_tree)){
            read_structure = false;

        }

        //incase you ask for to high a level delta
        max_level_delta = std::max(max_level_delta,0);
        max_level_delta = std::min(max_level_delta,(int)apr.apr_access.level_max());

        if(read_structure) {
            readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, meta_data, &apr.apr_access.total_number_non_empty_rows);
            uint64_t type_size;
            readAttr(AprTypes::VectorSizeType, meta_data, &type_size);
            readAttr(AprTypes::NumberOfYType, meta_data, &apr.apr_access.org_dims[0]);
            readAttr(AprTypes::NumberOfXType, meta_data, &apr.apr_access.org_dims[1]);
            readAttr(AprTypes::NumberOfZType,meta_data, &apr.apr_access.org_dims[2]);

            readAttr(AprTypes::LambdaType, meta_data, &apr.parameters.lambda);

            readAttr(AprTypes::SigmaThType, meta_data, &apr.parameters.sigma_th);
            readAttr(AprTypes::SigmaThMaxType, meta_data, &apr.parameters.sigma_th_max);
            readAttr(AprTypes::IthType, meta_data, &apr.parameters.Ip_th);
            readAttr(AprTypes::DxType, meta_data, &apr.parameters.dx);
            readAttr(AprTypes::DyType, meta_data, &apr.parameters.dy);
            readAttr(AprTypes::DzType, meta_data, &apr.parameters.dz);
            readAttr(AprTypes::PsfXType, meta_data, &apr.parameters.psfx);
            readAttr(AprTypes::PsfYType, meta_data, &apr.parameters.psfy);
            readAttr(AprTypes::PsfZType, meta_data, &apr.parameters.psfz);
            readAttr(AprTypes::RelativeErrorType, meta_data, &apr.parameters.rel_error);
            readAttr(AprTypes::BackgroundIntensityEstimateType, meta_data,
                     &apr.parameters.background_intensity_estimate);
            readAttr(AprTypes::NoiseSdEstimateType, meta_data, &apr.parameters.noise_sd_estimate);

            apr.apr_access.x_num.resize(apr.apr_access.level_max() + 1);
            apr.apr_access.y_num.resize(apr.apr_access.level_max() + 1);
            apr.apr_access.z_num.resize(apr.apr_access.level_max() + 1);

            for (size_t i = apr.apr_access.level_min(); i < apr.apr_access.level_max(); i++) {
                int x_num, y_num, z_num;
                //TODO: x_num and other should have HDF5 type uint64?
                readAttr(AprTypes::NumberOfLevelXType, i, meta_data, &x_num);
                readAttr(AprTypes::NumberOfLevelYType, i, meta_data, &y_num);
                readAttr(AprTypes::NumberOfLevelZType, i,meta_data, &z_num);
                apr.apr_access.x_num[i] = x_num;
                apr.apr_access.y_num[i] = y_num;
                apr.apr_access.z_num[i] = z_num;
            }

            apr.apr_access.y_num[apr.apr_access.level_max()] = apr.apr_access.org_dims[0];
            apr.apr_access.x_num[apr.apr_access.level_max()] = apr.apr_access.org_dims[1];
            apr.apr_access.z_num[apr.apr_access.level_max()] = apr.apr_access.org_dims[2];

            // ------------- map handling ----------------------------

            timer.start_timer("map loading data");

            auto map_data = std::make_shared<MapStorageData>();

            map_data->global_index.resize(apr.apr_access.total_number_non_empty_rows);

            timer_f.start_timer("index");
            std::vector<int16_t> index_delta(apr.apr_access.total_number_non_empty_rows);
            readData(AprTypes::MapGlobalIndexType, f.objectId, index_delta.data());
            std::vector<uint64_t> index_delta_big(apr.apr_access.total_number_non_empty_rows);
            std::copy(index_delta.begin(), index_delta.end(), index_delta_big.begin());
            std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data->global_index.begin());

            timer_f.stop_timer();

            timer_f.start_timer("y_b_e");
            map_data->y_end.resize(apr.apr_access.total_number_gaps);
            readData(AprTypes::MapYendType, f.objectId, map_data->y_end.data());
            map_data->y_begin.resize(apr.apr_access.total_number_gaps);
            readData(AprTypes::MapYbeginType, f.objectId, map_data->y_begin.data());

            timer_f.stop_timer();


            timer_f.start_timer("zxl");
            map_data->number_gaps.resize(apr.apr_access.total_number_non_empty_rows);
            readData(AprTypes::MapNumberGapsType, f.objectId, map_data->number_gaps.data());
            map_data->level.resize(apr.apr_access.total_number_non_empty_rows);
            readData(AprTypes::MapLevelType, f.objectId, map_data->level.data());
            map_data->x.resize(apr.apr_access.total_number_non_empty_rows);
            readData(AprTypes::MapXType, f.objectId, map_data->x.data());
            map_data->z.resize(apr.apr_access.total_number_non_empty_rows);
            readData(AprTypes::MapZType, f.objectId, map_data->z.data());
            timer_f.stop_timer();

            timer_f.start_timer("type");
            //apr.apr_access.particle_cell_type.data.resize(type_size);
            //readData(AprTypes::ParticleCellType, f.objectId, apr.apr_access.particle_cell_type.data.data());
            timer_f.stop_timer();

            timer.stop_timer();

            timer.start_timer("map building");

            apr.apr_access.rebuild_map(*map_data);

            timer.stop_timer();
        }

        uint64_t max_read_level = apr.apr_access.level_max()-max_level_delta;
        uint64_t max_read_level_tree = std::min(apr.apr_access.level_max()-1,max_read_level);
        uint64_t prev_read_level = 0;

        if(build_tree){


            if(read_structure) {


                timer.start_timer("build tree - map");

                apr.apr_tree.tree_access.l_max = apr.level_max() - 1;
                apr.apr_tree.tree_access.l_min = apr.level_min() - 1;

                apr.apr_tree.tree_access.x_num.resize(apr.apr_tree.tree_access.level_max() + 1);
                apr.apr_tree.tree_access.z_num.resize(apr.apr_tree.tree_access.level_max() + 1);
                apr.apr_tree.tree_access.y_num.resize(apr.apr_tree.tree_access.level_max() + 1);

                for (int i = apr.apr_tree.tree_access.level_min(); i <= apr.apr_tree.tree_access.level_max(); ++i) {
                    apr.apr_tree.tree_access.x_num[i] = apr.spatial_index_x_max(i);
                    apr.apr_tree.tree_access.y_num[i] = apr.spatial_index_y_max(i);
                    apr.apr_tree.tree_access.z_num[i] = apr.spatial_index_z_max(i);
                }

                apr.apr_tree.tree_access.x_num[apr.level_min() - 1] = ceil(apr.spatial_index_x_max(apr.level_min()) / 2.0f);
                apr.apr_tree.tree_access.y_num[apr.level_min() - 1] = ceil(apr.spatial_index_y_max(apr.level_min()) / 2.0f);
                apr.apr_tree.tree_access.z_num[apr.level_min() - 1] = ceil(apr.spatial_index_z_max(apr.level_min()) / 2.0f);

                readAttr(AprTypes::TotalNumberOfParticlesType, f.objectIdTree,
                         &apr.apr_tree.tree_access.total_number_particles);
                readAttr(AprTypes::TotalNumberOfGapsType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_gaps);
                readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, f.objectIdTree,
                         &apr.apr_tree.tree_access.total_number_non_empty_rows);

                auto map_data_tree = std::make_shared<MapStorageData>();

                map_data_tree->global_index.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);


                std::vector<int16_t> index_delta(apr.apr_tree.tree_access.total_number_non_empty_rows);
                readData(AprTypes::MapGlobalIndexType, f.objectIdTree, index_delta.data());
                std::vector<uint64_t> index_delta_big(apr.apr_tree.tree_access.total_number_non_empty_rows);
                std::copy(index_delta.begin(), index_delta.end(), index_delta_big.begin());
                std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data_tree->global_index.begin());


                map_data_tree->y_end.resize(apr.apr_tree.tree_access.total_number_gaps);
                readData(AprTypes::MapYendType, f.objectIdTree, map_data_tree->y_end.data());
                map_data_tree->y_begin.resize(apr.apr_tree.tree_access.total_number_gaps);
                readData(AprTypes::MapYbeginType, f.objectIdTree, map_data_tree->y_begin.data());

                map_data_tree->number_gaps.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
                readData(AprTypes::MapNumberGapsType, f.objectIdTree, map_data_tree->number_gaps.data());
                map_data_tree->level.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
                readData(AprTypes::MapLevelType, f.objectIdTree, map_data_tree->level.data());
                map_data_tree->x.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
                readData(AprTypes::MapXType, f.objectIdTree, map_data_tree->x.data());
                map_data_tree->z.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
                readData(AprTypes::MapZType, f.objectIdTree, map_data_tree->z.data());

                apr.apr_tree.tree_access.rebuild_map_tree(*map_data_tree,apr.apr_access);

                //Important needs linking to the APR
                apr.apr_tree.APROwn = &apr;

                timer.stop_timer();
            }

            if(!read_structure) {
                uint64_t current_parts_size = apr.apr_tree.particles_ds_tree.data.size();

                for (int j = apr.level_min(); j <apr.level_max(); ++j) {
                    if((apr.apr_tree.tree_access.global_index_by_level_end[j] + 1)==current_parts_size){
                        prev_read_level = j;
                    }
                }
            }

            timer.start_timer("tree intensities");

            uint64_t parts_start = 0;
            if(prev_read_level > 0){
                parts_start = apr.apr_tree.tree_access.global_index_by_level_end[prev_read_level] + 1;
            }
            uint64_t parts_end = apr.apr_tree.tree_access.global_index_by_level_end[max_read_level_tree] + 1;

            apr.apr_tree.particles_ds_tree.data.resize(parts_end);

            if ( apr.apr_tree.particles_ds_tree.data.size() > 0) {
                readData(AprTypes::ParticleIntensitiesType, f.objectIdTree, apr.apr_tree.particles_ds_tree.data.data() + parts_start,parts_start,parts_end);
            }

            APRCompress<ImageType> apr_compress;
            apr_compress.set_compression_type(1);
            apr_compress.set_quantization_factor(2);
            apr_compress.decompress(apr, apr.apr_tree.particles_ds_tree,parts_start);

            timer.stop_timer();

        }

        uint64_t parts_start = 0;
        uint64_t parts_end = apr.apr_access.global_index_by_level_end[max_read_level] + 1;

        prev_read_level = 0;
        if(!read_structure) {
            uint64_t current_parts_size = apr.particles_intensities.data.size();

            for (int j = apr.level_min(); j <apr.level_max(); ++j) {
                if((apr.apr_access.global_index_by_level_end[j] + 1)==current_parts_size){
                    prev_read_level = j;
                }
            }
        }

        if(prev_read_level > 0){
            parts_start = apr.apr_access.global_index_by_level_end[prev_read_level] + 1;
        }

        //apr.apr_access.level_max = max_read_level;

        timer.start_timer("Read intensities");
        // ------------- read data ------------------------------
        apr.particles_intensities.data.resize(parts_end);
        if (apr.particles_intensities.data.size() > 0) {
            readData(AprTypes::ParticleIntensitiesType, f.objectId, apr.particles_intensities.data.data() + parts_start,parts_start,parts_end);
        }


        timer.stop_timer();

        std::cout << "Data rate intensities: " << (apr.particles_intensities.data.size()*2)/(timer.timings.back()*1000000.0f) << " MB/s" << std::endl;


        timer.start_timer("decompress");
        // ------------ decompress if needed ---------------------
        if (compress_type > 0) {
            APRCompress<ImageType> apr_compress;
            apr_compress.set_compression_type(compress_type);
            apr_compress.set_quantization_factor(quantization_factor);
            apr_compress.decompress(apr, apr.particles_intensities,parts_start);
        }
        timer.stop_timer();
    }

    template<typename ImageType>
    void read_apr_time(APR<ImageType>& apr, const std::string &file_name,bool build_tree = false, int max_level_delta = 0,unsigned int time_point = UINT32_MAX) {

        APRTimer timer;
        timer.verbose_flag = true;

        APRTimer timer_f;
        timer_f.verbose_flag = false;

        AprFile::Operation op;

        if (build_tree) {
            op = AprFile::Operation::READ_WITH_TREE;
        } else {
            op = AprFile::Operation::READ;
        }

        unsigned int t = 0;

        if (time_point != UINT32_MAX) {
            t = time_point;
        }

        AprFile f(file_name, op, t);

        hid_t meta_data = f.groupId;

        if (time_point != UINT32_MAX) {
            meta_data = f.objectId;
        }


        if (!f.isOpened()) return;

        if(time_point == 0){

            read_apr(apr,file_name,false,0,0);
        } else {

            timer.start_timer("read");

            hid_t type = Hdf5Type<ImageType>::type();

            hid_t type_index = H5T_NATIVE_UINT64;

            TimeData<ImageType> timeData;

            uint64_t update_num;
            readAttr({type_index,"update_num"}, meta_data, &update_num);

            uint64_t add_num;
            readAttr({type_index,"add_num"}, meta_data, &add_num);

            uint64_t remove_num;
            readAttr({type_index,"remove_num"}, meta_data, &remove_num);

            uint64_t update_f_num;
            readAttr({type_index,"update_f_num"}, meta_data, &update_f_num);

            uint64_t add_f_num;
            readAttr({type_index,"add_f_num"}, meta_data, &add_f_num);

            uint64_t remove_f_num;
            readAttr({type_index,"remove_f_num"}, meta_data, &remove_f_num);

            //Intensities

            std::vector<ImageType> update_fp;



            std::vector<ImageType> add_fp;
            std::vector<ImageType> remove_fp;


            //Spatial information

            std::vector<uint64_t> update_index;



            MapStorageData map_data_add;

            hid_t type_pc = H5T_NATIVE_UINT16;

            if(add_num > 0) {

                add_fp.resize(add_f_num);
                readData({type, "add_fp"}, meta_data, add_fp.data());

                map_data_add.y_begin.resize(add_num);
                map_data_add.x.resize(add_num);
                map_data_add.z.resize(add_num);
                map_data_add.level.resize(add_num);

                readData({type_pc, "add_y"}, meta_data, map_data_add.y_begin.data());
                readData({H5T_NATIVE_UINT8, "add_l"}, meta_data, map_data_add.level.data());
                readData({type_pc, "add_x"}, meta_data, map_data_add.x.data());
                readData({type_pc, "add_z"}, meta_data, map_data_add.z.data());

            }

            MapStorageData map_data_remove;

            if(remove_num > 0) {

                remove_fp.resize(remove_f_num);
                readData({type, "remove_fp"}, meta_data, remove_fp.data());

                map_data_remove.y_begin.resize(remove_num);
                map_data_remove.x.resize(remove_num);
                map_data_remove.z.resize(remove_num);
                map_data_remove.level.resize(remove_num);

                readData({type_pc, "remove_y"}, meta_data, map_data_remove.y_begin.data());
                readData({H5T_NATIVE_UINT8, "remove_l"}, meta_data, map_data_remove.level.data());
                readData({type_pc, "remove_x"}, meta_data, map_data_remove.x.data());
                readData({type_pc, "remove_z"}, meta_data, map_data_remove.z.data());
            }

            MapStorageData map_data_update;

            if(update_num > 0) {

                update_fp.resize(update_f_num);
                readData({type, "update_fp"}, meta_data, update_fp.data());

                map_data_update.y_begin.resize(update_num);
                map_data_update.x.resize(update_num);
                map_data_update.z.resize(update_num);
                map_data_update.level.resize(update_num);

                readData({type_pc, "update_y"}, meta_data, map_data_update.y_begin.data());
                readData({H5T_NATIVE_UINT8, "update_l"}, meta_data, map_data_update.level.data());
                readData({type_pc, "update_x"}, meta_data, map_data_update.x.data());
                readData({type_pc, "update_z"}, meta_data, map_data_update.z.data());
            }


            timer.stop_timer();

            timer.start_timer("insert");

            if((remove_num + add_num) > 0){

                auto apr_iterator = apr.iterator();
                std::vector<PixelData<uint8_t>> layers;

                layers.resize(apr_iterator.level_max());

                for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
                    int z = 0;
                    int x = 0;

                    if(level < apr.level_max()) {
                        layers[level].initWithValue(apr_iterator.spatial_index_y_max(level),
                                                    apr_iterator.spatial_index_x_max(level),
                                                    apr_iterator.spatial_index_z_max(level), 0);
                    }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
                    for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                        for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                            for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                                 apr_iterator.set_iterator_to_particle_next_particle()) {

                                if(level==apr_iterator.level_max()){
                                    layers[level-1].at(apr_iterator.y()/2,apr_iterator.x()/2,apr_iterator.z()/2)=1;
                                } else {
                                    layers[level].at(apr_iterator.y(),apr_iterator.x(),apr_iterator.z())=3;
                                }


                            }
                        }
                    }

                }

                //remove (this loop shoudl go first, as the shared l_max-1, otherwise will not work correclty)
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i < remove_num; ++i) {

                    uint8_t l = map_data_remove.level[i];
                    uint16_t y = map_data_remove.y_begin[i];
                    uint16_t x = map_data_remove.x[i];
                    uint16_t z = map_data_remove.z[i];

                    if (l < apr.level_max()) {
                        layers[l].at(y, x, z) = 0;
                    } else {
                        layers[l - 1].at(y, x, z) = 0;
                    }

                }

                //add
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i < add_num; ++i) {

                    uint8_t l = map_data_add.level[i];
                    uint16_t y = map_data_add.y_begin[i];
                    uint16_t x = map_data_add.x[i];
                    uint16_t z = map_data_add.z[i];

                    if(l < apr.level_max()){
                        layers[l].at(y,x,z) = 3;
                    } else {
                        layers[l-1].at(y,x,z) = 1;
                    }

                }


                APR<ImageType> apr_img_temp;
                apr_img_temp.apr_access.org_dims[0] = apr.apr_access.org_dims[0];
                apr_img_temp.apr_access.org_dims[1] = apr.apr_access.org_dims[1];
                apr_img_temp.apr_access.org_dims[2] = apr.apr_access.org_dims[2];
                apr_img_temp.apr_access.l_max = apr.apr_access.l_max;
                apr_img_temp.apr_access.l_min = apr.apr_access.l_min;
                apr_img_temp.parameters = apr.parameters;

                timer.stop_timer();

                timer.start_timer("old_update");

                apr_img_temp.apr_access.initialize_structure_from_particle_cell_tree(apr.parameters,layers);

                apr_img_temp.particles_intensities.data.resize(apr.particles_intensities.data.size() + add_f_num - remove_f_num);

                apr.copy_from_APR(apr_img_temp);
                timer.stop_timer();


            }


        }

    }

    template<typename ImageType>
    FileSizeInfo write_apr(APR<ImageType>& apr, const std::string &save_loc, const std::string &file_name) {
        APRCompress<ImageType> apr_compressor;
        apr_compressor.set_compression_type(0);
        return write_apr(apr, save_loc, file_name, apr_compressor);
    }

    template<typename ImageType>
    FileSizeInfo write_apr_append(APR<ImageType>& apr, const std::string &save_loc, const std::string &file_name) {
        APRCompress<ImageType> apr_compressor;
        apr_compressor.set_compression_type(0);
        return write_apr(apr, save_loc, file_name, apr_compressor,BLOSC_ZSTD,4,1,false,true);
    }

    template<typename ImageType>
    struct TimeData{
        ExtraParticleData<ImageType>* add_fp;
        ExtraParticleData<ImageType>* update_fp;
        ExtraParticleData<ImageType>* remove_fp;

        ExtraParticleData<uint64_t>* update_index;
        ExtraParticleData<uint64_t>* add_index;
        ExtraParticleData<uint64_t>* remove_index;

        ExtraParticleData<ParticleCell>* update_pc;
        ExtraParticleData<YGap_map>* update_gap;

        ExtraParticleData<ParticleCell>* remove_pc;
        ExtraParticleData<ParticleCell>* add_pc;

        unsigned int l_max;


    };


    template<typename ImageType>
    void write_particle_cells(TimeData<ImageType>& tdata,hid_t location,unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 4, unsigned int blosc_shuffle=1){

        MapStorageData map_data;

        for (int i = 0; i < tdata.remove_pc->data.size(); ++i) {
            if(tdata.remove_pc->data[i].level<tdata.l_max) {
                map_data.y_begin.push_back(tdata.remove_pc->data[i].y);
                map_data.x.push_back(tdata.remove_pc->data[i].x);
                map_data.z.push_back(tdata.remove_pc->data[i].z);

                map_data.level.push_back(tdata.remove_pc->data[i].level);
            } else {
                map_data.y_begin.push_back(tdata.remove_pc->data[i].y/2);
                map_data.x.push_back(tdata.remove_pc->data[i].x/2);
                map_data.z.push_back(tdata.remove_pc->data[i].z/2);
                map_data.level.push_back(tdata.remove_pc->data[i].level);
            }
        }

        hid_t type_index = H5T_NATIVE_UINT16;

        if(tdata.remove_fp->data.size() > 0) {
            writeData({type_index, "remove_y"}, location, map_data.y_begin, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({H5T_NATIVE_UINT8, "remove_l"}, location, map_data.level, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({type_index, "remove_x"}, location, map_data.x, blosc_comp_type, blosc_comp_level, blosc_shuffle);
            writeData({type_index, "remove_z"}, location, map_data.z, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        }
        MapStorageData map_data_add;

        for (int i = 0; i < tdata.add_pc->data.size(); ++i) {
            if(tdata.add_pc->data[i].level<tdata.l_max) {
                map_data_add.y_begin.push_back(tdata.add_pc->data[i].y);
                map_data_add.x.push_back(tdata.add_pc->data[i].x);
                map_data_add.z.push_back(tdata.add_pc->data[i].z);
                map_data_add.level.push_back(tdata.add_pc->data[i].level);
            } else {
                map_data_add.y_begin.push_back(tdata.add_pc->data[i].y/2);
                map_data_add.x.push_back(tdata.add_pc->data[i].x/2);
                map_data_add.z.push_back(tdata.add_pc->data[i].z/2);
                map_data_add.level.push_back(tdata.add_pc->data[i].level);
            }
        }
        if(tdata.add_fp->data.size() > 0) {
            writeData({type_index, "add_y"}, location, map_data_add.y_begin, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({H5T_NATIVE_UINT8, "add_l"}, location, map_data_add.level, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({type_index, "add_x"}, location, map_data_add.x, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({type_index, "add_z"}, location, map_data_add.z, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
        }

        MapStorageData map_data_update;

        for (int i = 0; i < tdata.update_pc->data.size(); ++i) {
            if(tdata.update_pc->data[i].level<tdata.l_max) {
                map_data_update.y_begin.push_back(tdata.update_pc->data[i].y);
                map_data_update.x.push_back(tdata.update_pc->data[i].x);
                map_data_update.z.push_back(tdata.update_pc->data[i].z);
                map_data_update.level.push_back(tdata.update_pc->data[i].level);
            } else {
                map_data_update.y_begin.push_back(tdata.update_pc->data[i].y/2);
                map_data_update.x.push_back(tdata.update_pc->data[i].x/2);
                map_data_update.z.push_back(tdata.update_pc->data[i].z/2);
                map_data_update.level.push_back(tdata.update_pc->data[i].level);
            }
        }
        if(tdata.update_pc->data.size() > 0) {
            writeData({type_index, "update_y"}, location, map_data_update.y_begin, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({H5T_NATIVE_UINT8, "update_l"}, location, map_data_update.level, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({type_index, "update_x"}, location, map_data_update.x, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
            writeData({type_index, "update_z"}, location, map_data_update.z, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);
        }

    }

    template<typename ImageType>
    FileSizeInfoTime write_apr_time(TimeData<ImageType>& timeData, const std::string &save_loc, const std::string &file_name, unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 4, unsigned int blosc_shuffle=1,bool write_tree = false,bool append_apr_time = true) {
        APRTimer write_timer;
        write_timer.verbose_flag = true;

        std::string hdf5_file_name = save_loc + file_name + "_apr.h5";


        AprFile::Operation op;

        if (write_tree) {
            op = AprFile::Operation::WRITE_WITH_TREE;
        } else {
            op = AprFile::Operation::WRITE;
        }

        unsigned int t = 0;

        if (append_apr_time) {
            t = current_t;
            current_t++;
        }

        AprFile f(hdf5_file_name, op, t);

        FileSizeInfo fileSizeInfo1;
        FileSizeInfoTime fzt;
        if (!f.isOpened()) return fzt;

        hid_t meta_location = f.groupId;

        if (append_apr_time) {
            meta_location = f.objectId;
        }

        hid_t type = Hdf5Type<ImageType>::type();

        hid_t type_index = H5T_NATIVE_UINT64;



        auto o_size = f.getFileSize();

        if(timeData.update_fp->data.size() > 0) {
            writeData({type, "update_fp"}, meta_location, timeData.update_fp->data, blosc_comp_type, blosc_comp_level,
                      blosc_shuffle);

            fzt.update_fp = f.getFileSize() - o_size;
            o_size = f.getFileSize();

//            writeData({type_index, "update_index"}, meta_location, timeData.update_index->data, blosc_comp_type, blosc_comp_level,
//                      blosc_shuffle);

            fzt.update_index = f.getFileSize() - o_size;
            o_size = f.getFileSize();


            //write_particle_cells(timeData,meta_location);
        }

        uint64_t update_num = timeData.update_fp->data.size();
        writeAttr({type_index,"update_f_num"}, meta_location, &update_num);

        uint64_t update_num_index = timeData.update_index->data.size();
        writeAttr({type_index,"update_num"}, meta_location, &update_num_index);

        if(timeData.add_fp->data.size() > 0) {

            writeData({type, "add_fp"}, meta_location, timeData.add_fp->data, blosc_comp_type, blosc_comp_level, blosc_shuffle);

            fzt.add_fp = f.getFileSize() - o_size;
            o_size = f.getFileSize();


            //writeData({type_index, "add_index"}, meta_location, timeData.add_index->data, blosc_comp_type, blosc_comp_level, blosc_shuffle);


            fzt.add_index = f.getFileSize() - o_size;
            o_size = f.getFileSize();



        }

        uint64_t add_num = timeData.add_fp->data.size();
        writeAttr({type_index,"add_f_num"}, meta_location, &add_num);

        add_num = timeData.add_index->data.size();
        writeAttr({type_index,"add_num"}, meta_location, &add_num);

        if(timeData.remove_index->data.size() > 0) {

            //writeData({type_index, "remove_index"}, meta_location, timeData.remove_index->data, blosc_comp_type, blosc_comp_level, blosc_shuffle);


            writeData({type, "remove_fp"}, meta_location, timeData.remove_fp->data, blosc_comp_type, blosc_comp_level, blosc_shuffle);


            fzt.remove_fp = f.getFileSize() - o_size;
            o_size = f.getFileSize();



        }

        uint64_t remove_num = timeData.remove_fp->data.size();
        writeAttr({type_index,"remove_f_num"}, meta_location, &remove_num);

        remove_num = timeData.remove_index->data.size();
        writeAttr({type_index,"remove_num"}, meta_location, &remove_num);


        write_particle_cells(timeData,meta_location);

        fzt.remove_index = f.getFileSize() - o_size;
        o_size = f.getFileSize();

        // ------------- output the file size -------------------
        auto file_size = f.getFileSize();
        double sizeMB = file_size / 1e6;


        std::cout << "HDF5 Total Filesize: " << sizeMB << " MB\n" << "Writing Complete" << std::endl;

        return fzt;
    }



    /**
     * Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
     */
    template<typename ImageType>
    FileSizeInfo write_apr(APR<ImageType> &apr, const std::string &save_loc, const std::string &file_name, APRCompress<ImageType> &apr_compressor, unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 2, unsigned int blosc_shuffle=1,bool write_tree = false,bool append_apr_time = false) {
        APRTimer write_timer;
        write_timer.verbose_flag = true;

        std::string hdf5_file_name = save_loc + file_name + "_apr.h5";


        AprFile::Operation op;

        if(write_tree){
            op = AprFile::Operation::WRITE_WITH_TREE;
        } else {
            op = AprFile::Operation::WRITE;
        }


        unsigned int t = 0;

        if(append_apr_time){
            t = current_t;
            current_t++;
        }

        AprFile f(hdf5_file_name, op,t);

        FileSizeInfo fileSizeInfo1;
        if (!f.isOpened()) return fileSizeInfo1;

        hid_t meta_location = f.groupId;

        if(append_apr_time){
            meta_location = f.objectId;
        }

        // ------------- write metadata -------------------------
        writeAttr(AprTypes::NumberOfXType, meta_location, &apr.apr_access.org_dims[1]);
        writeAttr(AprTypes::NumberOfYType, meta_location, &apr.apr_access.org_dims[0]);
        writeAttr(AprTypes::NumberOfZType, meta_location, &apr.apr_access.org_dims[2]);
        writeAttr(AprTypes::TotalNumberOfGapsType, meta_location, &apr.apr_access.total_number_gaps);
        writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, meta_location, &apr.apr_access.total_number_non_empty_rows);
        uint64_t type_vector_size = apr.apr_access.particle_cell_type.data.size();
        writeAttr(AprTypes::VectorSizeType, meta_location, &type_vector_size);

        writeString(AprTypes::NameType,meta_location, (apr.name.size() == 0) ? "no_name" : apr.name);
        writeString(AprTypes::GitType, meta_location, ConfigAPR::APR_GIT_HASH);
        writeAttr(AprTypes::TotalNumberOfParticlesType, meta_location, &apr.apr_access.total_number_particles);
        writeAttr(AprTypes::MaxLevelType, meta_location, &apr.apr_access.l_max);
        writeAttr(AprTypes::MinLevelType, meta_location, &apr.apr_access.l_min);

        int compress_type_num = apr_compressor.get_compression_type();
        writeAttr(AprTypes::CompressionType, meta_location, &compress_type_num);
        float quantization_factor = apr_compressor.get_quantization_factor();
        writeAttr(AprTypes::QuantizationFactorType, meta_location, &quantization_factor);
        writeAttr(AprTypes::LambdaType, meta_location, &apr.parameters.lambda);
        writeAttr(AprTypes::SigmaThType, meta_location, &apr.parameters.sigma_th);
        writeAttr(AprTypes::SigmaThMaxType, meta_location, &apr.parameters.sigma_th_max);
        writeAttr(AprTypes::IthType, meta_location, &apr.parameters.Ip_th);
        writeAttr(AprTypes::DxType, meta_location, &apr.parameters.dx);
        writeAttr(AprTypes::DyType, meta_location, &apr.parameters.dy);
        writeAttr(AprTypes::DzType, meta_location, &apr.parameters.dz);
        writeAttr(AprTypes::PsfXType, meta_location, &apr.parameters.psfx);
        writeAttr(AprTypes::PsfYType, meta_location, &apr.parameters.psfy);
        writeAttr(AprTypes::PsfZType, meta_location, &apr.parameters.psfz);
        writeAttr(AprTypes::RelativeErrorType, meta_location, &apr.parameters.rel_error);
        writeAttr(AprTypes::NoiseSdEstimateType, meta_location, &apr.parameters.noise_sd_estimate);
        writeAttr(AprTypes::BackgroundIntensityEstimateType, meta_location,
                  &apr.parameters.background_intensity_estimate);


        write_timer.start_timer("access_data");
        MapStorageData map_data;
        apr.apr_access.flatten_structure( map_data);

        std::vector<uint16_t> index_delta;
        index_delta.resize(map_data.global_index.size());
        std::adjacent_difference(map_data.global_index.begin(),map_data.global_index.end(),index_delta.begin());
        writeData(AprTypes::MapGlobalIndexType, f.objectId, index_delta, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        writeData(AprTypes::MapYendType, f.objectId, map_data.y_end, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapYbeginType, f.objectId, map_data.y_begin, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapNumberGapsType, f.objectId, map_data.number_gaps, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapLevelType, f.objectId, map_data.level, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapXType, f.objectId, map_data.x, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapZType, f.objectId, map_data.z, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        //optional storage of type
        if(apr.apr_access.particle_cell_type.data.size()> 0) {
            writeData(AprTypes::ParticleCellType, f.objectId, apr.apr_access.particle_cell_type.data, blosc_comp_type,
                      blosc_comp_level, blosc_shuffle);
        }

        bool store_tree = write_tree;

        if(store_tree){

            if(apr.apr_tree.particles_ds_tree.data.size()==0) {
                apr.apr_tree.init(apr);
                apr.apr_tree.fill_tree_mean_downsample(apr.particles_intensities);
            }

            writeAttr(AprTypes::TotalNumberOfGapsType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_gaps);
            writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_non_empty_rows);
            writeAttr(AprTypes::TotalNumberOfParticlesType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_particles);

            MapStorageData map_data_tree;
            apr.apr_tree.tree_access.flatten_structure( map_data_tree);

            std::vector<uint16_t> index_delta;
            index_delta.resize(map_data_tree.global_index.size());
            std::adjacent_difference(map_data_tree.global_index.begin(),map_data_tree.global_index.end(),index_delta.begin());
            writeData(AprTypes::MapGlobalIndexType, f.objectIdTree, index_delta, blosc_comp_type, blosc_comp_level, blosc_shuffle);

            writeData(AprTypes::MapYendType, f.objectIdTree, map_data_tree.y_end, blosc_comp_type, blosc_comp_level, blosc_shuffle);
            writeData(AprTypes::MapYbeginType, f.objectIdTree, map_data_tree.y_begin, blosc_comp_type, blosc_comp_level, blosc_shuffle);
            writeData(AprTypes::MapNumberGapsType, f.objectIdTree, map_data_tree.number_gaps, blosc_comp_type, blosc_comp_level, blosc_shuffle);
            writeData(AprTypes::MapLevelType, f.objectIdTree, map_data_tree.level, blosc_comp_type, blosc_comp_level, blosc_shuffle);
            writeData(AprTypes::MapXType, f.objectIdTree, map_data_tree.x, blosc_comp_type, blosc_comp_level, blosc_shuffle);
            writeData(AprTypes::MapZType, f.objectIdTree, map_data_tree.z, blosc_comp_type, blosc_comp_level, blosc_shuffle);

            APRCompress<ImageType> tree_compress;
            tree_compress.set_compression_type(1);
            tree_compress.set_quantization_factor(2);

            tree_compress.compress(apr, apr.apr_tree.particles_ds_tree);

            unsigned int tree_blosc_comp_type = 6;

            hid_t type = Hdf5Type<ImageType>::type();
            writeData({type, AprTypes::ParticleIntensitiesType}, f.objectIdTree, apr.apr_tree.particles_ds_tree.data, tree_blosc_comp_type, blosc_comp_level, blosc_shuffle);

        }

        write_timer.stop_timer();

        for (size_t i = apr.level_min(); i <apr.level_max() ; ++i) {
            int x_num = apr.apr_access.x_num[i];
            writeAttr(AprTypes::NumberOfLevelXType, i, meta_location, &x_num);
            int y_num = apr.apr_access.y_num[i];
            writeAttr(AprTypes::NumberOfLevelYType, i, meta_location, &y_num);
            int z_num = apr.apr_access.z_num[i];
            writeAttr(AprTypes::NumberOfLevelZType, i, meta_location, &z_num);
        }


        // ------------- output the file size -------------------
        hsize_t file_size = f.getFileSize();
        double sizeMB_access = file_size / 1e6;

        FileSizeInfo fileSizeInfo;
        fileSizeInfo.access_data = sizeMB_access;

        // ------------- write data ----------------------------
        write_timer.start_timer("intensities");
        if (compress_type_num > 0){
            apr_compressor.compress(apr,apr.particles_intensities);
        }
        hid_t type = Hdf5Type<ImageType>::type();
        writeData({type, AprTypes::ParticleIntensitiesType}, meta_location, apr.particles_intensities.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        write_timer.stop_timer();

        // ------------- output the file size -------------------
        file_size = f.getFileSize();
        double sizeMB = file_size / 1e6;

        fileSizeInfo.total_file_size = sizeMB;
        fileSizeInfo.intensity_data = fileSizeInfo.total_file_size - fileSizeInfo.access_data;

        std::cout << "HDF5 Total Filesize: " << sizeMB << " MB\n" << "Writing Complete" << std::endl;
        return fileSizeInfo;
    }




    template<typename ImageType,typename T>
    void write_apr_paraview(APR<ImageType> &apr, const std::string &save_loc, const std::string &file_name, const ExtraParticleData<T> &parts,std::vector<uint64_t> previous_num = {0}) {
        std::string hdf5_file_name = save_loc + file_name + "_paraview.h5";

        bool write_time = false;
        unsigned int t = 0;
        if(previous_num[0]!=0){
            //time series write
            write_time = true;
            t = (previous_num.size()-1);
        }


        AprFile f(hdf5_file_name, AprFile::Operation::WRITE,t);
        if (!f.isOpened()) return;

        // ------------- write metadata -------------------------

        writeString(AprTypes::NameType, f.objectId, (apr.name.size() == 0) ? "no_name" : apr.name);
        writeString(AprTypes::GitType, f.objectId, ConfigAPR::APR_GIT_HASH);
        writeAttr(AprTypes::MaxLevelType, f.objectId, &apr.apr_access.l_max);
        writeAttr(AprTypes::MinLevelType, f.objectId, &apr.apr_access.l_min);
        writeAttr(AprTypes::TotalNumberOfParticlesType, f.objectId, &apr.apr_access.total_number_particles);

        // ------------- write data ----------------------------
        writeDataStandard({(Hdf5Type<T>::type()), AprTypes::ParticlePropertyType}, f.objectId, parts.data);

        auto apr_iterator = apr.iterator();
        std::vector<uint16_t> xv(apr_iterator.total_number_particles());
        std::vector<uint16_t> yv(apr_iterator.total_number_particles());
        std::vector<uint16_t> zv(apr_iterator.total_number_particles());
        std::vector<uint8_t> levelv(apr_iterator.total_number_particles());
        //std::vector<uint8_t> typev(apr_iterator.total_number_particles());

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {
                        xv[apr_iterator.global_index()] = apr_iterator.x_global();
                        yv[apr_iterator.global_index()] = apr_iterator.y_global();
                        zv[apr_iterator.global_index()] = apr_iterator.z_global();
                        levelv[apr_iterator.global_index()] = apr_iterator.level();
                    }
                }
            }
        }

        writeDataStandard(AprTypes::ParaviewXType, f.objectId, xv);
        writeDataStandard(AprTypes::ParaviewYType, f.objectId, yv);
        writeDataStandard(AprTypes::ParaviewZType, f.objectId, zv);
        writeDataStandard(AprTypes::ParaviewLevelType, f.objectId, levelv);
        //writeDataStandard(AprTypes::ParaviewTypeType, f.objectId, typev);

        // TODO: This needs to be able extended to handle more general type, currently it is assuming uint16
        if(write_time){
            write_main_paraview_xdmf_xml_time(save_loc, hdf5_file_name, file_name,previous_num);
        } else {
            write_main_paraview_xdmf_xml(save_loc, hdf5_file_name, file_name, apr_iterator.total_number_particles());
        }

        // ------------- output the file size -------------------
        hsize_t file_size;
        H5Fget_filesize(f.fileId, &file_size);
        std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;
        std::cout << "Writing Complete" << std::endl;
    }






    /**
     * Writes only the particle data, requires the same APR to be read in correctly.
     */
    template<typename S>
    float write_particles_only(const std::string &save_loc, const std::string &file_name, const ExtraParticleData<S> &parts_extra) {
        std::string hdf5_file_name = save_loc + file_name + "_apr_extra_parts.h5";

        AprFile f{hdf5_file_name, AprFile::Operation::WRITE};
        if (!f.isOpened()) return 0;

        // ------------- write metadata -------------------------
        uint64_t total_number_parts = parts_extra.data.size();
        writeAttr(AprTypes::TotalNumberOfParticlesType, f.groupId, &total_number_parts);
        writeString(AprTypes::GitType, f.groupId, ConfigAPR::APR_GIT_HASH);

        // ------------- write data ----------------------------
        unsigned int blosc_comp_type = 6;
        unsigned int blosc_comp_level = 9;
        unsigned int blosc_shuffle = 1;
        hid_t type = Hdf5Type<S>::type();
        writeData({type, AprTypes::ExtraParticleDataType}, f.objectId, parts_extra.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        // ------------- output the file size -------------------
        hsize_t file_size = f.getFileSize();
        std::cout << "HDF5 Filesize: " << file_size/1e6 << " MB" << std::endl;
        std::cout << "Writing ExtraPartCellData Complete" << std::endl;

        return file_size/1e6; //returns file size in MB
    }

    template<typename T>
    void read_parts_only(const std::string &aFileName, ExtraParticleData<T>& extra_parts) {
        AprFile f{aFileName, AprFile::Operation::READ};
        if (!f.isOpened()) return;

        // ------------- read metadata --------------------------
        uint64_t numberOfParticles;
        readAttr(AprTypes::TotalNumberOfParticlesType, f.groupId, &numberOfParticles);

        // ------------- read data -----------------------------
        extra_parts.data.resize(numberOfParticles);
        readData(AprTypes::ExtraParticleDataType, f.objectId, extra_parts.data.data());
    }

    template<typename ImageType>
    float write_mesh_to_hdf5(PixelData<ImageType>& input_mesh,const std::string &save_loc, const std::string &file_name,unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 2, unsigned int blosc_shuffle=1){
        std::string hdf5_file_name = save_loc + file_name + "_pixels.h5";

        AprFile f{hdf5_file_name, AprFile::Operation::WRITE};
        if (!f.isOpened()) return 0;

        // ------------- write metadata -------------------------
        writeAttr(AprTypes::NumberOfXType, f.groupId, &input_mesh.x_num);
        writeAttr(AprTypes::NumberOfYType, f.groupId, &input_mesh.y_num);
        writeAttr(AprTypes::NumberOfZType, f.groupId, &input_mesh.z_num);

        hid_t type = Hdf5Type<ImageType>::type();

        AprType aType = {type, AprTypes::ParticleIntensitiesType};

        hsize_t dims[] = {input_mesh.mesh.size()};

        const hsize_t rank = 1;
        hdf5_write_data_blosc(f.objectId,aType.hdf5type, aType.typeName, rank, dims, input_mesh.mesh.begin(), blosc_comp_type, blosc_comp_level, blosc_shuffle);

        // ------------- output the file size -------------------
        hsize_t file_size = f.getFileSize();
        std::cout << "HDF5 Filesize: " << file_size/1e6 << " MB" << std::endl;
        std::cout << "Writing ExtraPartCellData Complete" << std::endl;

        return file_size/1e6; //returns file size in MB

    }


    struct AprFile {
        enum class Operation {READ, WRITE,READ_WITH_TREE,WRITE_WITH_TREE,WRITE_APPEND};
        hid_t fileId = -1;
        hid_t groupId = -1;
        hid_t objectId = -1;
        hid_t objectIdTree = -1;


        AprFile(const std::string &aFileName, const Operation aOp,const unsigned int t = 0) {

            std::string t_string;

            if(t==0){
                t_string = "t";
            } else{
                t_string = "t" + std::to_string(t);
            }

            std::string subGroup1 = ("ParticleRepr/" + t_string);
            std::string subGroupTree1 = "ParticleRepr/" + t_string + "/Tree";

            const char * const mainGroup =  ("ParticleRepr");
            const char * const subGroup  = subGroup1.c_str();
            const char * const subGroupTree  = subGroupTree1.c_str();

            hdf5_register_blosc();
            switch(aOp) {
                case Operation::READ:
                    fileId = H5Fopen(aFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                    if (fileId == -1) {
                        std::cerr << "Could not open file [" << aFileName << "]" << std::endl;
                        return;
                    }
                    groupId = H5Gopen2(fileId, mainGroup, H5P_DEFAULT);
                    objectId = H5Gopen2(fileId, subGroup, H5P_DEFAULT);

                    break;
                case Operation::READ_WITH_TREE:
                    fileId = H5Fopen(aFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                    if (fileId == -1) {
                        std::cerr << "Could not open file [" << aFileName << "]" << std::endl;
                        return;
                    }
                    groupId = H5Gopen2(fileId, mainGroup, H5P_DEFAULT);
                    objectId = H5Gopen2(fileId, subGroup, H5P_DEFAULT);
                    objectIdTree = H5Gopen2(fileId, subGroupTree, H5P_DEFAULT);
                    break;
                case Operation::WRITE:

                    if(t==0) {
                        fileId = hdf5_create_file_blosc(aFileName);
                        if (fileId == -1) {
                            std::cerr << "Could not create file [" << aFileName << "]" << std::endl;
                            return;
                        }
                        groupId = H5Gcreate2(fileId, mainGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                        objectId = H5Gcreate2(fileId, subGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    } else{
                        fileId = H5Fopen(aFileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                        if (fileId == -1) {
                            std::cerr << "Could not open file [" << aFileName << "]" << std::endl;
                            return;
                        }
                        groupId = H5Gopen2(fileId, mainGroup, H5P_DEFAULT);
                        objectId = H5Gcreate2(fileId, subGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                    }
                    break;
                case Operation::WRITE_APPEND:

                    fileId = H5Fopen(aFileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                    if (fileId == -1) {
                        std::cerr << "Could not open file [" << aFileName << "]" << std::endl;
                        return;
                    }
                    groupId = H5Gopen2(fileId, mainGroup, H5P_DEFAULT);
                    objectId = H5Gopen2(fileId, subGroup, H5P_DEFAULT);

                    break;
                case Operation::WRITE_WITH_TREE:
                    fileId = hdf5_create_file_blosc(aFileName);
                    if (fileId == -1) {
                        std::cerr << "Could not create file [" << aFileName << "]" << std::endl;
                        return;
                    }
                    groupId = H5Gcreate2(fileId, mainGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    objectId = H5Gcreate2(fileId, subGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    objectIdTree = H5Gcreate2(fileId, subGroupTree, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    break;
            }
            if (groupId == -1 || objectId == -1) { H5Fclose(fileId); fileId = -1; }
        }
        ~AprFile() {
            if (objectIdTree != -1) H5Gclose(objectIdTree);
            if (objectId != -1) H5Gclose(objectId);
            if (groupId != -1) H5Gclose(groupId);
            if (fileId != -1) H5Fclose(fileId);
        }

        /**
         * Is File opened?
         */
        bool isOpened() const { return fileId != -1 && groupId != -1 ; }

        hsize_t getFileSize() const {
            hsize_t size;
            H5Fget_filesize(fileId, &size);
            return size;
        }
    };

protected:
    void readAttr(const AprType &aType, hid_t aGroupId, void *aDest) {
        hid_t attr_id = H5Aopen(aGroupId, aType.typeName, H5P_DEFAULT);
        H5Aread(attr_id, aType.hdf5type, aDest);
        H5Aclose(attr_id);
    }

    void readAttr(const AprType &aType, size_t aSuffix, hid_t aGroupId, void *aDest) {
        std::string typeNameWithPrefix = std::string(aType.typeName) + std::to_string(aSuffix);
        hid_t attr_id = H5Aopen(aGroupId, typeNameWithPrefix.c_str(), H5P_DEFAULT);
        H5Aread(attr_id, aType.hdf5type, aDest);
        H5Aclose(attr_id);
    }

    void writeAttr(const AprType &aType, hid_t aGroupId, const void * const aSrc) {
        hsize_t dims[] = {1};
        hdf5_write_attribute_blosc(aGroupId, aType.hdf5type, aType.typeName, 1, dims, aSrc);
    }

    void writeAttr(const AprType &aType, size_t aSuffix, hid_t aGroupId, const void * const aSrc) {
        std::string typeNameWithPrefix = std::string(aType.typeName) + std::to_string(aSuffix);
        writeAttr({aType.hdf5type, typeNameWithPrefix.c_str()}, aGroupId, aSrc);
    }

    void readData(const AprType &aType, hid_t aObjectId, void *aDest) {
        hdf5_load_data_blosc(aObjectId, aType.hdf5type, aDest, aType.typeName);
    }

    void readData(const char * const aAprTypeName, hid_t aObjectId, void *aDest) {
        hdf5_load_data_blosc(aObjectId, aDest, aAprTypeName);
    }

    void readData(const char * const aAprTypeName, hid_t aObjectId, void *aDest,uint64_t elements_start,uint64_t elements_end) {
        //reads partial dataset
        hdf5_load_data_blosc_partial(aObjectId, aDest, aAprTypeName,elements_start,elements_end);
    }

    //hdf5_load_data_blosc_partial(hid_t obj_id, void* buff, const char* data_name,uint64_t number_of_elements_read,uint64_t number_of_elements_total)


    template<typename T>
    void writeData(const AprType &aType, hid_t aObjectId, T aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_blosc(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);
    }

    template<typename T>
    void writeDataAppend(const AprType &aType, hid_t aObjectId, T aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {


        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;

        int out = H5Lexists( aObjectId, aType.typeName, H5P_DEFAULT );

        if(out==0){
            std::cout << "dne" << std::endl;
            hdf5_write_data_blosc_create(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);
        } else {
            std::cout << "exists" << std::endl;
            hdf5_write_data_blosc_append(aObjectId, aType.hdf5type, aType.typeName, aContainer.data(),dims);


        }


        //hdf5_write_data_blosc_create(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);


        //hdf5_write_data_blosc_append(aObjectId, aType.hdf5type, aType.typeName, void *data,size_t elements_start,size_t elements_end);

        //hdf5_write_data_blosc(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);
    }


    template<typename T>
    void writeDataStandard(const AprType &aType, hid_t aObjectId, T aContainer) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_standard(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data());
    }

    void writeString(AprType aTypeName, hid_t aGroupId, const std::string &aValue) {
        if (aValue.size() > 0){
            hid_t aid = H5Screate(H5S_SCALAR);
            hid_t atype = H5Tcopy (aTypeName.hdf5type);
            H5Tset_size(atype, aValue.size());
            hid_t attr = H5Acreate2(aGroupId, aTypeName.typeName, atype, aid, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attr, atype, aValue.c_str());
            H5Aclose(attr);
            H5Tclose(atype);
            H5Sclose(aid);
        }
    }

    template<typename T> struct Hdf5Type {static hid_t type() {return  T::CANNOT_DETECT_TYPE_AND_WILL_NOT_COMPILE;}};
};


template<> struct APRWriter::Hdf5Type<int8_t> {static hid_t type() {return H5T_NATIVE_INT8;}};
template<> struct APRWriter::Hdf5Type<uint8_t> {static hid_t type() {return H5T_NATIVE_UINT8;}};
template<> struct APRWriter::Hdf5Type<int16_t> {static hid_t type() {return H5T_NATIVE_INT16;}};
template<> struct APRWriter::Hdf5Type<uint16_t> {static hid_t type() {return H5T_NATIVE_UINT16;}};
template<> struct APRWriter::Hdf5Type<int> {static hid_t type() {return H5T_NATIVE_INT;}};
template<> struct APRWriter::Hdf5Type<unsigned int> {static hid_t type() {return H5T_NATIVE_UINT;}};
template<> struct APRWriter::Hdf5Type<int64_t> {static hid_t type() {return H5T_NATIVE_INT64;}};
template<> struct APRWriter::Hdf5Type<uint64_t> {static hid_t type() {return H5T_NATIVE_UINT64;}};
template<> struct APRWriter::Hdf5Type<float> {static hid_t type() {return H5T_NATIVE_FLOAT;}};
template<> struct APRWriter::Hdf5Type<double> {static hid_t type() {return H5T_NATIVE_DOUBLE;}};


#endif //APRWRITER_HPP
