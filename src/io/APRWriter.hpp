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
#include "numerics/APRCompress.hpp"


struct FileSizeInfo {
    float total_file_size=0;
    float intensity_data=0;
    float access_data=0;
};

struct FileSizeInfoTime {
//    float total_file_size=0;
//    float update_fp = 0;
//    float update_index = 0;
//    float add_index = 0;
//    float add_fp = 0;
//    float remove_index = 0;
//    float remove_fp = 0;
};


struct AprType {hid_t hdf5type; const char * const typeName;};
namespace AprTypes  {

    const AprType TotalNumberOfParticlesType = {H5T_NATIVE_UINT64, "total_number_particles"};
    const AprType TotalNumberOfGapsType = {H5T_NATIVE_UINT64, "total_number_gaps"};
    const AprType TotalNumberOfNonEmptyRowsType = {H5T_NATIVE_UINT64, "total_number_non_empty_rows"};
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
    const AprType NameType = {H5T_C_S1, "name"};
    const AprType GitType = {H5T_C_S1, "githash"};
    const AprType TimeStepType = {H5T_NATIVE_UINT64, "time_steps"};

    const char * const ParticleIntensitiesType = "particle_intensities"; // type read from file
    const char * const ExtraParticleDataType = "extra_particle_data"; // type read from file
    const char * const ParticlePropertyType = "particle property"; // user defined type

    // Paraview specific
    const AprType ParaviewXType = {H5T_NATIVE_UINT16, "x"};
    const AprType ParaviewYType = {H5T_NATIVE_UINT16, "y"};
    const AprType ParaviewZType = {H5T_NATIVE_UINT16, "z"};
    const AprType ParaviewLevelType = {H5T_NATIVE_UINT8, "level"};
   // const AprType ParaviewTypeType = {H5T_NATIVE_UINT8, "type"};
}


class APRWriter {
    friend class APRFile;

protected:
    unsigned int current_t = 0;


public:

    uint64_t get_num_time_steps(const std::string &file_name){
        //
        //  Gets the number of time steps saved to the file.
        //

        FileStructure::Operation op;
        op = FileStructure::Operation::READ;
        FileStructure f(file_name, op);

        uint64_t num_time_steps;

        readAttr(AprTypes::TimeStepType, f.groupId, &num_time_steps);

        return num_time_steps;

    }

    size_t get_number_groups(const std::string &file_name){
        //
        //  Returns the number of groups in the main file.
        //

        FileStructure::Operation op;

        size_t number_time_steps = 0;

        op = FileStructure::Operation::READ;

        FileStructure f(file_name, op);

        if (!f.isOpened()) return 0;

        hsize_t num_obj;

        H5Gget_num_objs(f.groupId,&num_obj);

        number_time_steps = num_obj;

        return number_time_steps;

    }

    static bool time_adaptation_check(const std::string &file_name){
        FileStructure::Operation op;

        op = FileStructure::Operation::READ;

        FileStructure f(file_name, op);

        if (!f.isOpened()) return 0;

        H5G_info_t group_info;
        hid_t lapl_id = 0;

        //we need to turn of the error handling so if it is not found it doesn't output to cerr.
        herr_t (*old_func)(void*);
        void *old_client_data;
        H5Eget_auto1(&old_func, &old_client_data);

        /* Turn off error handling */
        H5Eset_auto1(NULL, NULL);

        herr_t exists = H5Gget_info_by_name(f.groupId,"dt",&group_info,lapl_id);

        /* Restore previous error handler */
        H5Eset_auto1(old_func, old_client_data);

        bool time_adaptive = (exists >= 0);

        return time_adaptive;

    }



    static void read_apr_parameters(hid_t dataset_id,APRParameters& parameters){
        //
        //  Reads in from hdf5 pipeline parameters
        //


        readAttr(AprTypes::LambdaType, dataset_id, &parameters.lambda);

        readAttr(AprTypes::SigmaThType, dataset_id, &parameters.sigma_th);
        readAttr(AprTypes::SigmaThMaxType, dataset_id, &parameters.sigma_th_max);
        readAttr(AprTypes::IthType, dataset_id, &parameters.Ip_th);
        readAttr(AprTypes::DxType, dataset_id, &parameters.dx);
        readAttr(AprTypes::DyType, dataset_id, &parameters.dy);
        readAttr(AprTypes::DzType, dataset_id, &parameters.dz);
        readAttr(AprTypes::PsfXType, dataset_id, &parameters.psfx);
        readAttr(AprTypes::PsfYType, dataset_id, &parameters.psfy);
        readAttr(AprTypes::PsfZType, dataset_id, &parameters.psfz);
        readAttr(AprTypes::RelativeErrorType, dataset_id, &parameters.rel_error);
        readAttr(AprTypes::BackgroundIntensityEstimateType, dataset_id,
                 &parameters.background_intensity_estimate);
        readAttr(AprTypes::NoiseSdEstimateType, dataset_id, &parameters.noise_sd_estimate);

    }

    static void read_access_info(hid_t dataset_id,APRAccess& aprAccess){
        //
        //  Reads in from hdf5 access information
        //

        readAttr(AprTypes::TotalNumberOfParticlesType, dataset_id, &aprAccess.total_number_particles);
        readAttr(AprTypes::TotalNumberOfGapsType, dataset_id, &aprAccess.total_number_gaps);

        readAttr(AprTypes::MaxLevelType, dataset_id, &aprAccess.l_max);
        readAttr(AprTypes::MinLevelType, dataset_id, &aprAccess.l_min);

        readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, dataset_id, &aprAccess.total_number_non_empty_rows);

        readAttr(AprTypes::NumberOfYType, dataset_id, &aprAccess.org_dims[0]);
        readAttr(AprTypes::NumberOfXType, dataset_id, &aprAccess.org_dims[1]);
        readAttr(AprTypes::NumberOfZType,dataset_id, &aprAccess.org_dims[2]);

        aprAccess.x_num.resize(aprAccess.level_max() + 1);
        aprAccess.y_num.resize(aprAccess.level_max() + 1);
        aprAccess.z_num.resize(aprAccess.level_max() + 1);

        for (size_t i = aprAccess.level_min(); i < aprAccess.level_max(); i++) {
            int x_num, y_num, z_num;
            //TODO: x_num and other should have HDF5 type uint64?
            readAttr(AprTypes::NumberOfLevelXType, i, dataset_id, &x_num);
            readAttr(AprTypes::NumberOfLevelYType, i, dataset_id, &y_num);
            readAttr(AprTypes::NumberOfLevelZType, i,dataset_id, &z_num);
            aprAccess.x_num[i] = x_num;
            aprAccess.y_num[i] = y_num;
            aprAccess.z_num[i] = z_num;
        }

        aprAccess.y_num[aprAccess.level_max()] = aprAccess.org_dims[0];
        aprAccess.x_num[aprAccess.level_max()] = aprAccess.org_dims[1];
        aprAccess.z_num[aprAccess.level_max()] = aprAccess.org_dims[2];


    }


    void read_apr(APR& apr, const std::string &file_name,bool build_tree = false, int max_level_delta = 0,unsigned int time_point = UINT32_MAX) {

        APRTimer timer;
        timer.verbose_flag = false;

        APRTimer timer_f;
        timer_f.verbose_flag = false;

        FileStructure::Operation op;

        if(build_tree){
            op = FileStructure::Operation::READ_WITH_TREE;
        } else {
            op = FileStructure::Operation::READ;
        }

        unsigned int t = 0;

        if(time_point!=UINT32_MAX){
            t = time_point;
        }

        FileStructure f(file_name, op);

        hid_t meta_data = f.groupId;

        //check if old or new file, for location of the properties. (The metadata moved to the time point.)
        bool old = attribute_exists(f.objectId,AprTypes::MaxLevelType.typeName);

        if(old) {
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

        //read in access information
        read_access_info(meta_data,apr.apr_access);

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
            //read in pipeline parameters
            read_apr_parameters(meta_data,apr.parameters);

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

            timer.stop_timer();

            timer.start_timer("map building");

            apr.apr_access.rebuild_map(*map_data);

            timer.stop_timer();
        }

        uint64_t max_read_level = apr.apr_access.level_max()-max_level_delta;
        uint64_t max_read_level_tree = std::min(apr.apr_access.level_max()-1,max_read_level);
        uint64_t prev_read_level = 0;

//        if(build_tree){
//
//
//            if(read_structure) {
//
//
//                timer.start_timer("build tree - map");
//
//                apr.apr_tree.tree_access.l_max = apr.level_max() - 1;
//                apr.apr_tree.tree_access.l_min = apr.level_min() - 1;
//
//                apr.apr_tree.tree_access.x_num.resize(apr.apr_tree.tree_access.level_max() + 1);
//                apr.apr_tree.tree_access.z_num.resize(apr.apr_tree.tree_access.level_max() + 1);
//                apr.apr_tree.tree_access.y_num.resize(apr.apr_tree.tree_access.level_max() + 1);
//
//                for (int i = apr.apr_tree.tree_access.level_min(); i <= apr.apr_tree.tree_access.level_max(); ++i) {
//                    apr.apr_tree.tree_access.x_num[i] = apr.spatial_index_x_max(i);
//                    apr.apr_tree.tree_access.y_num[i] = apr.spatial_index_y_max(i);
//                    apr.apr_tree.tree_access.z_num[i] = apr.spatial_index_z_max(i);
//                }
//
//                apr.apr_tree.tree_access.x_num[apr.level_min() - 1] = ceil(apr.spatial_index_x_max(apr.level_min()) / 2.0f);
//                apr.apr_tree.tree_access.y_num[apr.level_min() - 1] = ceil(apr.spatial_index_y_max(apr.level_min()) / 2.0f);
//                apr.apr_tree.tree_access.z_num[apr.level_min() - 1] = ceil(apr.spatial_index_z_max(apr.level_min()) / 2.0f);
//
//                readAttr(AprTypes::TotalNumberOfParticlesType, f.objectIdTree,
//                         &apr.apr_tree.tree_access.total_number_particles);
//                readAttr(AprTypes::TotalNumberOfGapsType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_gaps);
//                readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, f.objectIdTree,
//                         &apr.apr_tree.tree_access.total_number_non_empty_rows);
//
//                auto map_data_tree = std::make_shared<MapStorageData>();
//
//                map_data_tree->global_index.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
//
//
//                std::vector<int16_t> index_delta(apr.apr_tree.tree_access.total_number_non_empty_rows);
//                readData(AprTypes::MapGlobalIndexType, f.objectIdTree, index_delta.data());
//                std::vector<uint64_t> index_delta_big(apr.apr_tree.tree_access.total_number_non_empty_rows);
//                std::copy(index_delta.begin(), index_delta.end(), index_delta_big.begin());
//                std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data_tree->global_index.begin());
//
//
//                map_data_tree->y_end.resize(apr.apr_tree.tree_access.total_number_gaps);
//                readData(AprTypes::MapYendType, f.objectIdTree, map_data_tree->y_end.data());
//                map_data_tree->y_begin.resize(apr.apr_tree.tree_access.total_number_gaps);
//                readData(AprTypes::MapYbeginType, f.objectIdTree, map_data_tree->y_begin.data());
//
//                map_data_tree->number_gaps.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
//                readData(AprTypes::MapNumberGapsType, f.objectIdTree, map_data_tree->number_gaps.data());
//                map_data_tree->level.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
//                readData(AprTypes::MapLevelType, f.objectIdTree, map_data_tree->level.data());
//                map_data_tree->x.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
//                readData(AprTypes::MapXType, f.objectIdTree, map_data_tree->x.data());
//                map_data_tree->z.resize(apr.apr_tree.tree_access.total_number_non_empty_rows);
//                readData(AprTypes::MapZType, f.objectIdTree, map_data_tree->z.data());
//
//                apr.apr_tree.tree_access.rebuild_map_tree(*map_data_tree,apr.apr_access);
//
//                //Important needs linking to the APR
//                apr.apr_tree.APROwn = &apr;
//
//                timer.stop_timer();
//            }
//
//            if(!read_structure) {
//                uint64_t current_parts_size = apr.apr_tree.particles_ds_tree.data.size();
//
//                for (int j = apr.level_min(); j <apr.level_max(); ++j) {
//                    if((apr.apr_tree.tree_access.global_index_by_level_end[j] + 1)==current_parts_size){
//                        prev_read_level = j;
//                    }
//                }
//            }
//
//            timer.start_timer("tree intensities");
//
//            uint64_t parts_start = 0;
//            if(prev_read_level > 0){
//                parts_start = apr.apr_tree.tree_access.global_index_by_level_end[prev_read_level] + 1;
//            }
//            uint64_t parts_end = apr.apr_tree.tree_access.global_index_by_level_end[max_read_level_tree] + 1;
//
//            apr.apr_tree.particles_ds_tree.data.resize(parts_end);
//
//            if ( apr.apr_tree.particles_ds_tree.data.size() > 0) {
//                readData(AprTypes::ParticleIntensitiesType, f.objectIdTree, apr.apr_tree.particles_ds_tree.data.data() + parts_start,parts_start,parts_end);
//            }
//
//            APRCompress apr_compress;
//            apr_compress.set_compression_type(1);
//            apr_compress.set_quantization_factor(2);
//            apr_compress.decompress(apr, apr.apr_tree.particles_ds_tree,parts_start);
//
//            timer.stop_timer();
//
//        }

        uint64_t parts_start = 0;
        uint64_t parts_end = apr.apr_access.global_index_by_level_end[max_read_level] + 1;

        prev_read_level = 0;
        if(!read_structure) {
            uint64_t current_parts_size = apr.total_number_particles();

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

//        timer.start_timer("Read intensities");
//        // ------------- read data ------------------------------
//        apr.particles_intensities.data.resize(parts_end);
//        if (apr.particles_intensities.data.size() > 0) {
//            readData(AprTypes::ParticleIntensitiesType, f.objectId, apr.particles_intensities.data.data() + parts_start,parts_start,parts_end);
//        }


        timer.stop_timer();

        //std::cout << "Data rate intensities: " << (apr.particles_intensities.data.size()*2)/(timer.timings.back()*1000000.0f) << " MB/s" << std::endl;


        timer.start_timer("decompress");
        // ------------ decompress if needed ---------------------
        if (compress_type > 0) {
//            APRCompress apr_compress;
//            apr_compress.set_compression_type(compress_type);
//            apr_compress.set_quantization_factor(quantization_factor);
//            apr_compress.decompress(apr, apr.particles_intensities,parts_start);
        }
        timer.stop_timer();
    }


    FileSizeInfo write_apr(APR& apr, const std::string &save_loc, const std::string &file_name) {
        APRCompress apr_compressor;
        apr_compressor.set_compression_type(0);
        return write_apr(apr, save_loc, file_name, apr_compressor);
    }

    FileSizeInfo write_apr_append(APR& apr, const std::string &save_loc, const std::string &file_name) {
        APRCompress apr_compressor;
        apr_compressor.set_compression_type(0);
        return write_apr(apr, save_loc, file_name, apr_compressor,BLOSC_ZSTD,4,1,false,true);
    }



    /**
     * Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
     */

    FileSizeInfo write_apr(APR &apr, const std::string &save_loc, const std::string &file_name, APRCompress &apr_compressor, unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 2, unsigned int blosc_shuffle=1,bool write_tree = false,bool append_apr_time = true) {
        APRTimer write_timer;
        write_timer.verbose_flag = true;

        std::string hdf5_file_name = save_loc + file_name + ".h5";


        FileStructure::Operation op;

        if(write_tree){
            op = FileStructure::Operation::WRITE_WITH_TREE;
        } else {
            op = FileStructure::Operation::WRITE;
        }

        uint64_t t = 0;

        if(append_apr_time){
            t = current_t;
            current_t++;
        }

        FileStructure f(hdf5_file_name, op);

        FileSizeInfo fileSizeInfo1;
        if (!f.isOpened()) return fileSizeInfo1;

        hid_t meta_location = f.groupId;

        if(append_apr_time){
            meta_location = f.objectId;
        }

        //global property
        writeAttr(AprTypes::TimeStepType, f.groupId, &t);

        // ------------- write metadata -------------------------
        //per time step

        writeAttr(AprTypes::NumberOfXType, meta_location, &apr.apr_access.org_dims[1]);
        writeAttr(AprTypes::NumberOfYType, meta_location, &apr.apr_access.org_dims[0]);
        writeAttr(AprTypes::NumberOfZType, meta_location, &apr.apr_access.org_dims[2]);
        writeAttr(AprTypes::TotalNumberOfGapsType, meta_location, &apr.apr_access.total_number_gaps);
        writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, meta_location, &apr.apr_access.total_number_non_empty_rows);

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

        // #TODO: New write file APRTree

//        bool store_tree = write_tree;
//
//        if(store_tree){
//
//            if(apr.apr_tree.particles_ds_tree.data.size()==0) {
//                apr.apr_tree.init(apr);
//                apr.apr_tree.fill_tree_mean_downsample(apr.particles_intensities);
//            }
//
//            writeAttr(AprTypes::TotalNumberOfGapsType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_gaps);
//            writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_non_empty_rows);
//            writeAttr(AprTypes::TotalNumberOfParticlesType, f.objectIdTree, &apr.apr_tree.tree_access.total_number_particles);
//
//            MapStorageData map_data_tree;
//            apr.apr_tree.tree_access.flatten_structure( map_data_tree);
//
//            std::vector<uint16_t> index_delta;
//            index_delta.resize(map_data_tree.global_index.size());
//            std::adjacent_difference(map_data_tree.global_index.begin(),map_data_tree.global_index.end(),index_delta.begin());
//            writeData(AprTypes::MapGlobalIndexType, f.objectIdTree, index_delta, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//
//            writeData(AprTypes::MapYendType, f.objectIdTree, map_data_tree.y_end, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//            writeData(AprTypes::MapYbeginType, f.objectIdTree, map_data_tree.y_begin, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//            writeData(AprTypes::MapNumberGapsType, f.objectIdTree, map_data_tree.number_gaps, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//            writeData(AprTypes::MapLevelType, f.objectIdTree, map_data_tree.level, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//            writeData(AprTypes::MapXType, f.objectIdTree, map_data_tree.x, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//            writeData(AprTypes::MapZType, f.objectIdTree, map_data_tree.z, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//
//            APRCompress tree_compress;
//            tree_compress.set_compression_type(1);
//            tree_compress.set_quantization_factor(2);
//
//            tree_compress.compress(apr, apr.apr_tree.particles_ds_tree);
//
//            unsigned int tree_blosc_comp_type = 6;
//
//            hid_t type = Hdf5Type<ImageType>::type();
//            writeData({type, AprTypes::ParticleIntensitiesType}, f.objectIdTree, apr.apr_tree.particles_ds_tree.data, tree_blosc_comp_type, blosc_comp_level, blosc_shuffle);
//
//        }

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
//        write_timer.start_timer("intensities");
//        if (compress_type_num > 0){
//            apr_compressor.compress(apr,apr.particles_intensities);
//        }
//        hid_t type = Hdf5Type<ImageType>::type();
//        writeData({type, AprTypes::ParticleIntensitiesType}, meta_location, apr.particles_intensities.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);
//        write_timer.stop_timer();

        // ------------- output the file size -------------------
        file_size = f.getFileSize();
        double sizeMB = file_size / 1e6;

        fileSizeInfo.total_file_size = sizeMB;
        fileSizeInfo.intensity_data = fileSizeInfo.total_file_size - fileSizeInfo.access_data;

        std::cout << "HDF5 Total Filesize: " << sizeMB << " MB\n" << "Writing Complete" << std::endl;
        return fileSizeInfo;
    }




    template<typename T>
    void write_apr_paraview(APR &apr, const std::string &save_loc, const std::string &file_name, const ParticleData<T> &parts,std::vector<uint64_t> previous_num = {0}) {
        std::string hdf5_file_name = save_loc + file_name + ".h5";

        bool write_time = false;
        unsigned int t = 0;
        if(previous_num[0]!=0){
            //time series write
            write_time = true;
            t = (previous_num.size()-1);
        }


        FileStructure f(hdf5_file_name, FileStructure::Operation::WRITE);
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
    float write_particles_only(const std::string &save_loc, const std::string &file_name, const ParticleData<S> &parts_extra) {
        std::string hdf5_file_name = save_loc + file_name + ".h5";

        FileStructure f{hdf5_file_name, FileStructure::Operation::WRITE};
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
    void read_parts_only(const std::string &aFileName, ParticleData<T>& extra_parts) {
        FileStructure f{aFileName, FileStructure::Operation::READ};
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

        FileStructure f{hdf5_file_name, FileStructure::Operation::WRITE};
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


    struct FileStructure {
        enum class Operation {READ, WRITE,READ_WITH_TREE,WRITE_WITH_TREE,WRITE_APPEND,WRITE_APPEND_WITH_TREE};
        hid_t fileId = -1;
        hid_t groupId = -1;
        hid_t objectId = -1;
        hid_t objectIdTree = -1;

        FileStructure(){};

        FileStructure(const std::string &aFileName, const Operation aOp){
            init(aFileName,aOp);
        }


        bool open_time_point(const unsigned int t,bool tree,std::string t_string = "t"){

            //first close existing time point
            close_time_point();

            if(t==0){
                //do nothing
            } else{
                t_string = t_string + std::to_string(t);
            }

            std::string subGroup1 = ("ParticleRepr/" + t_string);
            std::string subGroupTree1 = "ParticleRepr/" + t_string + "/Tree";

            const char * const subGroup  = subGroup1.c_str();
            const char * const subGroupTree  = subGroupTree1.c_str();

            //need to check if the tree exists

            if(tree){

                objectId = H5Gopen2(fileId, subGroup, H5P_DEFAULT);

                bool tree_exists = group_exists(fileId,subGroupTree);

                if(tree_exists) {
                    //tree doesn't exist can't open it
                    objectIdTree = H5Gopen2(fileId, subGroupTree, H5P_DEFAULT);
                    return false;
                }

            } else {
                objectId = H5Gopen2(fileId, subGroup, H5P_DEFAULT);
            }

            return true;

        }



        void create_time_point(const unsigned int t,bool tree,std::string t_string = "t"){

            close_time_point();

            if(t==0){
                //do nothing
            } else{
                t_string = t_string + std::to_string(t);
            }

            std::string subGroup1 = ("ParticleRepr/" + t_string);
            std::string subGroupTree1 = "ParticleRepr/" + t_string + "/Tree";


            const char * const subGroup  = subGroup1.c_str();
            const char * const subGroupTree  = subGroupTree1.c_str();


            if(tree){
                if(!group_exists(fileId,subGroup)) {
                    objectId = H5Gcreate2(fileId, subGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                }
                if(!group_exists(fileId,subGroupTree)) {
                    objectIdTree = H5Gcreate2(fileId, subGroupTree, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                }
            } else {
                if(!group_exists(fileId,subGroup)) {
                    objectId = H5Gcreate2(fileId, subGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                }
            }

        }

        bool init(const std::string &aFileName, const Operation aOp){

            const char * const mainGroup =  ("ParticleRepr");
//            const char * const subGroup  = subGroup1.c_str();
//            const char * const subGroupTree  = subGroupTree1.c_str();

            hdf5_register_blosc();
            switch(aOp) {
                case Operation::READ:

                    fileId = H5Fopen(aFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                    if (fileId == -1) {
                        std::cerr << "Could not open file [" << aFileName << "]" << std::endl;
                        return false;
                    }
                    groupId = H5Gopen2(fileId, mainGroup, H5P_DEFAULT);

                    break;
                case Operation::WRITE:

                    fileId = hdf5_create_file_blosc(aFileName);

                    if (fileId == -1) {
                        std::cerr << "Could not create file [" << aFileName << "]" << std::endl;
                        return false;
                    }


                    groupId = H5Gcreate2(fileId, mainGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                    break;
                case Operation::WRITE_APPEND:

                    fileId = H5Fopen(aFileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                    if (fileId == -1) {
                        std::cerr << "Could not open file [" << aFileName << "]" << std::endl;
                        return false;
                    }
                    groupId = H5Gopen2(fileId, mainGroup, H5P_DEFAULT);

                    break;
            }
            if (groupId == -1) { H5Fclose(fileId); fileId = -1; return false;}
            return true;
        }
        ~FileStructure() {
            if (objectIdTree != -1) H5Gclose(objectIdTree);
            if (objectId != -1) H5Gclose(objectId);
            if (groupId != -1) H5Gclose(groupId);
            if (fileId != -1) H5Fclose(fileId);
        }

        bool close(){
            if (objectIdTree != -1) H5Gclose(objectIdTree);
            if (objectId != -1) H5Gclose(objectId);
            if (groupId != -1) H5Gclose(groupId);
            if (fileId != -1) H5Fclose(fileId);

            fileId = -1;
            objectIdTree = -1;
            groupId = -1;
            objectId = -1;

            return true;
        }

        bool close_time_point(){
            if (objectIdTree != -1) H5Gclose(objectIdTree);
            if (objectId != -1) H5Gclose(objectId);

            objectId = -1;
            objectIdTree = -1;

            return true;
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
    static void readAttr(const AprType &aType, hid_t aGroupId, void *aDest) {
        hid_t attr_id = H5Aopen(aGroupId, aType.typeName, H5P_DEFAULT);
        H5Aread(attr_id, aType.hdf5type, aDest);
        H5Aclose(attr_id);
    }

    static void readAttr(const AprType &aType, size_t aSuffix, hid_t aGroupId, void *aDest) {
        std::string typeNameWithPrefix = std::string(aType.typeName) + std::to_string(aSuffix);
        hid_t attr_id = H5Aopen(aGroupId, typeNameWithPrefix.c_str(), H5P_DEFAULT);
        H5Aread(attr_id, aType.hdf5type, aDest);
        H5Aclose(attr_id);
    }

    static void writeAttr(const AprType &aType, hid_t aGroupId, const void * const aSrc) {
        hsize_t dims[] = {1};
        hdf5_write_attribute_blosc(aGroupId, aType.hdf5type, aType.typeName, 1, dims, aSrc);
    }

    static void writeAttr(const AprType &aType, size_t aSuffix, hid_t aGroupId, const void * const aSrc) {
        std::string typeNameWithPrefix = std::string(aType.typeName) + std::to_string(aSuffix);
        writeAttr({aType.hdf5type, typeNameWithPrefix.c_str()}, aGroupId, aSrc);
    }

    static void readData(const AprType &aType, hid_t aObjectId, void *aDest) {
        hdf5_load_data_blosc(aObjectId, aType.hdf5type, aDest, aType.typeName);
    }

    static void readData(const char * const aAprTypeName, hid_t aObjectId, void *aDest) {
        hdf5_load_data_blosc(aObjectId, aDest, aAprTypeName);
    }

    static void readData(const char * const aAprTypeName, hid_t aObjectId, void *aDest,uint64_t elements_start,uint64_t elements_end) {
        //reads partial dataset
        hdf5_load_data_blosc_partial(aObjectId, aDest, aAprTypeName,elements_start,elements_end);
    }

    //hdf5_load_data_blosc_partial(hid_t obj_id, void* buff, const char* data_name,uint64_t number_of_elements_read,uint64_t number_of_elements_total)


    template<typename T>
    static void writeData(const AprType &aType, hid_t aObjectId, T aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_blosc(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);
    }

    template<typename T>
    static uint64_t writeDataAppend(const AprType &aType, hid_t aObjectId, T aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {


        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;

        int out = H5Lexists( aObjectId, aType.typeName, H5P_DEFAULT );

        if(out==0){
            if(dims[0] > 0) {
                hdf5_write_data_blosc_create(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(),
                                             blosc_comp_type, blosc_comp_level, blosc_shuffle);
                return dims[0];
            }else {
                return 0;
            }
        } else {

            return hdf5_write_data_blosc_append(aObjectId, aType.hdf5type, aType.typeName, aContainer.data(), dims);

        }

    }


    template<typename T>
    static void writeDataStandard(const AprType &aType, hid_t aObjectId, T aContainer) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_standard(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data());
    }

    static void writeString(AprType aTypeName, hid_t aGroupId, const std::string &aValue) {
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
