//
// Created by cheesema on 14.01.18.
//

#ifndef APRWRITER_HPP
#define APRWRITER_HPP

#include "hdf5functions_blosc.h"
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/access/RandomAccess.hpp"
#include "ConfigAPR.h"
#include <numeric>
#include <memory>
#include <data_structures/APR/APR.hpp>
#include "numerics/APRCompress.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"


/*
 *  Class to handle generation of datasets using hdf5 using blosc
 *
 *  Bundles things together to allow some more general functionality in a modular way.
 *
 */
class Hdf5DataSet {

    hid_t obj_id=-1;
    hid_t data_id=-1;

    hid_t memspace_id=-1;
    hid_t dataspace_id=-1;

    hid_t dataType=-1;

    hsize_t dims;

    hsize_t offset;
    hsize_t count ;
    hsize_t stride;
    hsize_t block;

    std::string data_name;

public:

    unsigned int blosc_comp_type = BLOSC_ZSTD;
    unsigned int blosc_comp_level = 2;
    unsigned int blosc_shuffle=1;

    void init(hid_t obj_id_, const char* data_name_){

        obj_id = obj_id_;
        data_name = data_name_;

    }

    void create(hid_t type_id,uint64_t number_elements){
        hsize_t rank = 1;
        hsize_t dims = number_elements;

        hdf5_create_dataset_blosc(  obj_id,  type_id,  data_name.c_str(),  rank, &dims,blosc_comp_type,blosc_comp_level,blosc_shuffle);
    }

    void write(void* buff,uint64_t elements_start,uint64_t elements_end){

        dims = elements_end - elements_start;

        offset = elements_start;
        count = dims;

#ifdef HAVE_OPENMP
#pragma omp critical
#endif
        {
            H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, &offset,
                                &stride, &count, &block);

            memspace_id = H5Screate_simple(1, &dims, NULL);

            H5Dwrite(data_id, dataType, memspace_id, dataspace_id, H5P_DEFAULT, buff);
        }
    }

    void read(void* buff,uint64_t elements_start,uint64_t elements_end){
        dims = elements_end - elements_start;

        offset = elements_start;
        count = dims;
#ifdef HAVE_OPENMP
#pragma omp critical
#endif
        {
            H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, &offset,
                                 &stride, &count, &block);

            memspace_id = H5Screate_simple (1, &dims, NULL);


            H5Dread(data_id, dataType, memspace_id, dataspace_id, H5P_DEFAULT, buff);
        }
    }

    void open(){

        data_id =  H5Dopen2(obj_id, data_name.c_str() ,H5P_DEFAULT);

        dataType = H5Dget_type(data_id);

        stride = 1;
        block = 1;

        dataspace_id = H5Dget_space (data_id);
    }

    std::vector<uint64_t> get_dimensions(){

        const int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        std::vector<hsize_t> _dims;
        _dims.resize(ndims,0);

        H5Sget_simple_extent_dims(dataspace_id, _dims.data(), NULL);

        std::vector<uint64_t> dims_u64;
        dims_u64.resize(ndims,0);
        std::copy(_dims.begin(),_dims.end(),dims_u64.begin());

        return dims_u64;
    }


    inline hid_t get_type() {
        return dataType;
    }


    void close(){
        if(memspace_id!=-1) {
            H5Sclose(memspace_id);
            memspace_id = -1;
        }

        if(dataspace_id!=-1) {
            H5Sclose(dataspace_id);
            dataspace_id = -1;
        }

        if(dataType!=-1) {
            H5Tclose(dataType);
            dataType = -1;
        }

        if(data_id!=-1) {
            H5Dclose(data_id);
            data_id = -1;
        }

    }

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
    const AprType CompressBackgroundType = {H5T_NATIVE_FLOAT, "compress_background"};
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

    const AprType GradientThreshold = {H5T_NATIVE_FLOAT, "grad_th"};

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
    template<typename T>
    friend class LazyData;

    friend class LazyAccess;

protected:
    unsigned int current_t = 0;


public:

    struct ReadPatch{

        int x_begin =0;
        int x_end = 0;

        int z_begin =0;
        int z_end = 0;

        int level_begin = 0;
        int level_end = 0;

    };

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


    template<typename T>
    static void re_order_parts(APR& apr,ParticleData<T>& parts){
        // backward compatability function -- not performance orientated.

        ParticleData<T> parts_temp;
        parts_temp.init(parts.size());

        int level = apr.level_max();

        parts_temp.copy_parts(apr,parts);

        auto apr_iterator = apr.random_iterator();
        auto apr_iterator_2 = apr.random_iterator();
        int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(apr_iterator,apr_iterator_2)
#endif
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (int x = 0; x < apr_iterator.x_num(level); ++x) {

                apr_iterator_2.set_new_lzx_old(level, z, x);
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator++) {

                    parts[apr_iterator] = parts_temp[apr_iterator_2];

                    if(apr_iterator_2 < apr_iterator_2.end()){
                        apr_iterator_2++;
                    }
                }
            }
        }



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

    static void write_linear_access(hid_t meta_data,hid_t objectId, LinearAccess& linearAccess,unsigned int blosc_comp_type_access, unsigned int blosc_comp_level_access,unsigned int blosc_shuffle_access){

        APRWriter::writeData({H5T_NATIVE_UINT16,"y_vec"}, objectId, linearAccess.y_vec, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

        APRWriter::writeData({H5T_NATIVE_UINT64,"xz_end_vec"}, objectId, linearAccess.xz_end_vec, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

    }

    static void read_linear_access(hid_t objectId, LinearAccess& linearAccess,int max_level_delta){

        linearAccess.genInfo->l_max = std::max(linearAccess.genInfo->l_max - max_level_delta,linearAccess.level_min());
        linearAccess.initialize_xz_linear(); //initialize the structures based on size.

        auto level_ = linearAccess.genInfo->l_max;
        uint64_t index = linearAccess.level_xz_vec[level_] + linearAccess.x_num(level_) - 1 + (linearAccess.z_num(level_)-1)*linearAccess.x_num(level_);


        uint64_t begin_index = 0;
        uint64_t end_index = linearAccess.level_xz_vec[level_+1];

        APRWriter::readData("xz_end_vec", objectId, linearAccess.xz_end_vec.data(),begin_index,end_index);

        uint64_t begin_y = 0;
        uint64_t end_y = linearAccess.xz_end_vec[index];

        linearAccess.y_vec.resize(end_y - begin_y);
        read_linear_y(objectId, linearAccess.y_vec,begin_y,end_y);

    }


    template<typename T>
    static void read_linear_y(hid_t objectId,  T &aContainer,uint64_t begin, uint64_t end){

        APRWriter::readData("y_vec", objectId, aContainer.data(),begin,end);
    }


    static void write_random_access(hid_t meta_data,hid_t objectId, RandomAccess& apr_access,unsigned int blosc_comp_type_access, unsigned int blosc_comp_level_access,unsigned int blosc_shuffle_access){

        MapStorageData map_data;
        apr_access.flatten_structure( map_data);

        APRWriter::writeAttr(AprTypes::TotalNumberOfGapsType, meta_data, &apr_access.total_number_gaps);
        APRWriter::writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, meta_data, &apr_access.total_number_non_empty_rows);

        std::vector<uint16_t> index_delta;
        index_delta.resize(map_data.global_index.size());
        std::adjacent_difference(map_data.global_index.begin(),map_data.global_index.end(),index_delta.begin());
        APRWriter::writeData(AprTypes::MapGlobalIndexType, objectId, index_delta, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

        APRWriter::writeData(AprTypes::MapYendType, objectId, map_data.y_end, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapYbeginType, objectId, map_data.y_begin, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapNumberGapsType, objectId, map_data.number_gaps, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapLevelType, objectId, map_data.level, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapXType, objectId, map_data.x, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);
        APRWriter::writeData(AprTypes::MapZType, objectId, map_data.z, blosc_comp_type_access, blosc_comp_level_access, blosc_shuffle_access);

    }

    static void write_apr_info(hid_t meta_location,GenInfo& aprInfo){

        APRWriter::writeAttr(AprTypes::NumberOfXType, meta_location, &aprInfo.org_dims[1]);
        APRWriter::writeAttr(AprTypes::NumberOfYType, meta_location, &aprInfo.org_dims[0]);
        APRWriter::writeAttr(AprTypes::NumberOfZType, meta_location, &aprInfo.org_dims[2]);

        APRWriter::writeAttr(AprTypes::TotalNumberOfParticlesType, meta_location, &aprInfo.total_number_particles);
        APRWriter::writeAttr(AprTypes::MaxLevelType, meta_location, &aprInfo.l_max);
        APRWriter::writeAttr(AprTypes::MinLevelType, meta_location, &aprInfo.l_min);

        for (int i = aprInfo.l_min; i < aprInfo.l_max ; ++i) {
            int x_num = (int) aprInfo.x_num[i];
            APRWriter::writeAttr(AprTypes::NumberOfLevelXType, i, meta_location, &x_num);
            int y_num = (int) aprInfo.y_num[i];
            APRWriter::writeAttr(AprTypes::NumberOfLevelYType, i, meta_location, &y_num);
            int z_num = (int) aprInfo.z_num[i];
            APRWriter::writeAttr(AprTypes::NumberOfLevelZType, i, meta_location, &z_num);
        }

    }

    static void write_apr_parameters(hid_t dataset_id,APRParameters& parameters){

        APRWriter::writeAttr(AprTypes::LambdaType, dataset_id, &parameters.lambda);
        APRWriter::writeAttr(AprTypes::SigmaThType, dataset_id, &parameters.sigma_th);
        APRWriter::writeAttr(AprTypes::SigmaThMaxType, dataset_id, &parameters.sigma_th_max);
        APRWriter::writeAttr(AprTypes::IthType, dataset_id, &parameters.Ip_th);
        APRWriter::writeAttr(AprTypes::DxType, dataset_id, &parameters.dx);
        APRWriter::writeAttr(AprTypes::DyType, dataset_id, &parameters.dy);
        APRWriter::writeAttr(AprTypes::DzType, dataset_id, &parameters.dz);
        APRWriter::writeAttr(AprTypes::PsfXType, dataset_id, &parameters.psfx);
        APRWriter::writeAttr(AprTypes::PsfYType, dataset_id, &parameters.psfy);
        APRWriter::writeAttr(AprTypes::PsfZType, dataset_id, &parameters.psfz);
        APRWriter::writeAttr(AprTypes::RelativeErrorType, dataset_id, &parameters.rel_error);
        APRWriter::writeAttr(AprTypes::NoiseSdEstimateType, dataset_id, &parameters.noise_sd_estimate);
        APRWriter::writeAttr(AprTypes::BackgroundIntensityEstimateType, dataset_id,
                             &parameters.background_intensity_estimate);


        APRWriter::writeAttr(AprTypes::GradientThreshold, dataset_id,
                             &parameters.grad_th);


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

        if(attribute_exists(dataset_id,AprTypes::GradientThreshold.typeName)) {
            readAttr(AprTypes::GradientThreshold, dataset_id, &parameters.grad_th);
        }

    }

    static void read_random_tree_access(hid_t meta_data,hid_t objectId, RandomAccess& tree_access,RandomAccess& apr_access){
        read_random_access_int( meta_data, objectId,  tree_access, apr_access,true);
    }

    static void read_random_access(hid_t meta_data,hid_t objectId, RandomAccess& apr_access){
        RandomAccess empty_access;
        read_random_access_int( meta_data, objectId,  apr_access, empty_access,false);
    }

    static void read_random_access_int(hid_t meta_data,hid_t objectId, RandomAccess& apr_access,RandomAccess& own_access,bool tree = false){

        APRWriter::readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, meta_data, &apr_access.total_number_non_empty_rows);
        APRWriter::readAttr(AprTypes::TotalNumberOfGapsType, meta_data, &apr_access.total_number_gaps);

        // ------------- map handling ----------------------------

        auto map_data = std::make_shared<MapStorageData>();

        map_data->global_index.resize(apr_access.total_number_non_empty_rows);

        std::vector<int16_t> index_delta(apr_access.total_number_non_empty_rows);
        APRWriter::readData(AprTypes::MapGlobalIndexType, objectId, index_delta.data());
        std::vector<uint64_t> index_delta_big(apr_access.total_number_non_empty_rows);
        std::copy(index_delta.begin(), index_delta.end(), index_delta_big.begin());
        std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data->global_index.begin());

        map_data->y_end.resize(apr_access.total_number_gaps);
        APRWriter::readData(AprTypes::MapYendType, objectId, map_data->y_end.data());
        map_data->y_begin.resize(apr_access.total_number_gaps);
        APRWriter::readData(AprTypes::MapYbeginType, objectId, map_data->y_begin.data());

        map_data->number_gaps.resize(apr_access.total_number_non_empty_rows);
        APRWriter::readData(AprTypes::MapNumberGapsType, objectId, map_data->number_gaps.data());
        map_data->level.resize(apr_access.total_number_non_empty_rows);
        APRWriter::readData(AprTypes::MapLevelType, objectId, map_data->level.data());
        map_data->x.resize(apr_access.total_number_non_empty_rows);
        APRWriter::readData(AprTypes::MapXType, objectId, map_data->x.data());
        map_data->z.resize(apr_access.total_number_non_empty_rows);
        APRWriter::readData(AprTypes::MapZType, objectId, map_data->z.data());

        if(tree){
            //also needs the APR access
            apr_access.rebuild_map_tree(*map_data,own_access);
        } else{
            apr_access.rebuild_map(*map_data);
        }


    }


    static void read_access_info(hid_t dataset_id, GenInfo& aprInfo){
        //
        //  Reads in from hdf5 access information
        //

        readAttr(AprTypes::TotalNumberOfParticlesType, dataset_id, &aprInfo.total_number_particles);

        readAttr(AprTypes::NumberOfYType, dataset_id, &aprInfo.org_dims[0]);
        readAttr(AprTypes::NumberOfXType, dataset_id, &aprInfo.org_dims[1]);
        readAttr(AprTypes::NumberOfZType,dataset_id, &aprInfo.org_dims[2]);

        aprInfo.init(aprInfo.org_dims[0],aprInfo.org_dims[1],aprInfo.org_dims[2]);
    }

    static void read_dims(hid_t dataset_id, GenInfo& aprInfo) {
        readAttr(AprTypes::NumberOfYType, dataset_id, &aprInfo.org_dims[0]);
        readAttr(AprTypes::NumberOfXType, dataset_id, &aprInfo.org_dims[1]);
        readAttr(AprTypes::NumberOfZType,dataset_id, &aprInfo.org_dims[2]);
    }

    static void read_access_info_tree(hid_t dataset_id, GenInfo& aprInfo){
        readAttr(AprTypes::TotalNumberOfParticlesType, dataset_id, &aprInfo.total_number_particles);
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
        enum class Operation {READ, WRITE,WRITE_APPEND};
        hid_t fileId = -1;
        hid_t groupId = -1;
        hid_t objectId = -1;
        hid_t objectIdTree = -1;

        std::string subGroup1;
        std::string subGroupTree1;

        FileStructure(){};

        FileStructure(const std::string &aFileName, const Operation aOp){
            init(aFileName,aOp);
        }


        bool open_time_point(const unsigned int t, bool require_tree, std::string t_string = "t") {

            // first close existing time point
            close_time_point();

            if(t != 0){
                t_string = t_string + std::to_string(t);
            }

            subGroup1 = ("ParticleRepr/" + t_string);
            subGroupTree1 = "ParticleRepr/" + t_string + "/Tree";

            const char * const subGroup = subGroup1.c_str();
            const char * const subGroupTree = subGroupTree1.c_str();

            // check if APR data exists
            if(!group_exists(fileId, subGroup)){
                std::cerr << "Error reading APR file: could not open time point t=" << t << std::endl;
                return false;
            }

            // open group
            objectId = H5Gopen2(fileId, subGroup, H5P_DEFAULT);

            // check if tree data exists
            const bool tree_exists = group_exists(fileId, subGroupTree);

            if(require_tree && !tree_exists) {
                std::cerr << "Error reading APR file: requested tree data does not exist" << std::endl;
                return false;
            }

            // open tree group if it exists
            if(tree_exists) {
                objectIdTree = H5Gopen2(fileId, subGroupTree, H5P_DEFAULT);
            }

            return true;
        }



        void create_time_point(const unsigned int t, bool with_tree, std::string t_string = "t"){

            close_time_point();

            if(t != 0) {
                t_string = t_string + std::to_string(t);
            }

            subGroup1 = ("ParticleRepr/" + t_string);
            subGroupTree1 = "ParticleRepr/" + t_string + "/Tree";

            const char * const subGroup = subGroup1.c_str();
            const char * const subGroupTree = subGroupTree1.c_str();

            if(!group_exists(fileId, subGroup)) {
                objectId = H5Gcreate2(fileId, subGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            } else {
                objectId = H5Gopen2(fileId, subGroup, H5P_DEFAULT);
            }

            if(with_tree){
                if(!group_exists(fileId, subGroupTree)) {
                    objectIdTree = H5Gcreate2(fileId, subGroupTree, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                } else {
                    objectIdTree = H5Gopen2(fileId, subGroupTree, H5P_DEFAULT);
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

    static void writeDataExistingFile(const char * const aAprTypeName, hid_t aObjectId, void *aDest,uint64_t elements_start,uint64_t elements_end) {
        //reads partial dataset
        hdf5_write_data_blosc_partial(aObjectId, aDest, aAprTypeName,elements_start,elements_end);
    }



    template<typename T>
    static void writeData(const AprType &aType, hid_t aObjectId, T &aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_blosc(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);
    }

    template<typename T>
    static uint64_t writeDataAppend(const AprType &aType, hid_t aObjectId, T &aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {


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
