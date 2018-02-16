//
// Created by cheesema on 14.01.18.
//

#ifndef PARTPLAY_APRWRITER_HPP
#define PARTPLAY_APRWRITER_HPP

#include "hdf5functions_blosc.h"
#include "src/data_structures/APR/APR.hpp"
#include <src/data_structures/APR/APRAccess.hpp>
#include <numeric>
#include "ConfigAPR.h"
#include <memory>
#include <src/data_structures/APR/APR.hpp>

template<typename U>
class APR;

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
    const AprType ParticleIntensitiesDataType = {H5T_NATIVE_INT64, "data_type"};
    const AprType MapGlobalIndexType = {H5T_NATIVE_INT16, "map_global_index"};
    const AprType MapYendType = {H5T_NATIVE_INT16, "map_y_end"};
    const AprType MapYbeginType = {H5T_NATIVE_INT16, "map_y_begin"};
    const AprType MapNumberGapsType = {H5T_NATIVE_INT16, "map_number_gaps"};
    const AprType MapLevelType = {H5T_NATIVE_UINT8, "map_level"};
    const AprType MapXType = {H5T_NATIVE_INT16, "map_x"};
    const AprType MapZType = {H5T_NATIVE_INT16, "map_z"};
    const AprType ParticleCellType = {H5T_NATIVE_UINT8, "particle_cell_type"};

    const char * const ParticleIntensitiesType = "particle_intensities"; // type read from file
}


class APRWriter {
public:

    template<typename T> struct Hdf5Type {static hid_t type() {return  T::CANNOT_DETECT_TYPE_AND_WILL_NOT_COMPILE;}};

    void readAttr(const AprType &aType, hid_t aGroupId, void *aDest) {
        hid_t attr_id = H5Aopen(aGroupId, aType.typeName, H5P_DEFAULT);
        H5Aread(attr_id, aType.hdf5type, aDest);
        H5Aclose(attr_id);
    }

    void readAttr(const AprType &aType, int aSuffix, hid_t aGroupId, void *aDest) {
        std::string typeNameWithPrefix = std::string(aType.typeName) + std::to_string(aSuffix);
        hid_t attr_id = H5Aopen(aGroupId, typeNameWithPrefix.c_str(), H5P_DEFAULT);
        H5Aread(attr_id, aType.hdf5type, aDest);
        H5Aclose(attr_id);
    }

    void writeAttr(const AprType &aType, hid_t aGroupId, const void * const aSrc) {
        hsize_t dims[] = {1};
        hdf5_write_attribute_blosc(aGroupId, aType.hdf5type, aType.typeName, 1, dims, aSrc);
    }

    void writeAttr(const AprType &aType, int aSuffix, hid_t aGroupId, const void * const aSrc) {
        std::string typeNameWithPrefix = std::string(aType.typeName) + std::to_string(aSuffix);
        hsize_t dims[] = {1};
        hdf5_write_attribute_blosc(aGroupId, aType.hdf5type, typeNameWithPrefix.c_str(), 1, dims, aSrc);
    }

    void readData(const AprType &aType, hid_t aObjectId, void *aDest) {
        hdf5_load_data_blosc(aObjectId, aType.hdf5type, aDest, aType.typeName);
    }

    template<typename T>
    void writeData(const AprType &aType, hid_t aObjectId, T aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_blosc(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);
    }

    template<typename ImageType>
    void read_apr(APR<ImageType>& apr, const std::string &file_name) {
        // need to register the filters so they work properly
        register_blosc();

        hid_t fileId = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t groupId = H5Gopen2(fileId,"ParticleRepr", H5P_DEFAULT);
        hid_t objectId =  H5Gopen2(fileId, "ParticleRepr/t", H5P_DEFAULT);

        // ------------- read metadata --------------------------
        char string_out[100] = {0};
        hid_t attr_id = H5Aopen(groupId,"name",H5P_DEFAULT);
        hid_t atype = H5Aget_type(attr_id);
        hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);
        H5Aread(attr_id, atype_mem, string_out) ;
        H5Aclose(attr_id);
        apr.name= string_out;

        readAttr(AprTypes::TotalNumberOfParticlesType, groupId, &apr.apr_access.total_number_particles);
        readAttr(AprTypes::TotalNumberOfGapsType, groupId, &apr.apr_access.total_number_gaps);
        readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, groupId, &apr.apr_access.total_number_non_empty_rows);
        uint64_t type_size;
        readAttr(AprTypes::VectorSizeType, groupId, &type_size);
        readAttr(AprTypes::NumberOfYType, groupId, &apr.apr_access.org_dims[0]);
        readAttr(AprTypes::NumberOfXType, groupId, &apr.apr_access.org_dims[1]);
        readAttr(AprTypes::NumberOfZType, groupId, &apr.apr_access.org_dims[2]);
        readAttr(AprTypes::MaxLevelType, groupId, &apr.apr_access.level_max);
        readAttr(AprTypes::MinLevelType, groupId, &apr.apr_access.level_min);
        readAttr(AprTypes::LambdaType, groupId, &apr.parameters.lambda);
        int compress_type;
        readAttr(AprTypes::CompressionType, groupId, &compress_type);
        float quantization_factor;
        readAttr(AprTypes::QuantizationFactorType, groupId, &quantization_factor);
        readAttr(AprTypes::SigmaThType, groupId, &apr.parameters.sigma_th);
        readAttr(AprTypes::SigmaThMaxType, groupId, &apr.parameters.sigma_th_max);
        readAttr(AprTypes::IthType, groupId, &apr.parameters.Ip_th);
        readAttr(AprTypes::DxType, groupId, &apr.parameters.dx);
        readAttr(AprTypes::DyType, groupId, &apr.parameters.dy);
        readAttr(AprTypes::DzType, groupId, &apr.parameters.dz);
        readAttr(AprTypes::PsfXType, groupId, &apr.parameters.psfx);
        readAttr(AprTypes::PsfYType, groupId, &apr.parameters.psfy);
        readAttr(AprTypes::PsfZType, groupId, &apr.parameters.psfz);
        readAttr(AprTypes::RelativeErrorType, groupId, &apr.parameters.rel_error);
        readAttr(AprTypes::BackgroundIntensityEstimateType, groupId, &apr.parameters.background_intensity_estimate);
        readAttr(AprTypes::NoiseSdEstimateType, groupId, &apr.parameters.noise_sd_estimate);

        apr.apr_access.x_num.resize(apr.apr_access.level_max+1);
        apr.apr_access.y_num.resize(apr.apr_access.level_max+1);
        apr.apr_access.z_num.resize(apr.apr_access.level_max+1);

        for (int i = apr.apr_access.level_min;i < apr.apr_access.level_max; i++) {
            int x_num, y_num, z_num;
            //TODO: x_num and other should have HDF5 type uint64?
            readAttr(AprTypes::NumberOfLevelXType, i, groupId, &x_num);
            readAttr(AprTypes::NumberOfLevelYType, i, groupId, &y_num);
            readAttr(AprTypes::NumberOfLevelZType, i, groupId, &z_num);
            apr.apr_access.x_num[i] = x_num;
            apr.apr_access.y_num[i] = y_num;
            apr.apr_access.z_num[i] = z_num;
        }

        // ------------- read data ------------------------------
        hid_t dataType;
        readAttr(AprTypes::ParticleIntensitiesDataType, groupId, &dataType);
        apr.particles_intensities.data.resize(apr.apr_access.total_number_particles);
        if (apr.particles_intensities.data.size() > 0) {
            readData({dataType, AprTypes::ParticleIntensitiesType}, objectId, apr.particles_intensities.data.data());
        }
        apr.apr_access.y_num[apr.apr_access.level_max] = apr.apr_access.org_dims[0];
        apr.apr_access.x_num[apr.apr_access.level_max] = apr.apr_access.org_dims[1];
        apr.apr_access.z_num[apr.apr_access.level_max] = apr.apr_access.org_dims[2];

        // ------------- map handling ----------------------------
        auto map_data = std::make_shared<MapStorageData>();

        map_data->global_index.resize(apr.apr_access.total_number_gaps);

        std::vector<int16_t> index_delta(apr.apr_access.total_number_gaps);
        readData(AprTypes::MapGlobalIndexType, objectId, index_delta.data());
        std::vector<uint64_t> index_delta_big(apr.apr_access.total_number_gaps);
        std::copy(index_delta.begin(),index_delta.end(),index_delta_big.begin());
        std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data->global_index.begin());

        map_data->y_end.resize(apr.apr_access.total_number_gaps);
        readData(AprTypes::MapYendType, objectId, map_data->y_end.data());
        map_data->y_begin.resize(apr.apr_access.total_number_gaps);
        readData(AprTypes::MapYbeginType, objectId, map_data->y_begin.data());
        map_data->number_gaps.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapNumberGapsType, objectId, map_data->number_gaps.data());
        map_data->level.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapLevelType, objectId, map_data->level.data());
        map_data->x.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapXType, objectId, map_data->x.data());
        map_data->z.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapZType, objectId, map_data->z.data());
        apr.apr_access.particle_cell_type.data.resize(type_size);
        readData(AprTypes::ParticleCellType, objectId, apr.apr_access.particle_cell_type.data.data());

        apr.apr_access.rebuild_map(apr, *map_data);

        H5Gclose(objectId);
        H5Gclose(groupId);
        H5Fclose(fileId);

        //  Decompress if needed
        if (compress_type > 0) {
            APRCompress<ImageType> apr_compress;
            apr_compress.set_compression_type(compress_type);
            apr_compress.set_quantization_factor(quantization_factor);
            apr_compress.decompress(apr, apr.particles_intensities);
        }
    }

    template<typename ImageType>
    void write_apr(APR<ImageType>& apr,std::string save_loc,std::string file_name){
        APRCompress<ImageType> apr_compressor;
        apr_compressor.set_compression_type(0);
        write_apr(apr,save_loc,file_name,apr_compressor);
    }

    /**
     * Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
     * Bevan Cheeseman 2018
     */
    template<typename ImageType>
    float write_apr(APR<ImageType> &apr, const std::string &save_loc, const std::string &file_name, APRCompress<ImageType> &apr_compressor, unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 2, unsigned int blosc_shuffle=1) {
        int compress_type_num = apr_compressor.get_compression_type();
        float quantization_factor = apr_compressor.get_quantization_factor();

        APRTimer write_timer;
        write_timer.verbose_flag = false;

        //Neighbour Routine Checking
        register_blosc();

        std::string file_name_sufix = file_name + "_apr";
        std::string hdf5_file_name = save_loc + file_name_sufix + ".h5";
        hid_t fid = hdf5_create_file_blosc(hdf5_file_name);

        //////////////////////////////////////////////////////////////////
        //  Write meta-data to the file
        ///////////////////////////////////////////////////////////////////////
        hid_t pr_groupid = H5Gcreate2(fid, "ParticleRepr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        writeAttr(AprTypes::NumberOfXType, pr_groupid, &apr.apr_access.org_dims[1]);
        writeAttr(AprTypes::NumberOfYType, pr_groupid, &apr.apr_access.org_dims[0]);
        writeAttr(AprTypes::NumberOfZType, pr_groupid, &apr.apr_access.org_dims[2]);
        writeAttr(AprTypes::TotalNumberOfParticlesType, pr_groupid, &apr.apr_access.total_number_particles);
        writeAttr(AprTypes::TotalNumberOfGapsType, pr_groupid, &apr.apr_access.total_number_gaps);
        writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, pr_groupid, &apr.apr_access.total_number_non_empty_rows);
        uint64_t type_vector_size = apr.apr_access.particle_cell_type.data.size();
        writeAttr(AprTypes::VectorSizeType, pr_groupid, &type_vector_size);

        hdf5_write_string_blosc(pr_groupid,"name", (apr.name.size() == 0) ? "no_name" : apr.name);
        hdf5_write_string_blosc(pr_groupid,"githash", ConfigAPR::APR_GIT_HASH);
        writeAttr(AprTypes::CompressionType, pr_groupid, &compress_type_num);
        writeAttr(AprTypes::QuantizationFactorType, pr_groupid, &quantization_factor);
        writeAttr(AprTypes::LambdaType, pr_groupid, &apr.parameters.lambda);
        writeAttr(AprTypes::SigmaThType, pr_groupid, &apr.parameters.sigma_th);
        writeAttr(AprTypes::SigmaThMaxType, pr_groupid, &apr.parameters.sigma_th_max);
        writeAttr(AprTypes::IthType, pr_groupid, &apr.parameters.Ip_th);
        writeAttr(AprTypes::DxType, pr_groupid, &apr.parameters.dx);
        writeAttr(AprTypes::DyType, pr_groupid, &apr.parameters.dy);
        writeAttr(AprTypes::DzType, pr_groupid, &apr.parameters.dz);
        writeAttr(AprTypes::PsfXType, pr_groupid, &apr.parameters.psfx);
        writeAttr(AprTypes::PsfYType, pr_groupid, &apr.parameters.psfy);
        writeAttr(AprTypes::PsfZType, pr_groupid, &apr.parameters.psfz);
        writeAttr(AprTypes::RelativeErrorType, pr_groupid, &apr.parameters.rel_error);
        writeAttr(AprTypes::NoiseSdEstimateType, pr_groupid, &apr.parameters.noise_sd_estimate);
        writeAttr(AprTypes::BackgroundIntensityEstimateType, pr_groupid, &apr.parameters.background_intensity_estimate);

        //////////////////////////////////////////////////////////////////
        //  Write data to the file
        ///////////////////////////////////////////////////////////////////////
        hid_t obj_id = H5Gcreate2(fid, "ParticleRepr/t", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        write_timer.start_timer("intensities");
        if (compress_type_num > 0){
            apr_compressor.compress(apr,apr.particles_intensities);
        }
        hid_t type = Hdf5Type<ImageType>::type();
        writeAttr(AprTypes::ParticleIntensitiesDataType, pr_groupid, &type);
        writeData({type, AprTypes::ParticleIntensitiesType}, obj_id, apr.particles_intensities.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        write_timer.stop_timer();

        write_timer.start_timer("access_data");
        MapStorageData map_data;
        apr.apr_access.flatten_structure(apr, map_data);

        blosc_comp_level = 3;
        blosc_shuffle = 1;
        blosc_comp_type = BLOSC_ZSTD;

        std::vector<uint16_t> index_delta;
        index_delta.resize(map_data.global_index.size());
        std::adjacent_difference(map_data.global_index.begin(),map_data.global_index.end(),index_delta.begin());
        writeData(AprTypes::MapGlobalIndexType, obj_id, index_delta, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        writeData(AprTypes::MapYendType, obj_id, map_data.y_end, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapYbeginType, obj_id, map_data.y_begin, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapNumberGapsType, obj_id, map_data.number_gaps, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapLevelType, obj_id, map_data.level, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapXType, obj_id, map_data.x, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::MapZType, obj_id, map_data.z, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::ParticleCellType, obj_id, apr.apr_access.particle_cell_type.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        write_timer.stop_timer();

        for (int i = apr.level_min(); i <apr.level_max() ; ++i) {
            int x_num = apr.apr_access.x_num[i];
            writeAttr(AprTypes::NumberOfLevelXType, i, pr_groupid, &x_num);
            int y_num = apr.apr_access.y_num[i];
            writeAttr(AprTypes::NumberOfLevelYType, i, pr_groupid, &y_num);
            int z_num = apr.apr_access.z_num[i];
            writeAttr(AprTypes::NumberOfLevelZType, i, pr_groupid, &z_num);
        }
        writeAttr(AprTypes::MaxLevelType, pr_groupid, &apr.apr_access.level_max);
        writeAttr(AprTypes::MinLevelType, pr_groupid, &apr.apr_access.level_min);

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        // output the file size
        hsize_t file_size;
        H5Fget_filesize(fid, &file_size);
        double sizeMB = file_size * 1.0 / 1000000.0;
        std::cout << "HDF5 Filesize: " << sizeMB << " MB\n" << "Writing Complete" << std::endl;

        return sizeMB;
    }

    template<typename ImageType,typename T>
    void write_apr_paraview(APR<ImageType>& apr,std::string save_loc,std::string file_name,ExtraParticleData<T>& parts){
        //  Bevan Cheeseman 2018

        unsigned int blosc_comp_type = BLOSC_ZSTD;
        unsigned int blosc_comp_level = 1;
        unsigned int blosc_shuffle = 2;

        register_blosc();

        std::string hdf5_file_name = save_loc + file_name + "_paraview.h5";

        file_name = file_name + "_paraview";

        hid_t fid = hdf5_create_file_blosc(hdf5_file_name);

        hsize_t rank = 1;
        hsize_t dim_a=1;

        //////////////////////////////////////////////////////////////////
        //  Write meta-data to the file
        ///////////////////////////////////////////////////////////////////////
        hsize_t dims = 1;

        hid_t pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        hid_t obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

        hsize_t dims_out[2];
        dims_out[0] = 1;
        dims_out[1] = 1;

        writeAttr(AprTypes::TotalNumberOfParticlesType, pr_groupid, &apr.apr_access.total_number_particles);

        // New parameter and background data
        if (apr.name.size() == 0) { apr.name = "no_name"; }

        hdf5_write_string_blosc(pr_groupid,"name",apr.name);

        hdf5_write_string_blosc(pr_groupid,"githash", ConfigAPR::APR_GIT_HASH);

        //////////////////////////////////////////////////////////////////
        //
        //  Write data to the file
        ///////////////////////////////////////////////////////////////////////

        //write the parts
        std::string name = "particle property";

        hid_t data_type =  Hdf5Type<T>::type();

        APRIterator<ImageType> apr_iterator(apr);

        dims = apr_iterator.total_number_particles();
        hdf5_write_data_blosc(obj_id, data_type, name.c_str(), rank, &dims, parts.data.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        std::vector<uint16_t> xv;
        xv.resize(apr_iterator.total_number_particles());

        std::vector<uint16_t> yv;
        yv.resize(apr_iterator.total_number_particles());

        std::vector<uint16_t> zv;
        zv.resize(apr_iterator.total_number_particles());

        std::vector<uint8_t> levelv;
        levelv.resize(apr_iterator.total_number_particles());

        std::vector<uint8_t> typev;
        typev.resize(apr_iterator.total_number_particles());

        uint64_t particle_number = 0;
#ifdef HAVE_OPENMP
	#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number= 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            xv[particle_number] = apr_iterator.x_global();
            yv[particle_number] = apr_iterator.y_global();
            zv[particle_number] = apr_iterator.z_global();
            levelv[particle_number] = apr_iterator.level();
            typev[particle_number] = apr_iterator.type();
        }

        name = "x";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, xv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "y";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, yv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "z";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT16, name.c_str(), rank, &dims, zv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "level";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT8, name.c_str(), rank, &dims, levelv.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        name = "type";
        hdf5_write_data_blosc(obj_id, H5T_NATIVE_UINT8, name.c_str(), rank, &dims, typev.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"level_max",1,&dim_a, &apr.apr_access.level_max );
        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT,"level_min",1,&dim_a, &apr.apr_access.level_min );

        // output the file size
        hsize_t file_size;
        H5Fget_filesize(fid, &file_size);

        std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;

        // #TODO This needs to be able extended to handle more general type, currently it is assuming uint16
        write_main_paraview_xdmf_xml(save_loc,file_name,apr_iterator.total_number_particles());

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        std::cout << "Writing Complete" << std::endl;
    }

    template<typename S>
    float write_particles_only(std::string save_loc,std::string file_name,ExtraParticleData<S>& parts_extra){
        //  Bevan Cheeseman 2018
        //
        //  Writes only the particle data, requires the same APR to be read in correctly.

        register_blosc();

        std::string hdf5_file_name = save_loc + file_name + "_apr_extra_parts.h5";

        file_name = file_name + "_apr_extra_parts";

        hid_t fid = hdf5_create_file_blosc(hdf5_file_name);

        //hdf5 inits
        hid_t pr_groupid, obj_id;
        H5G_info_t info;
        hsize_t     dims_out[2];
        hsize_t rank = 1;
        hsize_t dims;

        //Get the group you want to open

        //////////////////////////////////////////////////////////////////
        //  Write meta-data to the file
        ///////////////////////////////////////////////////////////////////////
        dims = 1;

        pr_groupid = H5Gcreate2(fid,"ParticleRepr",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        H5Gget_info( pr_groupid, &info );

        obj_id = H5Gcreate2(fid,"ParticleRepr/t",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

        dims_out[0] = 1;
        dims_out[1] = 1;

        //just an identifier in here for the reading of the parts
        hid_t type_id = Hdf5Type<S>::type();

        hdf5_write_attribute_blosc(pr_groupid,H5T_NATIVE_INT64,"data_type",1,dims_out, &type_id);

        uint64_t total_number_parts = parts_extra.data.size();

        writeAttr(AprTypes::TotalNumberOfParticlesType, pr_groupid, &total_number_parts);

        //////////////////////////////////////////////////////////////////
        //  Write data to the file
        ///////////////////////////////////////////////////////////////////////

        unsigned int blosc_comp_type = BLOSC_ZSTD;
        unsigned int blosc_comp_level = 3;
        unsigned int blosc_shuffle = 2;

        dims = total_number_parts;
        std::string name = "extra_particle_data";
        hdf5_write_data_blosc(obj_id, type_id, name.c_str(), rank, &dims, parts_extra.data.data(),blosc_comp_type,blosc_comp_level,blosc_shuffle);


        // New parameter and background data
        hdf5_write_string_blosc(pr_groupid,"githash", ConfigAPR::APR_GIT_HASH);

        // output the file size
        hsize_t file_size;
        H5Fget_filesize(fid, &file_size);

        std::cout << "HDF5 Filesize: " << file_size*1.0/1000000.0 << " MB" << std::endl;

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);

        std::cout << "Writing ExtraPartCellData Complete" << std::endl;

        return file_size*1.0/1000000.0; //returns file size in MB
    }

    template<typename T>
    void read_parts_only(std::string file_name,ExtraParticleData<T>& extra_parts) {
        std::cout << "READING [" << file_name << "]\n";

        //hdf5 inits
        hid_t fid, pr_groupid, obj_id,attr_id;
        H5G_info_t info;

        //need to register the filters so they work properly
        register_blosc();

        fid = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);

        //Get the group you want to open

        pr_groupid = H5Gopen2(fid,"ParticleRepr",H5P_DEFAULT);
        H5Gget_info( pr_groupid, &info );

        //Getting an attribute
        obj_id =  H5Oopen_by_idx( fid, "ParticleRepr", H5_INDEX_NAME, H5_ITER_INC,0,H5P_DEFAULT);

        /////////////////////////////////////////////
        //  Get metadata
        //////////////////////////////////////////////
        uint64_t total_number_parts;
        readAttr(AprTypes::TotalNumberOfParticlesType, pr_groupid, &total_number_parts);

        H5Screate(H5S_SCALAR);

        hid_t data_type;
        attr_id = H5Aopen(pr_groupid,"data_type",H5P_DEFAULT);
        H5Aread(attr_id,H5T_NATIVE_INT64, &data_type ) ;
        H5Aclose(attr_id);

        extra_parts.data.resize(total_number_parts);
        std::string dataset_name = "extra_particle_data";
        hdf5_load_data_blosc(obj_id, data_type,extra_parts.data.data(),dataset_name.c_str());

        //close shiz
        H5Gclose(obj_id);
        H5Gclose(pr_groupid);
        H5Fclose(fid);
    }
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

#endif //PARTPLAY_APRWRITER_HPP
