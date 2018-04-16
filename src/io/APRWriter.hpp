//
// Created by cheesema on 14.01.18.
//

#ifndef APRWRITER_HPP
#define APRWRITER_HPP

#include "hdf5functions_blosc.h"
#include "../data_structures/APR/APR.hpp"
#include "../data_structures/APR/APRAccess.hpp"
#include "ConfigAPR.h"
#include <numeric>
#include <memory>


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
public:

    template<typename ImageType>
    void read_apr(APR<ImageType>& apr, const std::string &file_name) {
        AprFile f(file_name, AprFile::Operation::READ);
        if (!f.isOpened()) return;

        // ------------- read metadata --------------------------
        char string_out[100] = {0};
        hid_t attr_id = H5Aopen(f.groupId,"name",H5P_DEFAULT);
        hid_t atype = H5Aget_type(attr_id);
        hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);
        H5Aread(attr_id, atype_mem, string_out) ;
        H5Aclose(attr_id);
        apr.name= string_out;

        readAttr(AprTypes::TotalNumberOfParticlesType, f.groupId, &apr.apr_access.total_number_particles);
        readAttr(AprTypes::TotalNumberOfGapsType, f.groupId, &apr.apr_access.total_number_gaps);
        readAttr(AprTypes::TotalNumberOfNonEmptyRowsType, f.groupId, &apr.apr_access.total_number_non_empty_rows);
        uint64_t type_size;
        readAttr(AprTypes::VectorSizeType, f.groupId, &type_size);
        readAttr(AprTypes::NumberOfYType, f.groupId, &apr.apr_access.org_dims[0]);
        readAttr(AprTypes::NumberOfXType, f.groupId, &apr.apr_access.org_dims[1]);
        readAttr(AprTypes::NumberOfZType, f.groupId, &apr.apr_access.org_dims[2]);
        readAttr(AprTypes::MaxLevelType, f.groupId, &apr.apr_access.level_max);
        readAttr(AprTypes::MinLevelType, f.groupId, &apr.apr_access.level_min);
        readAttr(AprTypes::LambdaType, f.groupId, &apr.parameters.lambda);
        int compress_type;
        readAttr(AprTypes::CompressionType, f.groupId, &compress_type);
        float quantization_factor;
        readAttr(AprTypes::QuantizationFactorType, f.groupId, &quantization_factor);
        readAttr(AprTypes::SigmaThType, f.groupId, &apr.parameters.sigma_th);
        readAttr(AprTypes::SigmaThMaxType, f.groupId, &apr.parameters.sigma_th_max);
        readAttr(AprTypes::IthType, f.groupId, &apr.parameters.Ip_th);
        readAttr(AprTypes::DxType, f.groupId, &apr.parameters.dx);
        readAttr(AprTypes::DyType, f.groupId, &apr.parameters.dy);
        readAttr(AprTypes::DzType, f.groupId, &apr.parameters.dz);
        readAttr(AprTypes::PsfXType, f.groupId, &apr.parameters.psfx);
        readAttr(AprTypes::PsfYType, f.groupId, &apr.parameters.psfy);
        readAttr(AprTypes::PsfZType, f.groupId, &apr.parameters.psfz);
        readAttr(AprTypes::RelativeErrorType, f.groupId, &apr.parameters.rel_error);
        readAttr(AprTypes::BackgroundIntensityEstimateType, f.groupId, &apr.parameters.background_intensity_estimate);
        readAttr(AprTypes::NoiseSdEstimateType, f.groupId, &apr.parameters.noise_sd_estimate);

        apr.apr_access.x_num.resize(apr.apr_access.level_max+1);
        apr.apr_access.y_num.resize(apr.apr_access.level_max+1);
        apr.apr_access.z_num.resize(apr.apr_access.level_max+1);

        for (size_t i = apr.apr_access.level_min;i < apr.apr_access.level_max; i++) {
            int x_num, y_num, z_num;
            //TODO: x_num and other should have HDF5 type uint64?
            readAttr(AprTypes::NumberOfLevelXType, i, f.groupId, &x_num);
            readAttr(AprTypes::NumberOfLevelYType, i, f.groupId, &y_num);
            readAttr(AprTypes::NumberOfLevelZType, i, f.groupId, &z_num);
            apr.apr_access.x_num[i] = x_num;
            apr.apr_access.y_num[i] = y_num;
            apr.apr_access.z_num[i] = z_num;
        }

        // ------------- read data ------------------------------
        apr.particles_intensities.data.resize(apr.apr_access.total_number_particles);
        if (apr.particles_intensities.data.size() > 0) {
            readData(AprTypes::ParticleIntensitiesType, f.objectId, apr.particles_intensities.data.data());
        }
        apr.apr_access.y_num[apr.apr_access.level_max] = apr.apr_access.org_dims[0];
        apr.apr_access.x_num[apr.apr_access.level_max] = apr.apr_access.org_dims[1];
        apr.apr_access.z_num[apr.apr_access.level_max] = apr.apr_access.org_dims[2];

        // ------------- map handling ----------------------------
        auto map_data = std::make_shared<MapStorageData>();

        map_data->global_index.resize(apr.apr_access.total_number_gaps);

        std::vector<int16_t> index_delta(apr.apr_access.total_number_gaps);
        readData(AprTypes::MapGlobalIndexType, f.objectId, index_delta.data());
        std::vector<uint64_t> index_delta_big(apr.apr_access.total_number_gaps);
        std::copy(index_delta.begin(),index_delta.end(),index_delta_big.begin());
        std::partial_sum(index_delta_big.begin(), index_delta_big.end(), map_data->global_index.begin());

        map_data->y_end.resize(apr.apr_access.total_number_gaps);
        readData(AprTypes::MapYendType, f.objectId, map_data->y_end.data());
        map_data->y_begin.resize(apr.apr_access.total_number_gaps);
        readData(AprTypes::MapYbeginType, f.objectId, map_data->y_begin.data());
        map_data->number_gaps.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapNumberGapsType, f.objectId, map_data->number_gaps.data());
        map_data->level.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapLevelType, f.objectId, map_data->level.data());
        map_data->x.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapXType, f.objectId, map_data->x.data());
        map_data->z.resize(apr.apr_access.total_number_non_empty_rows);
        readData(AprTypes::MapZType, f.objectId, map_data->z.data());
        apr.apr_access.particle_cell_type.data.resize(type_size);
        readData(AprTypes::ParticleCellType, f.objectId, apr.apr_access.particle_cell_type.data.data());

        apr.apr_access.rebuild_map(apr, *map_data);

        // ------------ decompress if needed ---------------------
        if (compress_type > 0) {
            APRCompress<ImageType> apr_compress;
            apr_compress.set_compression_type(compress_type);
            apr_compress.set_quantization_factor(quantization_factor);
            apr_compress.decompress(apr, apr.particles_intensities);
        }
    }

    template<typename ImageType>
    void write_apr(APR<ImageType>& apr, const std::string &save_loc, const std::string &file_name) {
        APRCompress<ImageType> apr_compressor;
        apr_compressor.set_compression_type(0);
        write_apr(apr, save_loc, file_name, apr_compressor);
    }

    /**
     * Writes the APR to the particle cell structure sparse format, using the p_map for reconstruction
     */
    template<typename ImageType>
    float write_apr(APR<ImageType> &apr, const std::string &save_loc, const std::string &file_name, APRCompress<ImageType> &apr_compressor, unsigned int blosc_comp_type = BLOSC_ZSTD, unsigned int blosc_comp_level = 2, unsigned int blosc_shuffle=1) {
        APRTimer write_timer;
        write_timer.verbose_flag = false;

        std::string hdf5_file_name = save_loc + file_name + "_apr.h5";
        AprFile f{hdf5_file_name, AprFile::Operation::WRITE};
        if (!f.isOpened()) return 0;

        // ------------- write metadata -------------------------
        writeAttr(AprTypes::NumberOfXType, f.groupId, &apr.apr_access.org_dims[1]);
        writeAttr(AprTypes::NumberOfYType, f.groupId, &apr.apr_access.org_dims[0]);
        writeAttr(AprTypes::NumberOfZType, f.groupId, &apr.apr_access.org_dims[2]);
        writeAttr(AprTypes::TotalNumberOfGapsType, f.groupId, &apr.apr_access.total_number_gaps);
        writeAttr(AprTypes::TotalNumberOfNonEmptyRowsType, f.groupId, &apr.apr_access.total_number_non_empty_rows);
        uint64_t type_vector_size = apr.apr_access.particle_cell_type.data.size();
        writeAttr(AprTypes::VectorSizeType, f.groupId, &type_vector_size);

        writeString(AprTypes::NameType, f.groupId, (apr.name.size() == 0) ? "no_name" : apr.name);
        writeString(AprTypes::GitType, f.groupId, ConfigAPR::APR_GIT_HASH);
        writeAttr(AprTypes::TotalNumberOfParticlesType, f.groupId, &apr.apr_access.total_number_particles);
        writeAttr(AprTypes::MaxLevelType, f.groupId, &apr.apr_access.level_max);
        writeAttr(AprTypes::MinLevelType, f.groupId, &apr.apr_access.level_min);

        int compress_type_num = apr_compressor.get_compression_type();
        writeAttr(AprTypes::CompressionType, f.groupId, &compress_type_num);
        float quantization_factor = apr_compressor.get_quantization_factor();
        writeAttr(AprTypes::QuantizationFactorType, f.groupId, &quantization_factor);
        writeAttr(AprTypes::LambdaType, f.groupId, &apr.parameters.lambda);
        writeAttr(AprTypes::SigmaThType, f.groupId, &apr.parameters.sigma_th);
        writeAttr(AprTypes::SigmaThMaxType, f.groupId, &apr.parameters.sigma_th_max);
        writeAttr(AprTypes::IthType, f.groupId, &apr.parameters.Ip_th);
        writeAttr(AprTypes::DxType, f.groupId, &apr.parameters.dx);
        writeAttr(AprTypes::DyType, f.groupId, &apr.parameters.dy);
        writeAttr(AprTypes::DzType, f.groupId, &apr.parameters.dz);
        writeAttr(AprTypes::PsfXType, f.groupId, &apr.parameters.psfx);
        writeAttr(AprTypes::PsfYType, f.groupId, &apr.parameters.psfy);
        writeAttr(AprTypes::PsfZType, f.groupId, &apr.parameters.psfz);
        writeAttr(AprTypes::RelativeErrorType, f.groupId, &apr.parameters.rel_error);
        writeAttr(AprTypes::NoiseSdEstimateType, f.groupId, &apr.parameters.noise_sd_estimate);
        writeAttr(AprTypes::BackgroundIntensityEstimateType, f.groupId, &apr.parameters.background_intensity_estimate);

        // ------------- write data ----------------------------
        write_timer.start_timer("intensities");
        if (compress_type_num > 0){
            apr_compressor.compress(apr,apr.particles_intensities);
        }
        hid_t type = Hdf5Type<ImageType>::type();
        writeData({type, AprTypes::ParticleIntensitiesType}, f.objectId, apr.particles_intensities.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        write_timer.stop_timer();

        write_timer.start_timer("access_data");
        MapStorageData map_data;
        apr.apr_access.flatten_structure(apr, map_data);

        // TODO: why those values are overwrite?
        blosc_comp_level = 3;
        blosc_shuffle = 1;
        blosc_comp_type = BLOSC_ZSTD;

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
        writeData(AprTypes::ParticleCellType, f.objectId, apr.apr_access.particle_cell_type.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        write_timer.stop_timer();

        for (size_t i = apr.level_min(); i <apr.level_max() ; ++i) {
            int x_num = apr.apr_access.x_num[i];
            writeAttr(AprTypes::NumberOfLevelXType, i, f.groupId, &x_num);
            int y_num = apr.apr_access.y_num[i];
            writeAttr(AprTypes::NumberOfLevelYType, i, f.groupId, &y_num);
            int z_num = apr.apr_access.z_num[i];
            writeAttr(AprTypes::NumberOfLevelZType, i, f.groupId, &z_num);
        }

        // ------------- output the file size -------------------
        hsize_t file_size = f.getFileSize();
        double sizeMB = file_size / 1e6;
        std::cout << "HDF5 Filesize: " << sizeMB << " MB\n" << "Writing Complete" << std::endl;
        return sizeMB;
    }

    template<typename ImageType,typename T>
    void write_apr_paraview(APR<ImageType> &apr, const std::string &save_loc, const std::string &file_name, const ExtraParticleData<T> &parts) {
        std::string hdf5_file_name = save_loc + file_name + "_paraview.h5";
        AprFile f{hdf5_file_name, AprFile::Operation::WRITE};
        if (!f.isOpened()) return;

        // ------------- write metadata -------------------------

        writeString(AprTypes::NameType, f.groupId, (apr.name.size() == 0) ? "no_name" : apr.name);
        writeString(AprTypes::GitType, f.groupId, ConfigAPR::APR_GIT_HASH);
        writeAttr(AprTypes::MaxLevelType, f.groupId, &apr.apr_access.level_max);
        writeAttr(AprTypes::MinLevelType, f.groupId, &apr.apr_access.level_min);
        writeAttr(AprTypes::TotalNumberOfParticlesType, f.groupId, &apr.apr_access.total_number_particles);

        // ------------- write data ----------------------------
        unsigned int blosc_comp_level = 1;
        unsigned int blosc_shuffle = 2;
        unsigned int blosc_comp_type = BLOSC_ZSTD;
        writeData({(Hdf5Type<T>::type()), AprTypes::ParticlePropertyType}, f.objectId, parts.data, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        APRIterator<ImageType> apr_iterator(apr);
        std::vector<uint16_t> xv(apr_iterator.total_number_particles());
        std::vector<uint16_t> yv(apr_iterator.total_number_particles());
        std::vector<uint16_t> zv(apr_iterator.total_number_particles());
        std::vector<uint8_t> levelv(apr_iterator.total_number_particles());
        std::vector<uint8_t> typev(apr_iterator.total_number_particles());

        #ifdef HAVE_OPENMP
	    #pragma omp parallel for schedule(static) firstprivate(apr_iterator)
        #endif
        for (uint64_t particle_number= 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            xv[particle_number] = apr_iterator.x_global();
            yv[particle_number] = apr_iterator.y_global();
            zv[particle_number] = apr_iterator.z_global();
            levelv[particle_number] = apr_iterator.level();
            typev[particle_number] = apr_iterator.type();
        }
        writeData(AprTypes::ParaviewXType, f.objectId, xv, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::ParaviewYType, f.objectId, yv, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::ParaviewZType, f.objectId, zv, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::ParaviewLevelType, f.objectId, levelv, blosc_comp_type, blosc_comp_level, blosc_shuffle);
        writeData(AprTypes::ParaviewTypeType, f.objectId, typev, blosc_comp_type, blosc_comp_level, blosc_shuffle);

        // TODO: This needs to be able extended to handle more general type, currently it is assuming uint16
        write_main_paraview_xdmf_xml(save_loc,file_name,apr_iterator.total_number_particles());

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
        unsigned int blosc_comp_type = BLOSC_ZSTD;
        unsigned int blosc_comp_level = 3;
        unsigned int blosc_shuffle = 2;
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

    struct AprFile {
        enum class Operation {READ, WRITE};
        hid_t fileId = -1;
        hid_t groupId = -1;
        hid_t objectId = -1;
        const char * const mainGroup = "ParticleRepr";
        const char * const subGroup  = "ParticleRepr/t";

        AprFile(const std::string &aFileName, const Operation aOp) {
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
                case Operation::WRITE:
                    fileId = hdf5_create_file_blosc(aFileName);
                    if (fileId == -1) {
                        std::cerr << "Could not create file [" << aFileName << "]" << std::endl;
                        return;
                    }
                    groupId = H5Gcreate2(fileId, mainGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    objectId = H5Gcreate2(fileId, subGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    break;
            }
            if (groupId == -1 || objectId == -1) { H5Fclose(fileId); fileId = -1; }
        }
        ~AprFile() {
            if (objectId != -1) H5Gclose(objectId);
            if (groupId != -1) H5Gclose(groupId);
            if (fileId != -1) H5Fclose(fileId);
        }

        /**
         * Is File opened?
         */
        bool isOpened() const { return fileId != -1 && groupId != -1 && objectId != -1; }

        hsize_t getFileSize() const {
            hsize_t size;
            H5Fget_filesize(fileId, &size);
            return size;
        }
    };

private:
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

    template<typename T>
    void writeData(const AprType &aType, hid_t aObjectId, T aContainer, unsigned int blosc_comp_type, unsigned int blosc_comp_level,unsigned int blosc_shuffle) {
        hsize_t dims[] = {aContainer.size()};
        const hsize_t rank = 1;
        hdf5_write_data_blosc(aObjectId, aType.hdf5type, aType.typeName, rank, dims, aContainer.data(), blosc_comp_type, blosc_comp_level, blosc_shuffle);
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
