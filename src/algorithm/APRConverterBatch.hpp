////////////////////////////////
///
/// Created by Bevan Cheeseman 2018
/// Adapted by Joel Jonsson 2020
///
/// APR Converter class handles the methods and functions for creating an APR from large input images, by
/// performing the pixel computations on (potentially overlapping) tiles.
///
////////////////////////////////

#ifndef LIBAPR_APRCONVERTERBATCH_HPP
#define LIBAPR_APRCONVERTERBATCH_HPP

#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/Mesh/ImagePatch.hpp"

#include "io/TiffUtils.hpp"
#include "data_structures/APR/APR.hpp"

#include "APRConverter.hpp"

#include "ComputeGradient.hpp"
#include "LocalIntensityScale.hpp"
#include "LocalParticleCellSet.hpp"
#include "PullingSchemeSparse.hpp"

#ifdef APR_USE_CUDA
#include "algorithm/ComputeGradientCuda.hpp"
#endif

template<typename ImageType>
class APRConverterBatch: public LocalIntensityScale, public ComputeGradient, public LocalParticleCellSet, public PullingSchemeSparse {

public:

    int z_block_size = 128;
    int ghost_z = 32;       //TODO: figure out how the solution depends on this parameter, with and without B-spline smoothing
    bool verbose = false;

    APRParameters par;
    APRTimer fine_grained_timer;
    APRTimer method_timer;
    APRTimer total_timer;
    APRTimer allocation_timer;
    APRTimer computation_timer;

    bool get_apr(APR &aAPR) {
        aAPR.parameters = par;

        TiffUtils::TiffInfo inputTiff(par.input_dir + par.input_image_name);
        if (!inputTiff.isFileOpened()) return false;

        if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT8) {
            return get_apr_batch_method_from_file<uint8_t>(aAPR, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_FLOAT) {
            return get_apr_batch_method_from_file<float>(aAPR, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT16) {
            return get_apr_batch_method_from_file<uint16_t>(aAPR, inputTiff);
        } else {
            std::cerr << "Unsupported file type. Input image must be a TIFF of data type uint8, uint16 or float32" << std::endl;
            return false;
        }
    };

    template<typename T>
    bool get_apr_method_patch(APR &aAPR, PixelData<T>& input_image, ImagePatch &patch);

    void set_generate_linear(bool flag){
        generate_linear = flag;
    }

    void set_sparse_pulling_scheme(bool flag){
        sparse_pulling_scheme = flag;
    }

protected:

    PullingScheme iPullingScheme;
    LocalParticleCellSet iLocalParticleSet;
    LocalIntensityScale iLocalIntensityScale;
    ComputeGradient iComputeGradient;
    PullingSchemeSparse iPullingSchemeSparse;

    bool generate_linear = true;        //default is the linear structure
    bool sparse_pulling_scheme = false;

    float bspline_offset = 0;

    void init_apr(APR& aAPR, const TiffUtils::TiffInfo &aTiffFile);

    template<typename T>
    bool get_apr_batch_method_from_file(APR &aAPR, const TiffUtils::TiffInfo &aTiffFile);

    void applyParameters(PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp,
                         PixelData<float> &local_scale_temp2, APRParameters& aprParameters);

    void fill_global_particle_cell_tree(PixelData<float>& local_scale_temp, PixelData<float>& local_scale_temp2, ImagePatch& patch);

    void generateDatastructures(APR& aAPR);
};


/**
 * Main method reading the input TIFF file in tiles (currently split only in z), applying most of the APR pipeline
 * to each tile separately and computes the global solution.
 * @tparam ImageType
 * @tparam T
 * @param aAPR
 * @param aTiffFile
 * @return
 */
template<typename ImageType> template<typename T>
bool APRConverterBatch<ImageType>::get_apr_batch_method_from_file(APR &aAPR, const TiffUtils::TiffInfo &aTiffFile) {

    init_apr(aAPR, aTiffFile);

    const int y_num = aAPR.org_dims(0);
    const int x_num = aAPR.org_dims(1);
    const int z_num = aAPR.org_dims(2);

    if(verbose) {
        std::cout << "Full image size (z, y, x): (" << z_num << ", " << x_num << ", " << y_num << ")" << std::endl;
    }
    const int number_z_blocks = z_num / z_block_size; // last block may be bigger
    std::vector<ImagePatch> patches;
    patches.resize(number_z_blocks);

    method_timer.start_timer("initialize_particle_cell_tree");
    if(sparse_pulling_scheme) {
        iPullingSchemeSparse.initialize_particle_cell_tree(aAPR.aprInfo);
    } else {
        iPullingScheme.initialize_particle_cell_tree(aAPR.aprInfo);
    }
    method_timer.stop_timer();

    for (int i = 0; i < number_z_blocks; ++i) {

        int z_0 = i * z_block_size;
        int z_f = (i == (number_z_blocks-1)) ? z_num : (i+1) * z_block_size;

        int z_ghost_l = std::min(z_0, ghost_z);
        int z_ghost_r = std::min(z_num - z_f, ghost_z);

        initPatchGlobal(patches[i], z_0 - z_ghost_l, z_f + z_ghost_r, 0, x_num, 0, y_num);

        patches[i].z_ghost_l = z_ghost_l;
        patches[i].z_ghost_r = z_ghost_r;
        patches[i].z_offset = z_0 - z_ghost_l;

        //PixelData<T> patchImage(inputImage.y_num, inputImage.x_num, number_slices);
        method_timer.start_timer("load data");
        PixelData<T> patchImage = TiffUtils::getMesh<T>(aTiffFile, patches[i].z_begin_global, patches[i].z_end_global);
        method_timer.stop_timer();

        if(verbose) {
            std::cout << "Patch " << i+1 << " / " << number_z_blocks << " size: " << patchImage.mesh.size() * sizeof(T) * 1e-6 << " MB" << std::endl;
        }

        get_apr_method_patch(aAPR, patchImage, patches[i]);
    }

    method_timer.start_timer("compute_pulling_scheme");
    if(sparse_pulling_scheme) {
        iPullingSchemeSparse.pulling_scheme_main();
    } else {
        iPullingScheme.pulling_scheme_main();
    }
    method_timer.stop_timer();

    method_timer.start_timer("compute_apr_datastructure");
    generateDatastructures(aAPR);
    method_timer.stop_timer();

    return true;
}


/**
 * Performs the pixel image computations of the APR pipeline to an image tile, and offloads the result to the global
 * particle cell tree. After running this step for each tile in the image, the main pulling scheme method computes the
 * global APR solution.
 * @tparam ImageType
 * @tparam T
 * @param aAPR
 * @param input_image   PixelData<T> holding a tile of the full image
 * @param patch         ImagePatch describing the location of the tile within the global image
 * @return
 */
template<typename ImageType> template<typename T>
bool APRConverterBatch<ImageType>::get_apr_method_patch(APR &aAPR, PixelData<T>& input_image, ImagePatch &patch) {

    total_timer.start_timer("Total_pipeline_excluding_IO");

    ////////////////////////////////////////
    /// Memory allocation of variables
    ////////////////////////////////////////

    //assuming uint16, the total memory cost shoudl be approximately (1 + 1 + 1/8 + 2/8 + 2/8) = 2 5/8 original image size in u16bit
    //storage of the particle cell tree for computing the pulling scheme
    allocation_timer.start_timer("initialize pipeline buffers");
    PixelData<ImageType> image_temp(input_image, false /* don't copy */); // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)
    PixelData<ImageType> grad_temp; // should be a down-sampled image
    grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0, false);
    PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
    local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);
    PixelData<float> local_scale_temp2;
    local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);
    allocation_timer.stop_timer();

    /////////////////////////////////
    /// Pipeline
    ////////////////////////

    computation_timer.start_timer("Calculations");

    fine_grained_timer.start_timer("offset image");
    //offset image by factor (this is required if there are zero areas in the background with uint16_t and uint8_t images, as the Bspline co-efficients otherwise may be negative!)
    // Warning both of these could result in over-flow (if your image is non zero, with a 'buffer' and has intensities up to uint16_t maximum value then set image_type = "", i.e. uncomment the following line)
    if (std::is_same<uint16_t, ImageType>::value) {
        bspline_offset = 100;
        image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
    } else if (std::is_same<uint8_t, ImageType>::value){
        bspline_offset = 5;
        image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
    } else {
        image_temp.copyFromMesh(input_image);
    }
    fine_grained_timer.stop_timer();

//#ifndef APR_USE_CUDA
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines");
    iComputeGradient.get_gradient(image_temp, grad_temp, local_scale_temp, par);
    method_timer.stop_timer();
#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "gradient_step.tif", grad_temp);
    }
#endif
    method_timer.start_timer("compute_local_intensity_scale");
    iLocalIntensityScale.get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
    method_timer.stop_timer();
#ifdef HAVE_LIBTIFF
    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_step.tif", local_scale_temp);
    }
#endif

//#else
//    method_timer.start_timer("compute_gradient_magnitude_using_bsplines and local instensity scale CUDA");
//    getFullPipeline(image_temp, grad_temp, local_scale_temp, local_scale_temp2,bspline_offset, par);
//    method_timer.stop_timer();
//#endif

    //TODO: How to do auto parameters? compute from first patch only?
    computation_timer.start_timer("apply_parameters");
    applyParameters(grad_temp, local_scale_temp, local_scale_temp2,par);
    computation_timer.stop_timer();

    method_timer.start_timer("compute_local_particle_set");
    iLocalParticleSet.computeLevels(grad_temp, local_scale_temp, aAPR.level_max(), par.rel_error, par.dx, par.dy, par.dz);
    method_timer.stop_timer();

    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_particle_set_level_step.tif", local_scale_temp);
    }

    method_timer.start_timer("fill global particle cell tree");
    fill_global_particle_cell_tree(local_scale_temp, local_scale_temp2, patch);
    method_timer.stop_timer();

    computation_timer.stop_timer();

    total_timer.stop_timer();

    return true;
}


/**
 * Fills the seed particle cells from an image tile into the global particle cell tree
 * @tparam ImageType
 * @param local_scale_temp
 * @param local_scale_temp2
 * @param patch
 */
template<typename ImageType>
void APRConverterBatch<ImageType>::fill_global_particle_cell_tree(PixelData<float>& local_scale_temp, PixelData<float>& local_scale_temp2, ImagePatch& patch) {
    if(sparse_pulling_scheme) {
        const int l_max = iPullingSchemeSparse.pct_level_max();
        const int l_min = iPullingSchemeSparse.pct_level_min();
        iPullingSchemeSparse.fill_patch(l_max, local_scale_temp, patch);

        for (int l_ = l_max - 1; l_ >= l_min; l_--) {
            //down sample the resolution level k, using a max reduction
            downsample(local_scale_temp, local_scale_temp2,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);

            iPullingSchemeSparse.fill_patch(l_, local_scale_temp2, patch);
            //assign the previous mesh to now be downsampled.
            local_scale_temp.swap(local_scale_temp2);
        }
    } else {
        const int l_max = iPullingScheme.pct_level_max();
        const int l_min = iPullingScheme.pct_level_min();
        iPullingScheme.fill_patch(l_max, local_scale_temp, patch);

        for (int l_ = l_max - 1; l_ >= l_min; l_--) {
            //down sample the resolution level k, using a max reduction
            downsample(local_scale_temp, local_scale_temp2,
                       [](const float &x, const float &y) -> float { return std::max(x, y); },
                       [](const float &x) -> float { return x; }, true);

            iPullingScheme.fill_patch(l_, local_scale_temp2, patch);
            //assign the previous mesh to now be downsampled.
            local_scale_temp.swap(local_scale_temp2);
        }
    }
}


/**
 * Initialize the APRAccess data structure from the pulling scheme solution
 * @tparam ImageType
 * @param aAPR
 */
template<typename ImageType>
void APRConverterBatch<ImageType>::generateDatastructures(APR& aAPR){
    if(!generate_linear) {
        if(!sparse_pulling_scheme){
            aAPR.apr_access.initialize_structure_from_particle_cell_tree(aAPR.parameters,
                                                                         iPullingScheme.getParticleCellTree());
        } else{
            aAPR.apr_access.initialize_structure_from_particle_cell_tree_sparse(aAPR.parameters,
                                                                                iPullingSchemeSparse.particle_cell_tree);
        }
        aAPR.apr_initialized_random = true;

    } else {
        if(!sparse_pulling_scheme) {
            aAPR.linearAccess.initialize_linear_structure(aAPR.parameters,
                                                          iPullingScheme.getParticleCellTree());
        } else {
            aAPR.linearAccess.initialize_linear_structure_sparse(aAPR.parameters,
                                                                 iPullingSchemeSparse.particle_cell_tree);
        }
        aAPR.apr_initialized = true;
    }
}


/**
 * Initializes the APR info: maximum level and image dimensions for each level
 * @tparam ImageType
 * @param aAPR
 * @param aTiffFile
 */
template<typename ImageType>
void APRConverterBatch<ImageType>::init_apr(APR& aAPR,const TiffUtils::TiffInfo &aTiffFile){

    aAPR.aprInfo.init(aTiffFile.iImgWidth, aTiffFile.iImgHeight, aTiffFile.iNumberOfDirectories);
    aAPR.linearAccess.genInfo = &aAPR.aprInfo;
    aAPR.apr_access.genInfo = &aAPR.aprInfo;

    aAPR.parameters = par;
}


/**
 * Apply the threshold parameters (Ip_th, sigma_th, grad_th) to local intensity scale and gradient
 * @tparam ImageType
 * @param grad_temp
 * @param local_scale_temp
 * @param local_scale_temp2
 * @param aprParameters
 */
template<typename ImageType>
void APRConverterBatch<ImageType>::applyParameters(PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp,
                                                   PixelData<float> &local_scale_temp2, APRParameters& aprParameters) {

    fine_grained_timer.start_timer("load_and_apply_mask");
    // Apply mask if given
    if(par.mask_file != ""){
        iComputeGradient.mask_gradient(grad_temp, aprParameters);
    }
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("threshold");
    iComputeGradient.threshold_gradient(grad_temp, local_scale_temp2, aprParameters.Ip_th + bspline_offset);
    fine_grained_timer.stop_timer();

    float max_th = 60000;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for (size_t i = 0; i < grad_temp.mesh.size(); ++i) {

        float rescaled = local_scale_temp.mesh[i];
        if (rescaled < aprParameters.sigma_th) {
            rescaled = (rescaled < aprParameters.sigma_th_max) ? max_th : par.sigma_th;
            local_scale_temp.mesh[i] = rescaled;
        }
    }

    if(par.output_steps) {
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_rescaled.tif", local_scale_temp);
    }

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for (size_t i = 0; i < grad_temp.mesh.size(); ++i) {

        if(grad_temp.mesh[i] < aprParameters.grad_th){
            grad_temp.mesh[i] = 0;
        }
    }
}


#endif //LIBAPR_APRCONVERTERBATCH_HPP
