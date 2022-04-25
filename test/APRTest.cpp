//
// Created by cheesema on 21.01.18.
//

#include <gtest/gtest.h>
#include "numerics/APRFilter.hpp"
#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRConverter.hpp"
#include "algorithm/APRConverterBatch.hpp"
#include <utility>
#include <cmath>
#include "TestTools.hpp"
#include "numerics/APRNumerics.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include "io/APRWriter.hpp"
#include "numerics/APRStencil.hpp"
#include "data_structures/APR/particles/LazyData.hpp"

#include "io/APRFile.hpp"

#include "data_structures/APR/access/LazyAccess.hpp"
#include "data_structures/APR/iterators/LazyIterator.hpp"


struct TestData{

    APR apr;
    PixelData<uint16_t> img_level;
    PixelData<uint16_t> img_type;
    PixelData<uint16_t> img_original;
    PixelData<uint16_t> img_pc;
    PixelData<uint16_t> img_x;
    PixelData<uint16_t> img_y;
    PixelData<uint16_t> img_z;

    ParticleData<uint16_t> particles_intensities;

    std::string filename;
    std::string apr_filename;
    std::string output_name;
    std::string output_dir;

};

class CreateAPRTest : public ::testing::Test {
public:

    TestData test_data;

protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

};


class CreateSmallSphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateBigBigData : public CreateAPRTest
{
public:
    void SetUp() override;
};


class CreateDiffDimsSphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class Create210SphereTestAPROnly : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmallTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmall2DTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmall2DTestProperties : public CreateAPRTest
{
public:
    void SetUp() override;
};


class CreateGTSmall1DTestProperties : public CreateAPRTest
{
public:
    void SetUp() override;
};



class CreateGTSmall2DTestAPR : public CreateAPRTest
{
public:
    void SetUp() override;
};



class CreateGTSmall1DTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

template<typename Iterator1,typename Iterator2>
bool compare_two_iterators(Iterator1& it1, Iterator2& it2,bool success = true){


    if(it1.total_number_particles() != it2.total_number_particles()){
        success = false;
        std::cout << "Number of particles mismatch" << std::endl;
    }

    uint64_t counter_1 = 0;
    uint64_t counter_2 = 0;

    for (int level = it1.level_min(); level <= it1.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it1.z_num(level); z++) {
            for (x = 0; x < it1.x_num(level); ++x) {

                it2.begin(level, z, x);

                for (it1.begin(level, z, x); it1 < it1.end();
                     it1++) {

                    counter_1++;

                    if(it1 != it1){

                        uint64_t new_index = it1;
                        uint64_t org_index = it2;

                        (void) new_index;
                        (void) org_index;


                        success = false;
//                        std::cout << "1" << std::endl;
                    }

                    if(it1.y() != it2.y()){

                        auto y_new = it1.y();
                        auto y_org = it2.y();

                        (void) y_new;
                        (void) y_org;

                        success = false;
//                        std::cout << "y_new" << y_new << std::endl;
//                        std::cout << "y_org" << y_org << std::endl;
                    }

                    if(it2 < it2.end()){
                        it2++;
                    }

                }
            }
        }
    }



    for (int level = it2.level_min(); level <= it2.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it2.z_num(level); z++) {
            for (x = 0; x < it2.x_num(level); ++x) {

                it1.begin(level, z, x);

                for (it2.begin(level, z, x); it2 < it2.end();
                     it2++) {

                    counter_2++;

                    if(it1 != it1){

                        uint64_t new_index = it1;
                        uint64_t org_index = it2;

                        (void) new_index;
                        (void) org_index;


                        success = false;
                    }

                    if(it1.y() != it2.y()){

                        auto y_new = it1.y();
                        auto y_org = it2.y();

                        (void) y_new;
                        (void) y_org;

                        success = false;
                    }

                    if(it1 < it1.end()){
                        it1++;
                    }

                }
            }
        }
    }

    if(counter_1 != counter_2){
        success = false;
        std::cout << "Iteration mismatch" << std::endl;
    }


    return success;
}

bool check_neighbours(APR& apr,APRIterator &current, APRIterator &neigh){


    bool success = true;

    if (std::abs((float)neigh.level() - (float)current.level()) > 1.0f) {
        success = false;
    }

    float delta_x = current.x_global(current.level(),current.x()) - neigh.x_global(neigh.level(),neigh.x());
    float delta_y = current.y_global(current.level(),current.y()) - neigh.y_global(neigh.level(),neigh.y());
    float delta_z = current.z_global(current.level(),current.z()) - neigh.z_global(neigh.level(),neigh.z());

    float resolution_max = 1.11*(0.5*pow(2,current.level_max()-current.level()) + 0.5*pow(2,neigh.level_max()-neigh.level()));

    float distance = sqrt(pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2));

    if(distance > resolution_max){
        success = false;
    }

    return success;
}
bool check_neighbour_out_of_bounds(APRIterator &current,uint8_t face){


    uint64_t num_neigh = current.number_neighbours_in_direction(face);

    if(num_neigh ==0){
        ParticleCell neigh = current.get_neigh_particle_cell();

        if( (neigh.x >= current.x_num(neigh.level) ) | (neigh.y >= current.y_num(neigh.level) ) | (neigh.z >= current.z_num(neigh.level) )  ){
            return true;
        } else {
            return false;
        }
    }

    return true;
}

bool test_auto_parameters(TestData& test_data){

    bool success = true;

    APR apr;
    APRConverter<uint16_t> aprConverter;

    aprConverter.par.auto_parameters = true;
    aprConverter.par.output_steps = true;

    aprConverter.get_apr(apr, test_data.img_original);

    auto par = apr.get_apr_parameters();

    //checks if run, and the values are positive
    if(par.sigma_th <= 0){
        success = false;
    }

    if(par.grad_th <= 0){
        success = false;
    }

    return success;

}

bool test_random_access_it(TestData& test_data){
    //
    //  Testing some of the random access features of the random access operators. Note these functions do not exist for the linear iterators.
    //

    bool success = true;

    auto it = test_data.apr.iterator();

    // Find some particles
    int test_number = 1000;

    test_number = std::min(1000.0,(test_data.apr.total_number_particles()/4.0));

    std::vector<ParticleCell> parts_exist;

    ParticleCell current_p;

    uint64_t delta = std::floor(test_data.apr.total_number_particles()/(test_number*1.0)) - 1;

    // choose approximately test_number particles to search for.
    for (int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                for (it.begin(level, z, x); it < it.end();
                     it++) {
                    if((it % delta) == 0){
                        current_p.x = x;
                        current_p.z = z;
                        current_p.y = it.y();
                        current_p.level = level;
                        parts_exist.push_back(current_p);
                    }

                }
            }
        }
    }

    auto random_it = test_data.apr.random_iterator();
    ParticleCell parent_part;

    //now use these to particles to search for
    for(size_t i = 0; i < parts_exist.size(); ++i) {

        bool found = random_it.set_iterator_by_particle_cell(parts_exist[i]);

        //this should exist by construction
        if(!found){
            success = false;
        }

        //check for the parent
        parent_part.x = parts_exist[i].x/2;
        parent_part.y = parts_exist[i].y/2;
        parent_part.z = parts_exist[i].z/2;
        parent_part.level = parts_exist[i].level - 1;

        (void) parent_part;

        bool not_found = random_it.set_iterator_by_particle_cell(parts_exist[i]);

        //this should NOT exist by construction
        if(!not_found){
            success = false;
        }

        //this returns the global co-ordinate of the particle
        float x = std::min(random_it.x_global(parts_exist[i].level,parts_exist[i].x),(float)random_it.org_dims(1)-1);
        float y = std::min(random_it.y_global(parts_exist[i].level,parts_exist[i].y),(float)random_it.org_dims(0)-1);
        float z = std::min(random_it.z_global(parts_exist[i].level,parts_exist[i].z),(float)random_it.org_dims(2)-1);

        ///////////////////////
        ///
        /// Set the iterator using random access by using a global co-ordinate (in original pixels), and setting the iterator, to the Particle Cell that contains the point in its spatial domain.
        ///
        ////////////////////////

        //should find the particle cell
        found = random_it.set_iterator_by_global_coordinate(x, y, z);

        //we know by construction that this must be within the domain
        if(!found){
            success = false;
        }

        if(random_it.x() != parts_exist[i].x){
            success = false;
        }

        if(random_it.y() != parts_exist[i].y){
            success = false;
        }

        if(random_it.z() != parts_exist[i].z){
            success = false;
        }

        if(random_it.level() != parts_exist[i].level){
            success = false;
        }

    }


    //check out of bounds
    bool found = random_it.set_iterator_by_global_coordinate(0, 0, 2*test_data.apr.org_dims(2)+2);

    if(found){
        success = false;
    }

    found = random_it.set_iterator_by_global_coordinate(0, 2*test_data.apr.org_dims(0)+2, 0);

    if(found){
        success = false;
    }

    found = random_it.set_iterator_by_global_coordinate(2*test_data.apr.org_dims(1)+2, 0, 0);

    if(found){
        success = false;
    }

    return success;

}

template<typename DataType>
void sub_slices_img(PixelData<DataType>& org,PixelData<DataType>& slices, std::vector<int>& dim_start,std::vector<int>& dim_stop){

    slices.init(dim_stop[0] - dim_start[0],dim_stop[1] - dim_start[1],dim_stop[2] - dim_start[2]);

    for (int z = dim_start[2]; z < dim_stop[2]; ++z) {
        for (int x = dim_start[1]; x < dim_stop[1]; ++x) {
            for (int y = dim_start[0]; y < dim_stop[0]; ++y) {

                slices.at(y-dim_start[0],x-dim_start[1],z-dim_start[2]) = org.at(y,x,z);

            }
        }
    }


}



template<typename DataType>
bool check_symmetry(PixelData<DataType>& img){
    //
    //  This function checks the symmetry of an image for a given dimension
    //

    //check_pad_array();

    bool success = true;
    bool sym_y = true;
    bool sym_x = true;
    bool sym_z = true;

    //difference of two allowed to account for rounding both ways error.

    for (int k = 0; k < img.z_num; ++k) {
        for (int j = 0; j < img.x_num; ++j) {
            for (int i = 0; i < img.y_num; ++i) {

                 {

                     //dim 0
                    float val = img.at(i, j, k);
                    float val_reflect = img.at(img.y_num - i - 1, j, k);

                    if ((val - val_reflect) > 2) {
                        success = false;
                        sym_y = false;
                    }
                }
                {
                    //dim 1
                    float val = img.at(i, j, k);
                    float val_reflect = img.at(i, img.x_num - j - 1, k);

                    if ((val - val_reflect) > 2) {
                        success = false;
                        sym_x = false;

                    }

                }
                {
                    //dim 2
                    float val = img.at(i, j, k);
                    float val_reflect = img.at(i, j, img.z_num - k - 1);

                    if ((val - val_reflect) > 2) {
                        success = false;
                        sym_z = false;
                    }

                }


            }
        }
    }

    if(!sym_y){
        std::cout << "not symmetric y" << std::endl;
    }

    if(!sym_x){
        std::cout << "not symmetric x" << std::endl;
    }

    if(!sym_z){
        std::cout << "not symmetric z" << std::endl;
    }



    return success;


}


bool test_symmetry_pipeline(){
    //
    //  The pipeline steps should be symmetry about x,z,y boundaries.. this test checks this using a synthetic symmetric input image
    //
    //
    //  Tests this by creating a symmetry image that is constant in one dimension.
    //
    //

    bool success = true;

    for (int dim = 0; dim < 3; ++dim) {
        //compute the tests across all three dimensions

        for (int sz_slice = 1; sz_slice < 10; sz_slice++) {

            //Create a synethic symmetric image with a square in the middle.
            PixelData<uint16_t> img;

            int sz = 64;

            if(dim ==0){

                img.initWithValue(sz_slice, sz, sz, 100);

                int block_sz = 20;

                for (int i = block_sz; i < (sz - block_sz); ++i) {
                    for (int j = block_sz; j < (sz - block_sz); ++j) {
                        for (int k = 0; k < sz_slice; ++k) {
                            img.at(k, i, j) += 1000;
                        }
                    }
                }

            } else if (dim ==1){

                img.initWithValue(sz, sz_slice, sz, 100);

                int block_sz = 20;

                for (int i = block_sz; i < (sz - block_sz); ++i) {
                    for (int j = block_sz; j < (sz - block_sz); ++j) {
                        for (int k = 0; k < sz_slice; ++k) {
                            img.at(i, k, j) += 1000;
                        }
                    }
                }

            } else if (dim ==2){

                img.initWithValue(sz, sz, sz_slice, 100);

                int block_sz = 20;

                for (int i = block_sz; i < (sz - block_sz); ++i) {
                    for (int j = block_sz; j < (sz - block_sz); ++j) {
                        for (int k = 0; k < sz_slice; ++k) {
                            img.at(i, j, k) += 1000;
                        }
                    }
                }

            }


            APR apr;
            APRConverter<uint16_t> aprConverter;
            aprConverter.par.output_steps = true;

            aprConverter.par.lambda = 3;

            aprConverter.get_apr(apr, img);

            ParticleData<uint16_t> parts;
            parts.sample_image(apr, img);

            // get grad/scale/level/final level/final image. --> All should be symmetric!!! //could be nice to have a more elagant method for this.
            PixelData<float> scale = TiffUtils::getMesh<float>("local_intensity_scale_step.tif");
            PixelData<uint16_t> grad = TiffUtils::getMesh<uint16_t>("gradient_step.tif");
            PixelData<float> lps = TiffUtils::getMesh<float>("local_particle_set_level_step.tif");

            PixelData<uint16_t> smooth = TiffUtils::getMesh<uint16_t>("smooth_bsplines.tif");

            PixelData<uint16_t> recon_img;
            APRReconstruction::reconstruct_constant(apr, recon_img, parts);

            PixelData<uint16_t> level_img;
            APRReconstruction::reconstruct_level(apr, level_img);

//    TiffUtils::saveMeshAsTiff("level_image.tif",level_img);
//    TiffUtils::saveMeshAsTiff("img_recon.tif",recon_img);


            if (check_symmetry(img)) {
                //std::cout << "image symmetric" << std::endl;
            } else {
                std::cout << "image not symmetric" << std::endl;
                success = false;
            }

            if (check_symmetry(smooth)) {
                //std::cout << "smooth symmetric" << std::endl;
            } else {
                std::cout << "smooth not symmetric" << std::endl;
                success = false;
            }

            if (check_symmetry(grad)) {
                //std::cout << "grad symmetric" << std::endl;
            } else {
                std::cout << "grad not symmetric" << std::endl;
                success = false;
            }

            if (check_symmetry(scale)) {
                // std::cout << "scale symmetric" << std::endl;
            } else {
                std::cout << "scale not symmetric" << std::endl;
                success = false;
            }

            if (check_symmetry(lps)) {
                //std::cout << "lps symmetric" << std::endl;
            } else {
                std::cout << "lps not symmetric" << std::endl;
                success = false;
            }

            if (check_symmetry(level_img)) {
                //std::cout << "level_img symmetric" << std::endl;
            } else {
                std::cout << "level_img not symmetric" << std::endl;
                success = false;
            }

            if (check_symmetry(recon_img)) {
                //std::cout << "recon_img symmetric" << std::endl;
            } else {
                std::cout << "recon_img not symmetric" << std::endl;
                success = false;
            }
        }
    }


    return success;

}


bool test_pipeline_mask(TestData& test_data){
    //
    //
    //

    PixelData<uint16_t> mask;

    mask.init(test_data.img_original);

    float th = 3000;

    for (size_t i = 0; i < mask.mesh.size(); ++i) {
        mask.mesh[i] = (test_data.img_original.mesh[i] > th);
    }

    TiffUtils::saveMeshAsTiff("mask.tif",mask,false);

    APRConverter<uint16_t> aprConverter;

    APR apr;
    APR apr_masked;

    aprConverter.get_apr(apr,test_data.img_original);

    aprConverter.par.mask_file = "mask.tif";

    aprConverter.get_apr(apr_masked,test_data.img_original);

    if(apr.total_number_particles() < apr_masked.total_number_particles()){
        return false;
    }

    return true;

}




bool test_pipeline_different_sizes(TestData& test_data){

    //just a run test, no checks.

    bool success = true;

    APRConverter<uint16_t> aprConverter;

    PixelData<uint16_t> input_data;

    int min = 1;
    int max = 4;

    for (int i = min; i < max; ++i) {
        for (int j = min; j < max; ++j) {
            for (int k = min; k < max; ++k) {
                input_data.init(k,j,i);

                APR apr;

                aprConverter.get_apr(apr,input_data);

                ParticleData<uint16_t> parts;

                parts.sample_image(apr, input_data);

                APRFile aprFile;
                aprFile.open("test_small.apr","WRITE");
                aprFile.write_apr(apr);
                aprFile.write_particles("par",parts);
                aprFile.close();

            }
        }
    }

    //below code is for debbugging.

    //test slices
//
//    PixelData<uint16_t> img_slice;
//
//    std::vector<int> dim_start;
//    std::vector<int> dim_stop;
//
//    int num_slices = 3;
//
//    dim_start.resize(3);
//    dim_stop.resize(3);
//
//    dim_start[0] = 0;
//    dim_start[1] = 0;
//    dim_start[2] = test_data.img_original.z_num/2;
//
//    dim_stop[0] = test_data.img_original.y_num;
//    dim_stop[1] = test_data.img_original.x_num;
//    dim_stop[2] = dim_start[2] + num_slices;
//
//    sub_slices_img(test_data.img_original,img_slice,  dim_start, dim_stop);
//
//    APR apr;
//
//    aprConverter.par.auto_parameters = true;
//    aprConverter.par.output_steps = true;
//
//    aprConverter.get_apr(apr,img_slice);
//
//    ParticleData<uint16_t> parts;
//
//    parts.sample_parts_from_img_downsampled(apr,img_slice);
//
//    TiffUtils::saveMeshAsTiff("img.tif",img_slice);
//
//    PixelData<uint16_t> img;
//
//    APRReconstruction::interp_img(apr,img,parts);
//
//    TiffUtils::saveMeshAsTiff("img_recon.tif",img);
//
//    parts.fill_with_levels(apr);
//    APRReconstruction::interp_img(apr,img,parts);
//
//    TiffUtils::saveMeshAsTiff("img_levels.tif",img);


    return success;

}



bool test_pulling_scheme_sparse(TestData& test_data){
    bool success = true;

    APRConverter<uint16_t> aprConverter;

    //read in the command line options into the parameters file
    aprConverter.par.Ip_th = 0;
    aprConverter.par.rel_error = 0.1;
    aprConverter.par.lambda = 2;
    aprConverter.par.mask_file = "";

    aprConverter.par.sigma_th_max = 50;
    aprConverter.par.sigma_th = 100;

    aprConverter.par.auto_parameters = false;

    aprConverter.par.output_steps = false;

    //where things are
    aprConverter.par.input_image_name = test_data.filename;
    aprConverter.par.input_dir = "";
    aprConverter.par.name = test_data.output_name;
    aprConverter.par.output_dir = test_data.output_dir;

    aprConverter.method_timer.verbose_flag = true;

    //Gets the APR

    ParticleData<uint16_t> particles_intensities;

    APR apr_org;

    aprConverter.get_apr(apr_org,test_data.img_original);

    APR apr_org_sparse;
    aprConverter.set_sparse_pulling_scheme(true);

    aprConverter.get_apr(apr_org_sparse,test_data.img_original);

    APR apr_lin_sparse;
    aprConverter.set_generate_linear(true);

    aprConverter.get_apr(apr_lin_sparse,test_data.img_original);


    auto org_it = apr_org.random_iterator();
    auto sparse_it = apr_org_sparse.iterator();
    auto sparse_lin_it = apr_lin_sparse.iterator();


    success = compare_two_iterators(org_it,sparse_it,success);
    success = compare_two_iterators(sparse_lin_it,sparse_it,success);
    success = compare_two_iterators(org_it,sparse_lin_it,success);

    return success;
}



bool test_linear_access_create(TestData& test_data) {


    bool success = true;

    APR apr;

    APRConverter<uint16_t> aprConverter;

    //read in the command line options into the parameters file
    aprConverter.par.Ip_th = 0;
    aprConverter.par.rel_error = 0.1;
    aprConverter.par.lambda = 2;
    aprConverter.par.mask_file = "";

    aprConverter.par.sigma_th_max = 50;
    aprConverter.par.sigma_th = 100;

    aprConverter.par.auto_parameters = false;

    aprConverter.par.output_steps = false;

    //where things are
    aprConverter.par.input_image_name = test_data.filename;
    aprConverter.par.input_dir = "";
    aprConverter.par.name = test_data.output_name;
    aprConverter.par.output_dir = test_data.output_dir;

    aprConverter.method_timer.verbose_flag = true;

    aprConverter.set_generate_linear(false);

    //Gets the APR

    ParticleData<uint16_t> particles_intensities;

    aprConverter.get_apr(apr,test_data.img_original);

    particles_intensities.sample_image(apr, test_data.img_original);

    //test the partcell data structures
    PartCellData<uint16_t> partCellData_intensities;
    partCellData_intensities.sample_image(apr, test_data.img_original);

    auto it = apr.iterator();

    for (int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                for (it.begin(level, z, x); it < it.end();
                     it++) {

                    auto p1 = particles_intensities[it];
                    auto p2 = partCellData_intensities[it];


                    if(p1 != p2){

                        success = false;
                    }
                }
            }
        }
    }


    auto it_org = apr.random_iterator();

    APR apr_lin;

    aprConverter.set_generate_linear(true);

    aprConverter.get_apr(apr_lin,test_data.img_original);

    if(apr.total_number_particles() != apr_lin.total_number_particles()){

        auto org = apr.total_number_particles();
        auto direct = apr_lin.total_number_particles();

        (void) org;
        (void) direct;

        success = false;
    }

    auto it_lin_old = apr.iterator();

    //apr_lin.init_linear();
    auto it_new = apr_lin.iterator();

    success = compare_two_iterators(it_lin_old,it_new,success);

    //Test Linear -> Random generation
    auto it_new_random = apr_lin.random_iterator();

    success = compare_two_iterators(it_new_random,it_new,success);


    //Test the APR Tree construction.

    auto tree_it_org = apr.random_tree_iterator();
    auto tree_it_lin = apr_lin.tree_iterator();

    auto total_number_parts = tree_it_org.total_number_particles();
    auto total_number_parts_lin = tree_it_lin.total_number_particles();

    std::cout << "PARTS: " << total_number_parts << " " << total_number_parts_lin << std::endl;


    success = compare_two_iterators(tree_it_org,tree_it_lin,success);


    return success;
}



bool test_particles_compress(TestData& test_data){


    bool success = true;

    ParticleData<uint16_t> parts2compress;

    parts2compress.copy_parts(test_data.apr,test_data.particles_intensities);

    std::string file_name = "compress_test.apr";

    parts2compress.compressor.set_compression_type(1);
    parts2compress.compressor.set_background(900);
    parts2compress.compressor.set_quantization_factor(0.01);

    APRFile writeFile;

    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(test_data.apr);

    float file_size_1 = writeFile.current_file_size_MB();

    writeFile.write_particles("parts",parts2compress);

    float file_size_2 = writeFile.current_file_size_MB();

    writeFile.close();

    APR read_apr;
    ParticleData<uint16_t> read_parts;

    writeFile.open(file_name,"READ");

    writeFile.read_apr(read_apr);

    writeFile.read_particles(read_apr,"parts",read_parts);

    writeFile.close();

    for (size_t i = 0; i < test_data.particles_intensities.size(); ++i) {

        auto org = test_data.particles_intensities[i];
        auto comp = read_parts[i];

        //allowing quantization
        if((org - comp) > 1){
            success = false;
        }
    }

    parts2compress.copy_parts(test_data.apr,test_data.particles_intensities);

    writeFile.open(file_name,"READWRITE");

    parts2compress.compressor.set_quantization_factor(1);

    writeFile.write_particles("parts_2",parts2compress);

    float file_size_3 = writeFile.current_file_size_MB();

    writeFile.read_particles(read_apr,"parts_2",read_parts);

    writeFile.close();

    for (size_t i = 0; i < test_data.particles_intensities.size(); ++i) {

        auto org = test_data.particles_intensities[i];
        auto comp = read_parts[i];

        //allowing quantization
        if((org - comp)/(1.0f*org) > 0.05){
            success = false;
        }
    }

    float size_1 = file_size_2 - file_size_1;
    float size_2 = file_size_3 - file_size_2;

    if(size_2 > size_1){
        //success = false;
    }


    return success;


}


bool test_lazy_particles(TestData& test_data){

    bool success = true;

    auto it = test_data.apr.iterator();

    std::string file_name = "parts_lazy_test.apr";

    APRFile writeFile;

    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(test_data.apr);

    writeFile.write_particles("parts",test_data.particles_intensities);

    writeFile.close();

    writeFile.open(file_name,"READWRITE");

    LazyData<uint16_t> parts_lazy;

    parts_lazy.init(writeFile, "parts");

    parts_lazy.open();

    uint64_t dataset_size = parts_lazy.dataset_size();

    if(dataset_size != test_data.particles_intensities.size()){
        success = false;
    }

    APRTimer timer(true);

    timer.start_timer("load loop slice");

    for (int level = (it.level_max()); level >= it.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            parts_lazy.load_slice(level,z,it);
            for (x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level,z,x); it < it.end();
                     it++) {
                    //add caching https://support.hdfgroup.org/HDF5/doc/H5.user/Caching.html
                    if(test_data.particles_intensities[it] != parts_lazy[it]){
                        success = false;
                    }

                    parts_lazy[it] += 1;

                }
            }
            parts_lazy.write_slice(level,z,it);
        }
    }

    timer.stop_timer();



    timer.start_timer("load loop by row");

    for (int level = (it.level_max()); level >= it.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                for (parts_lazy.load_row(level,z,x,it); it < it.end();
                     it++) {
                    if((test_data.particles_intensities[it]+1) != parts_lazy[it]){
                        success = false;
                    }
                }
            }
        }
    }

    timer.stop_timer();


    parts_lazy.close();

    writeFile.close();

    //
    //  Test create file
    //

    writeFile.open(file_name,"READWRITE");

    LazyData<uint16_t> parts_lazy_create;
    parts_lazy_create.init(writeFile, "parts_create");

    parts_lazy_create.create_file(test_data.particles_intensities.size());

    parts_lazy_create.open();

    for (int level = (it.level_max()); level >= it.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            parts_lazy_create.load_slice(level,z,it);
            for (x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level,z,x); it < it.end();
                     it++) {
                    //add caching https://support.hdfgroup.org/HDF5/doc/H5.user/Caching.html

                    parts_lazy_create[it] = test_data.particles_intensities[it];

                }
            }
            parts_lazy_create.write_slice(level,z,it);
        }
    }

    parts_lazy_create.close();

    ParticleData<uint16_t> parts_read;
    writeFile.read_particles(test_data.apr,"parts_create",parts_read);

    writeFile.close();

    //check the read particles
    for (size_t i = 0; i < test_data.particles_intensities.size(); ++i) {
        if(test_data.particles_intensities[i] != parts_read[i]){
            success = false;
        }
    }

    return success;


}


bool test_linear_access_io(TestData& test_data) {


    bool success = true;

    //Test Basic IO
    std::string file_name = "read_write_random.apr";

    APRTimer timer(true);

    timer.start_timer("random write");

    //First write a file
    APRFile writeFile;
    writeFile.set_write_linear_flag(false); //write the random files

    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(test_data.apr);

    writeFile.write_particles("parts",test_data.particles_intensities);

    writeFile.close();

    timer.stop_timer();

    timer.start_timer("random read");

    APR apr_random;

    writeFile.open(file_name,"READ");
    writeFile.read_apr(apr_random);

    writeFile.close();

    timer.stop_timer();

    timer.start_timer("linear write");

    file_name = "read_write_linear.apr";

    writeFile.set_write_linear_flag(true);

    writeFile.open(file_name,"WRITE");

    writeFile.set_blosc_access_settings(BLOSC_ZSTD,4,1);

    writeFile.write_apr(test_data.apr);

    writeFile.write_particles("parts",test_data.particles_intensities);

    writeFile.close();

    timer.stop_timer();

    timer.start_timer("linear read");

    APR apr_lin;

    writeFile.open(file_name,"READ");
    writeFile.read_apr(apr_lin);

    timer.stop_timer();

    auto it_org = apr_random.random_iterator();
    auto it_new = apr_lin.iterator();
    auto it_read = test_data.apr.iterator();

    success = compare_two_iterators(it_org,it_new,success);
    success = compare_two_iterators(it_org,it_read,success);
    success = compare_two_iterators(it_new,it_read,success);

    // Test the tree IO

    auto tree_it_org = apr_random.random_tree_iterator();
    auto tree_it_lin = apr_lin.tree_iterator();

    success = compare_two_iterators(tree_it_org,tree_it_lin,success);

    return success;
}




bool test_apr_tree(TestData& test_data) {

    bool success = true;

    std::string save_loc = "";
    std::string file_name = "read_write_test";

    auto it_lin  = test_data.apr.iterator();
    auto it_random = test_data.apr.random_iterator();

    success = compare_two_iterators(it_lin,it_random,success);

    auto it_tree_t = test_data.apr.random_tree_iterator();

    success = compare_two_iterators(it_lin,it_random,success);

    ParticleData<float> tree_data;

    // tests the random access tree iteration.

    auto apr_tree_iterator = test_data.apr.random_tree_iterator();
    auto apr_tree_iterator_lin = test_data.apr.tree_iterator();

    success = compare_two_iterators(apr_tree_iterator,apr_tree_iterator_lin,success);


    for (int level = (apr_tree_iterator.level_max()); level >= apr_tree_iterator.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_tree_iterator.z_num(level); z++) {
            for (x = 0; x < apr_tree_iterator.x_num(level); ++x) {
                for (apr_tree_iterator.begin(level, z, x); apr_tree_iterator < apr_tree_iterator.end();
                     apr_tree_iterator++) {

                    int y_g = std::min(apr_tree_iterator.y_nearest_pixel(level,apr_tree_iterator.y()),apr_tree_iterator.org_dims(0)-1);
                    int x_g = std::min(apr_tree_iterator.x_nearest_pixel(level,x),apr_tree_iterator.org_dims(1)-1);
                    int z_g = std::min(apr_tree_iterator.z_nearest_pixel(level,z),apr_tree_iterator.org_dims(2)-1);

                    int val = test_data.img_level.at(y_g,x_g,z_g);

                    //since its in the tree the image much be at a higher resolution then the tree. Direct check.
                    if(level > val){
                        success = false;
                    }

                }
            }
        }
    }




//    aprTree.fill_tree_mean(test_data.apr,aprTree,test_data.particles_intensities,tree_data);
//
//    aprTree.fill_tree_mean_downsample(test_data.particles_intensities);

    APRTreeNumerics::fill_tree_mean(test_data.apr,test_data.particles_intensities,tree_data);



    //generate tree test data
    PixelData<float> pc_image;
    APRReconstruction::reconstruct_constant(test_data.apr,pc_image,test_data.particles_intensities);


    std::vector<PixelData<float>> downsampled_img;
    //Down-sample the image for particle intensity estimation
    downsamplePyramid(pc_image, downsampled_img, test_data.apr.level_max(), test_data.apr.level_min() - 1);


    auto tree_it_random = test_data.apr.random_tree_iterator();

    for (int level = (tree_it_random.level_max()); level >= tree_it_random.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < tree_it_random.z_num(level); z++) {
            for (x = 0; x < tree_it_random.x_num(level); ++x) {
                for (tree_it_random.begin(level, z, x); tree_it_random < tree_it_random.end();
                     tree_it_random++) {

                    uint16_t current_int = (uint16_t)std::round(downsampled_img[tree_it_random.level()].at(tree_it_random.y(),tree_it_random.x(),tree_it_random.z()));
                    //uint16_t parts_int = aprTree.particles_ds_tree[apr_tree_iterator];
                    uint16_t parts2 = (uint16_t)std::round(tree_data[tree_it_random]);

                    // uint16_t y = apr_tree_iterator.y();

                    if(abs(parts2 - current_int) > 1){
                        success = false;
                    }

                }
            }
        }
    }

    //Also check the sparse data-structure generated tree

    ParticleData<float> treedata_2;


    APRTreeNumerics::fill_tree_mean(test_data.apr,test_data.particles_intensities,treedata_2);

//    tree2.fill_tree_mean(test_data.apr,tree2,test_data.particles_intensities,treedata_2);
    auto apr_tree_iterator_s = test_data.apr.random_tree_iterator();

    for (int level = (apr_tree_iterator_s.level_max()); level >= apr_tree_iterator_s.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_tree_iterator_s.z_num(level); z++) {
            for (x = 0; x < apr_tree_iterator_s.x_num(level); ++x) {
                for (apr_tree_iterator_s.set_new_lzx(level, z, x); apr_tree_iterator_s < apr_tree_iterator_s.end();
                     apr_tree_iterator_s.set_iterator_to_particle_next_particle()) {

                    uint16_t current_int = (uint16_t)std::round(downsampled_img[apr_tree_iterator_s.level()].at(apr_tree_iterator_s.y(),apr_tree_iterator_s.x(),apr_tree_iterator_s.z()));
                    //uint16_t parts_int = aprTree.particles_ds_tree[apr_tree_iterator];
                    uint16_t parts2 = (uint16_t)std::round(treedata_2[apr_tree_iterator_s]);

                    // uint16_t y = apr_tree_iterator.y();

                    if(abs(parts2 - current_int) > 1){
                        success = false;
                    }

                }
            }
        }
    }


    if(it_lin.number_dimensions() == 3) {
        //note there is no current support for 2D and 1D random neighbour access.

        auto neigh_tree_iterator = test_data.apr.random_tree_iterator();


        for (int level = apr_tree_iterator.level_min(); level <= apr_tree_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

            for (z = 0; z < apr_tree_iterator.z_num(level); z++) {
                for (x = 0; x < apr_tree_iterator.x_num(level); ++x) {
                    for (apr_tree_iterator.set_new_lzx(level, z, x); apr_tree_iterator < apr_tree_iterator.end();
                         apr_tree_iterator.set_iterator_to_particle_next_particle()) {

                        //loop over all the neighbours and set the neighbour iterator to it
                        for (int direction = 0; direction < 6; ++direction) {
                            apr_tree_iterator.find_neighbours_same_level(direction);
                            // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                            for (int index = 0;
                                 index < apr_tree_iterator.number_neighbours_in_direction(direction); ++index) {

                                if (neigh_tree_iterator.set_neighbour_iterator(apr_tree_iterator, direction, index)) {
                                    //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)

                                    uint16_t current_int = (uint16_t) std::round(
                                            downsampled_img[neigh_tree_iterator.level()].at(neigh_tree_iterator.y(),
                                                                                            neigh_tree_iterator.x(),
                                                                                            neigh_tree_iterator.z()));
                                    //uint16_t parts_int = aprTree.particles_ds_tree[apr_tree_iterator];
                                    uint16_t parts2 = (uint16_t) std::round(tree_data[neigh_tree_iterator]);

                                    //uint16_t y = apr_tree_iterator.y();

                                    if (abs(parts2 - current_int) > 1) {
                                        success = false;
                                    }

                                }
                            }
                        }
                    }
                }
            }
        }
    }



    return success;
}

template<typename T>
bool check_particle_type(std::string& type_check){

    //Test Basic IO
    std::string file_name = "read_write_test.apr";

    //First write a file
    APRFile writeFile;
    writeFile.open(file_name,"WRITE");

    const size_t parts_num = 10;

    ParticleData<T> part_write;
    part_write.init(parts_num);

    writeFile.write_particles("part_write",part_write);

    writeFile.close();

    //First read a file
    APRFile readFile;

    bool success = true;

    readFile.open(file_name,"READ");

    std::string read_type = readFile.get_particle_type("part_write");

    readFile.close();

    if(type_check != read_type){
        success = false;
    }

    return success;

}



bool test_apr_file_particle_type(){

    std::vector<std::string> type_list = {"uint16", "float", "uint8", "uint32", "uint64", "double", "int8", "int16", "int32", "int64"};

    bool success = true;

    success = success & check_particle_type<uint16_t>(type_list[0]);
    success = success & check_particle_type<float>(type_list[1]);
    success = success & check_particle_type<uint8_t>(type_list[2]);
    success = success & check_particle_type<uint32_t>(type_list[3]);
    success = success & check_particle_type<uint64_t>(type_list[4]);
    success = success & check_particle_type<double>(type_list[5]);
    success = success & check_particle_type<int8_t>(type_list[6]);
    success = success & check_particle_type<int16_t>(type_list[7]);
    success = success & check_particle_type<int32_t>(type_list[8]);
    success = success & check_particle_type<int64_t>(type_list[9]);

    return success;
}


bool test_apr_file(TestData& test_data){


    //Test Basic IO
    std::string file_name = "read_write_test.apr";

    //First write a file
    APRFile writeFile;
    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(test_data.apr);

    writeFile.write_particles("parts",test_data.particles_intensities);

    ParticleData<float> parts2;
    parts2.init(test_data.apr.total_number_particles());

    auto apr_it = test_data.apr.iterator();

    for (size_t i = 0; i < apr_it.total_number_particles(); ++i) {
        parts2[i] = test_data.particles_intensities[i]*3 - 1;
    }

    writeFile.write_particles("parts2",parts2);

    writeFile.close();

    //First read a file
    APRFile readFile;

    bool success = true;

    readFile.open(file_name,"READ");

    APR aprRead;

    readFile.read_apr(aprRead);

    ParticleData<uint16_t> parts_read;

    readFile.read_particles(aprRead,"parts",parts_read);

    // Default read without name, takes the first particles.
    ParticleData<uint16_t> parts_default;
    bool no_name = readFile.read_particles(aprRead,parts_default);

    if(!no_name){
        success = false;
    }

    ParticleData<float> parts2_read;

    readFile.read_particles(aprRead,"parts2",parts2_read);

    readFile.close();

    auto apr_iterator = test_data.apr.iterator();
    auto apr_iterator_read = aprRead.iterator();

    success = compare_two_iterators(apr_iterator,apr_iterator_read,success);

    //test apr iterator with channel
    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(test_data.apr, 0, "channel_0");
    writeFile.write_apr(test_data.apr, 55, "channel_0");

    writeFile.write_particles("parts", test_data.particles_intensities, true, 0, "channel_0");
    writeFile.write_particles("parts", test_data.particles_intensities, true, 55, "channel_0");

    writeFile.close();

    writeFile.open(file_name, "READ");

    APR apr_channel_0;
    ParticleData<uint16_t> parts_channel_0;

    APR apr_channel_0_55;
    ParticleData<uint16_t> parts_channel_0_55;

    writeFile.read_apr(apr_channel_0, 0, "channel_0");

    if(!writeFile.read_particles("parts", parts_channel_0, true, 0, "channel_0")){
        success = false;
    }

    //without name
    if(!writeFile.read_particles(apr_channel_0, parts_channel_0, true, 0, "channel_0")){
        success = false;
    }

    writeFile.read_apr(apr_channel_0_55, 55, "channel_0");

    if(!writeFile.read_particles("parts", parts_channel_0_55, true, 55, "channel_0")){
        success = false;
    }

    writeFile.close();

    auto it1 = apr_channel_0.iterator();
    auto it2 = test_data.apr.iterator();
    success = compare_two_iterators(it1,it2,success);

    it1 = apr_channel_0_55.iterator();
    it2 = test_data.apr.iterator();
    success = compare_two_iterators(it1,it2,success);



    //Test Tree IO and RW and channel
    APRFile TreeFile;
    file_name = "read_write_test_tree.apr";
    TreeFile.open(file_name,"WRITE");

    TreeFile.write_apr(test_data.apr,0,"mem");


    ParticleData<float> treeMean;

    APRTreeNumerics::fill_tree_mean(test_data.apr,test_data.particles_intensities,treeMean);

    TreeFile.write_particles("tree_parts",treeMean,false,0,"mem");
    TreeFile.write_particles("tree_parts1",treeMean,false,0,"mem");
    TreeFile.write_particles("tree_parts2",treeMean,false,0,"mem");

    TreeFile.close();

    TreeFile.open(file_name,"READWRITE");
    TreeFile.write_apr(test_data.apr,1,"mem");
    TreeFile.write_particles("tree_parts",treeMean,false,1,"mem");

    TreeFile.write_particles("particle_demo",test_data.particles_intensities,true,1,"mem");

    TreeFile.write_apr(test_data.apr,10,"ch1_");

    TreeFile.close();

    TreeFile.open(file_name,"READ");

    APR aprRead2;
    TreeFile.read_apr(aprRead2,1,"mem");

    ParticleData<float> treeMeanRead;

    TreeFile.read_particles(aprRead2,"tree_parts",treeMeanRead,false,1,"mem");

    auto tree_it = aprRead2.random_tree_iterator();
    auto tree_it_org = test_data.apr.random_tree_iterator();

    success = compare_two_iterators(tree_it,tree_it_org,success);

    //Test file list
    std::vector<std::string> correct_names = {"tree_parts","tree_parts1","tree_parts2"};

    std::vector<std::string> dataset_names = TreeFile.get_particles_names(false,0,"mem");

    if(correct_names.size() == dataset_names.size()){

        for(size_t i = 0; i < correct_names.size(); ++i) {
            bool found = false;
            for(size_t j = 0; j < dataset_names.size(); ++j) {
                if(correct_names[i] == dataset_names[j]){
                    found = true;
                }
            }
            if(!found){
                success = false;
            }
        }

    } else{
        success = false;
    }

    std::vector<std::string> channel_names;
    channel_names = TreeFile.get_channel_names();

    //Test file list
    std::vector<std::string> channel_names_c = {"mem","mem1","ch1_10"};

    if(channel_names_c.size() == channel_names.size()){

        for (size_t i = 0; i < channel_names_c.size(); ++i) {
            bool found = false;
            for (size_t j = 0; j < channel_names.size(); ++j) {
                if(channel_names_c[i] == channel_names[j]){
                    found = true;
                }
            }
            if(!found){
                success = false;
            }
        }

    } else{
        success = false;
    }


    uint64_t time_steps = TreeFile.get_number_time_steps("mem");

    if(time_steps != 2){
        success = false;
    }

    TreeFile.close();


    /*
     * Testing files and datasets that don't exist
     */


    //file doesn't exist
    APRFile testFile2;
    bool exist = testFile2.open("does_not_exist.apr","READ");

    if(exist){
        success = false;
    }

    APR apr_ne;
    //read from file thats not open
    bool exist_2 = testFile2.read_apr(apr_ne);

    if(exist_2){
        success = false;
    }

    ParticleData<uint16_t> parts_dne;

    bool exist_3 = testFile2.read_particles(apr_ne,"parts_dne",parts_dne);

    if(exist_3){
        success = false;
    }

    testFile2.close();

    //now lets create a file, and then try and ended.
    bool exist_4 = testFile2.open("exists.apr","WRITE");

    if(!exist_4){
        success = false;
    }

    testFile2.close();

    //opens file exists
    bool exist_5 = testFile2.open("exists.apr","READWRITE");

    if(!exist_5){
        success = false;
    }

    //apr does not exist
    bool exist_6 = testFile2.read_apr(apr_ne);

    if(exist_6){
        success = false;
    }

    //apr is not valid.
    bool exist_7 = testFile2.read_particles(apr_ne,"parts_dne",parts_dne);

    if(exist_7){
        success = false;
    }

    //particle dataset does not exist
    bool exist_8 = testFile2.read_particles(aprRead2,"parts_dne",parts_dne);

    if(exist_8){
        success = false;
    }

    parts_dne.init(5);

    bool exist_10 = testFile2.write_particles("parts_exist",parts_dne);

    if(!exist_10){
        success = false;
    }

    testFile2.close();

    testFile2.open("exists.apr","READ");

    bool exist_9 = testFile2.read_particles(aprRead2,"parts_exist",parts_dne);

    if(exist_9){
        success = false;
    }

    testFile2.close();


    return success;

}


bool test_read_upto_level(TestData& test_data){

    bool success = true;

    //Test Basic IO
    std::string file_name = "read_write_test.apr";

    //First write a file
    APRFile writeFile;
    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(test_data.apr);

    writeFile.write_particles("parts",test_data.particles_intensities);

    writeFile.close();

    //
    //  Read Full
    //

    ParticleData<uint16_t> parts_full;

    writeFile.open(file_name,"READ");

    writeFile.read_particles(test_data.apr,"parts",parts_full);


    //
    //  Partial Read Parts
    //

    for (int delta = 0; delta < (test_data.apr.level_max()- test_data.apr.level_min()); ++delta) {

        writeFile.set_max_level_read_delta(delta);

        ParticleData<uint16_t> parts_partial;
        writeFile.read_particles(test_data.apr, "parts", parts_partial);

        auto it = test_data.apr.iterator();

        uint64_t counter = 0;

        for (int level = it.level_min(); level <= (it.level_max() - delta); ++level) {
            for (int z = 0; z < it.z_num(level); ++z) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it != it.end(); ++it) {
                        counter++;
                        auto val_org = parts_full[it];
                        auto val_partial = parts_partial[it];

                        if (val_org != val_partial) {
                            success = false;
                        }
                    }
                }
            }
        }

        auto size_partial = parts_partial.size();
        if (counter != size_partial) {
            success = false;
        }

    }

    writeFile.close();

    //
    //  Partial Read Parts Tree
    //


    ParticleData<uint16_t> parts_tree;
    APRTreeNumerics::fill_tree_mean(test_data.apr,test_data.particles_intensities,parts_tree);

    writeFile.open(file_name,"READWRITE");

    writeFile.write_particles("parts_tree",parts_tree,false,0);

    for (int delta = 0; delta < (test_data.apr.level_max()- test_data.apr.level_min()); ++delta) {

        writeFile.set_max_level_read_delta(delta);

        ParticleData<uint16_t> parts_partial;
        writeFile.read_particles(test_data.apr, "parts_tree", parts_partial,false,0);

        auto it_tree = test_data.apr.tree_iterator();
        auto it = test_data.apr.iterator();

        uint64_t counter = 0;

        for (int level = it_tree.level_min(); level <= std::min((int)it_tree.level_max(),(it.level_max() - delta)); ++level) {
            for (int z = 0; z < it_tree.z_num(level); ++z) {
                for (int x = 0; x < it_tree.x_num(level); ++x) {
                    for (it_tree.begin(level, z, x); it_tree != it_tree.end(); ++it_tree) {
                        counter++;
                        auto val_org = parts_tree[it_tree];
                        auto val_partial = parts_partial[it_tree];

                        if (val_org != val_partial) {
                            success = false;
                        }
                    }
                }
            }
        }

        auto size_partial = parts_partial.size();
        if (counter != size_partial) {
            success = false;
        }

    }


    //
    //  Partial Read APR
    //

    for (int delta = 0; delta < (test_data.apr.level_max()- test_data.apr.level_min()); ++delta) {

        writeFile.set_max_level_read_delta(delta);

        APR apr_partial;

        writeFile.read_apr(apr_partial);

        auto it_org = test_data.apr.iterator();
        auto it_partial = apr_partial.iterator();

        //test apr

        for (int level = it_partial.level_min(); level <= (it_partial.level_max()); ++level) {
            for (int z = 0; z < it_partial.z_num(level); ++z) {
                for (int x = 0; x < it_partial.x_num(level); ++x) {

                    it_org.begin(level, z, x);
                    for (it_partial.begin(level, z, x); it_partial != it_partial.end(); ++it_partial) {

                        if(it_partial != it_org){
                            success = false;
                        }

                        if(it_org.y() != it_partial.y()){
                            success = false;
                        }

                        if(it_org < it_org.end()){
                            it_org++;
                        }

                    }
                }
            }
        }

        //test tree

        auto it_org_tree = test_data.apr.tree_iterator();
        auto it_partial_tree = apr_partial.tree_iterator();

        for (int level = it_partial_tree.level_min(); level <= (it_partial_tree.level_max()); ++level) {
            for (int z = 0; z < it_partial_tree.z_num(level); ++z) {
                for (int x = 0; x < it_partial_tree.x_num(level); ++x) {

                    it_org_tree.begin(level, z, x);
                    for (it_partial_tree.begin(level, z, x); it_partial_tree != it_partial_tree.end(); ++it_partial_tree) {

                        if(it_partial_tree != it_org_tree){
                            success = false;
                        }

                        if(it_org_tree.y() != it_partial_tree.y()){
                            success = false;
                        }

                        if(it_org_tree < it_org_tree.end()){
                            it_org_tree++;
                        }

                    }
                }
            }
        }


    }


    return success;

}





bool test_apr_neighbour_access(TestData& test_data){

    bool success = true;

    auto neighbour_iterator = test_data.apr.random_iterator();
    auto apr_iterator = test_data.apr.random_iterator();

    ParticleData<uint16_t> x_p(test_data.apr.total_number_particles());
    ParticleData<uint16_t> y_p(test_data.apr.total_number_particles());
    ParticleData<uint16_t> z_p(test_data.apr.total_number_particles());

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    x_p[apr_iterator] = apr_iterator.x();
                    y_p[apr_iterator] = apr_iterator.y();
                    z_p[apr_iterator] = apr_iterator.z();

                }
            }
        }
    }



    int number_directions;

    if(apr_iterator.number_dimensions() == 3){
        number_directions = 6;
    } else{
        std::cerr << "No current support for random neighbour access in 2D and 1D " << std::endl;
        return success;
    }

    uint64_t counter = 0;

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    counter++;

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < number_directions; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                        apr_iterator.find_neighbours_in_direction(direction);

                        success = check_neighbour_out_of_bounds(apr_iterator, direction);

                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                            // on each face, there can be 0-4 neighbours accessed by index
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //will return true if there is a neighbour defined
                                uint16_t apr_intensity = test_data.particles_intensities[neighbour_iterator];
                                uint16_t check_intensity = test_data.img_pc(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                            neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                            neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

//                                uint16_t x_n = x_p[neighbour_iterator];
//                                uint16_t y_n = y_p[neighbour_iterator];
//                                uint16_t z_n = z_p[neighbour_iterator];

                                if (check_intensity != apr_intensity) {
                                    success = false;
                                }

                                uint16_t apr_level = neighbour_iterator.level();
                                uint16_t check_level = test_data.img_level(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                           neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                           neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_level != apr_level) {
                                    success = false;
                                }

                                uint16_t apr_x = neighbour_iterator.x();
                                uint16_t check_x = test_data.img_x(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_x != apr_x) {
                                    success = false;
                                }

                                uint16_t apr_y = neighbour_iterator.y();
                                uint16_t check_y = test_data.img_y(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_y != apr_y) {
                                    success = false;
                                }

                                uint16_t apr_z = neighbour_iterator.z();
                                uint16_t check_z = test_data.img_z(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_z != apr_z) {
                                    success = false;
                                }

                                if (!check_neighbours(test_data.apr, apr_iterator, neighbour_iterator)) {
                                    success = false;
                                }

                            }
                        }
                    }
                }
            }
        }
    }

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator,neighbour_iterator)
#endif
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    //loop over all the neighbours and set the neighbour iterator to it
                    for (int direction = 0; direction < 6; ++direction) {
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        apr_iterator.find_neighbours_in_direction(direction);

                        success = check_neighbour_out_of_bounds(apr_iterator, direction);

                        for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {

                            // on each face, there can be 0-4 neighbours accessed by index
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //will return true if there is a neighbour defined
                                uint16_t apr_intensity = (test_data.particles_intensities[neighbour_iterator]);
                                uint16_t check_intensity = test_data.img_pc(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                            neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                            neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_intensity != apr_intensity) {
                                    success = false;
                                }

                                uint16_t apr_level = neighbour_iterator.level();
                                uint16_t check_level = test_data.img_level(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                           neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                           neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_level != apr_level) {
                                    success = false;
                                }


                                uint16_t apr_x = neighbour_iterator.x();
                                uint16_t check_x = test_data.img_x(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_x != apr_x) {
                                    success = false;
                                }

                                uint16_t apr_y = neighbour_iterator.y();
                                uint16_t check_y = test_data.img_y(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_y != apr_y) {
                                    success = false;
                                }

                                uint16_t apr_z = neighbour_iterator.z();
                                uint16_t check_z = test_data.img_z(neighbour_iterator.y_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.y()),
                                                                   neighbour_iterator.x_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.x()),
                                                                   neighbour_iterator.z_nearest_pixel(neighbour_iterator.level(),neighbour_iterator.z()));

                                if (check_z != apr_z) {
                                    success = false;
                                }

                            }
                        }
                    }
                }
            }
        }
    }




    return success;


}


bool test_particle_structures(TestData& test_data) {
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator
    //

    bool success = true;

    ParticleData<uint16_t> parts;

    parts.init(test_data.apr);

    APRTimer timer(true);

    auto lin_it = test_data.apr.iterator();

    timer.start_timer("LinearIteration - normal - OpenMP");

    auto counter_p = 0;

    for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
        int z = 0;

        for (z = 0; z < lin_it.z_num(level); z++) {
            for (int x = 0; x < lin_it.x_num(level); ++x) {
                for (lin_it.begin(level, z, x); lin_it < lin_it.end();
                     lin_it++) {
                    //need to add the ability to get y, and x,z but as possible should be lazy.
                    parts[lin_it] += 1;
                    counter_p++;

                }
            }
        }
    }


    timer.stop_timer();

    PartCellData<uint16_t> partCellData;
    partCellData.init(test_data.apr);

    auto pcd_size = partCellData.size();
    auto p_size = parts.size();

    if(pcd_size != p_size){
        success = false;
    }


    timer.start_timer("LinearIteration - PartCell - OpenMP");

    auto counter_pc = 0;

    for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
        int z = 0;

        for (z = 0; z < lin_it.z_num(level); z++) {
            for (int x = 0; x < lin_it.x_num(level); ++x) {
                for (auto begin = lin_it.begin(level, z, x); lin_it < lin_it.end();
                     lin_it++) {
                    //need to add the ability to get y, and x,z but as possible should be lazy.
                    auto off = z*lin_it.x_num(level) + x;
                    auto indx = lin_it - begin;
                    partCellData.data[level][off][indx] += 1;
                    if(partCellData.data[level][off][indx]!=parts[lin_it]){
                        success = false;
                    }

                    if(partCellData.data[level][off][indx]!=partCellData[lin_it]){
                        success = false;
                    }

                    counter_pc++;
                }
            }
        }
    }

    timer.stop_timer();

    if(counter_pc != counter_p){
        success = false;
    }

    /*
     * Test the level function
     *
     */

    ParticleData<uint8_t> level_p;
    PartCellData<uint8_t> level_pc;

    level_p.fill_with_levels(test_data.apr);
    level_pc.fill_with_levels(test_data.apr);

    for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
        int z = 0;

        for (z = 0; z < lin_it.z_num(level); z++) {
            for (int x = 0; x < lin_it.x_num(level); ++x) {
                for (lin_it.begin(level, z, x); lin_it < lin_it.end();
                     lin_it++) {

                    if(level_p[lin_it]!=level){
                        success = false;
                    }

                    if(level_pc[lin_it]!=level){
                        success = false;
                    }

                }
            }
        }
    }

    auto it_tree = test_data.apr.tree_iterator();

    ParticleData<uint8_t> level_p_tree;
    PartCellData<uint8_t> level_pc_tree;

    level_p_tree.fill_with_levels_tree(test_data.apr);
    level_pc_tree.fill_with_levels_tree(test_data.apr);

    for (int level = it_tree.level_min(); level <= it_tree.level_max(); ++level) {
        int z = 0;

        for (z = 0; z < it_tree.z_num(level); z++) {
            for (int x = 0; x < it_tree.x_num(level); ++x) {
                for (it_tree.begin(level, z, x); it_tree < it_tree.end();
                     it_tree++) {

                    if(level_p_tree[it_tree]!=level){
                        success = false;
                    }

                    if(level_pc_tree[it_tree]!=level){
                        success = false;
                    }

                }
            }
        }
    }


    return success;

}


bool test_linear_iterate(TestData& test_data) {
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator
    //

    bool success = true;

    auto it = test_data.apr.iterator();

    uint64_t particle_number = 0;

    uint64_t counter = 0;

    auto it_c = test_data.apr.random_iterator();

    //need to transfer the particles across


    ParticleData<uint16_t> parts;
    parts.init(test_data.apr.total_number_particles());

    uint64_t c_t = 0;
    for (int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {
                for (it_c.begin(level, z, x); it_c < it_c.end();
                     it_c++) {

                    parts[c_t] = test_data.particles_intensities[it_c];
                    c_t++;
                }
            }
        }
    }



    for (int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                it_c.begin(level, z, x);

                for (it.begin(level, z, x); it < it.end();
                     it++) {

                    uint16_t apr_intensity = (parts[it]);
                    uint16_t check_intensity = test_data.img_pc(it.y_nearest_pixel(level,it.y()),
                                                                it.x_nearest_pixel(level,x),
                                                                it.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                        particle_number++;
                    }

                    uint16_t apr_level = level;
                    uint16_t check_level = test_data.img_level(it.y_nearest_pixel(level,it.y()),
                                                               it.x_nearest_pixel(level,x),
                                                               it.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }


                    uint16_t apr_x = x;
                    uint16_t check_x = test_data.img_x(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = it.y();
                    uint16_t check_y = test_data.img_y(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = z;
                    uint16_t check_z = test_data.img_z(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_z != apr_z) {
                        success = false;
                    }

                    counter++;

                    if(it_c < it_c.end()){
                        it_c++;
                    }

                }
            }
        }

    }

    std::cout << particle_number << std::endl;

    //Test parallel loop

    for (int level = it.level_min(); level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(it)
#endif
        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level, z, x); it < it.end();
                     it++) {

                    uint16_t apr_intensity = (parts[it]);
                    uint16_t check_intensity = test_data.img_pc(it.y_nearest_pixel(level,it.y()),
                                                                it.x_nearest_pixel(level,x),
                                                                it.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                    }

                    uint16_t apr_level = level;
                    uint16_t check_level = test_data.img_level(it.y_nearest_pixel(level,it.y()),
                                                               it.x_nearest_pixel(level,x),
                                                               it.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }



                    uint16_t apr_x = x;
                    uint16_t check_x = test_data.img_x(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = it.y();
                    uint16_t check_y = test_data.img_y(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = z;
                    uint16_t check_z = test_data.img_z(it.y_nearest_pixel(level,it.y()),
                                                       it.x_nearest_pixel(level,x),
                                                       it.z_nearest_pixel(level,z));

                    if (check_z != apr_z) {
                        success = false;
                    }
                }
            }
        }

    }

    auto it_l = test_data.apr.iterator();

    for (int level = it_l.level_min(); level <= it_l.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it_l.z_num(level); z++) {
            for (x = 0; x < it_l.x_num(level); ++x) {

                for (it_l.begin(level, z, x); it_l < it_l.end();
                     it_l++) {

                    uint16_t apr_intensity = (parts[it_l]);
                    uint16_t check_intensity = test_data.img_pc(it_l.y_nearest_pixel(level, it_l.y()),
                                                                it_l.x_nearest_pixel(level, x),
                                                                it_l.z_nearest_pixel(level, z));

                    if(apr_intensity != check_intensity){
                        success = false;
                    }
                }
            }
        }
    }



    return success;


}


bool test_apr_random_iterate(TestData& test_data){
    //
    //  Bevan Cheeseman 2018
    //
    //  Test for the serial APR iterator
    //

    bool success = true;

    auto apr_iterator = test_data.apr.random_iterator();

    uint64_t particle_number = 0;

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.begin(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator++) {

                    uint16_t apr_intensity = (test_data.particles_intensities[apr_iterator]);
                    uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                                apr_iterator.x_nearest_pixel(level,x),
                                                                apr_iterator.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                        particle_number++;
                    }

                    uint16_t apr_level = apr_iterator.level();
                    uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                               apr_iterator.x_nearest_pixel(level,x),
                                                               apr_iterator.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }

                    uint16_t apr_x = apr_iterator.x();
                    uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = apr_iterator.y();
                    uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = apr_iterator.z();
                    uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_z != apr_z) {
                        success = false;
                    }
                }
            }
        }

    }

    std::cout << particle_number << std::endl;

    //Test parallel loop

    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    uint16_t apr_intensity = (test_data.particles_intensities[apr_iterator]);
                    uint16_t check_intensity = test_data.img_pc(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                                apr_iterator.x_nearest_pixel(level,x),
                                                                apr_iterator.z_nearest_pixel(level,z));

                    if (check_intensity != apr_intensity) {
                        success = false;
                    }

                    uint16_t apr_level = apr_iterator.level();
                    uint16_t check_level = test_data.img_level(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                               apr_iterator.x_nearest_pixel(level,x),
                                                               apr_iterator.z_nearest_pixel(level,z));

                    if (check_level != apr_level) {
                        success = false;
                    }



                    uint16_t apr_x = apr_iterator.x();
                    uint16_t check_x = test_data.img_x(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_x != apr_x) {
                        success = false;
                    }

                    uint16_t apr_y = apr_iterator.y();
                    uint16_t check_y = test_data.img_y(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_y != apr_y) {
                        success = false;
                    }

                    uint16_t apr_z = apr_iterator.z();
                    uint16_t check_z = test_data.img_z(apr_iterator.y_nearest_pixel(level,apr_iterator.y()),
                                                       apr_iterator.x_nearest_pixel(level,x),
                                                       apr_iterator.z_nearest_pixel(level,z));

                    if (check_z != apr_z) {
                        success = false;
                    }
                }
            }
        }

    }

    uint64_t counter = 0;
    for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < apr_iterator.z_num(level); z++) {
            for (x = 0; x < apr_iterator.x_num(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator < apr_iterator.end();
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    counter++;

                    if (apr_iterator.level() != level) {
                        //set all particles in calc_ex with an particle intensity greater then 100 to 0.
                        success = false;
                    }
                }
            }
        }
    }

    if(counter != apr_iterator.total_number_particles()){
        success = false;
    }



    return success;
}

bool test_pipeline_u16(TestData& test_data){
    //
    //  The purpose of this test is a full check that you get the same answer out.
    //
    //

    bool success = true;

    //the apr datastructure
    APR apr;
    APRConverter<uint16_t> aprConverter;

    APRParameters readPars = test_data.apr.get_apr_parameters();

    //read in the command line options into the parameters file
    aprConverter.par.Ip_th = 0;
    aprConverter.par.rel_error = readPars.rel_error;
    aprConverter.par.lambda = readPars.lambda;

    aprConverter.par.sigma_th_max = readPars.sigma_th_max;
    aprConverter.par.sigma_th = readPars.sigma_th;

    aprConverter.par.grad_th = readPars.grad_th;

    aprConverter.par.auto_parameters = false;

    //where things are
    aprConverter.par.input_image_name = test_data.filename;
    aprConverter.par.input_dir = "";
    aprConverter.par.name = test_data.output_name;
    aprConverter.par.output_dir = test_data.output_dir;

    aprConverter.par.output_steps = true;

    aprConverter.get_apr(apr,test_data.img_original);

    PixelData<float> scale_computed = TiffUtils::getMesh<float>(test_data.output_dir +"local_intensity_scale_step.tif");
    PixelData<uint16_t> gradient_computed = TiffUtils::getMesh<uint16_t>(test_data.output_dir + "gradient_step.tif");

    PixelData<float> scale_saved = TiffUtils::getMesh<float>(test_data.output_dir + "scale_saved.tif");
    PixelData<uint16_t> gradient_saved = TiffUtils::getMesh<uint16_t>(test_data.output_dir + "gradient_saved.tif");

    for (size_t i = 0; i < scale_computed.mesh.size(); ++i) {
        float computed_val = scale_computed.mesh[i];
        float saved_val = scale_saved.mesh[i];

        if(std::abs(computed_val - saved_val) > 1){
            success = false;
        }
    }

    for (size_t i = 0; i < gradient_computed.mesh.size(); ++i) {
        float computed_val = gradient_computed.mesh[i];
        float saved_val = gradient_saved.mesh[i];

        if(std::abs(computed_val - saved_val) > 1){
            success = false;
        }
    }

    APR apr_c;
    aprConverter.initPipelineAPR(apr_c, test_data.img_original);

    aprConverter.get_apr_custom_grad_scale(apr_c,gradient_saved,scale_saved);

    auto it_org = test_data.apr.iterator();
    auto it_gen = apr_c.iterator();

    //test the access
    success = compare_two_iterators(it_org,it_gen,success);

    ParticleData<uint16_t> particles_intensities;

    particles_intensities.sample_image(apr_c, test_data.img_original);

    //test the particles
    for (size_t j = 0; j < particles_intensities.size(); ++j) {
        if(particles_intensities.data[j] != test_data.particles_intensities.data[j]){
            success = false;
        }
    }


    return success;

}


/**
 * Extend value range to `uint16 max - bspline offset` and run pipeline test. If smoothing is used (lambda > 0), the
 * pipeline should throw.
 * @param test_data
 * @return true if lambda > 0 and overflow caught, or if lambda = 0 (bspline smoothing not used). Otherwise false.
 */
bool test_u16_overflow_detection(TestData& test_data) {

    if(test_data.apr.get_apr_parameters().lambda > 0) {
        try{
            const auto idx = test_data.img_original.mesh.size() / 2;
            test_data.img_original.mesh[idx] = 65535-100;
            test_pipeline_u16(test_data);
        } catch(std::invalid_argument&) {
            std::cout << "overflow successfully detected" << std::endl;
            return true;
        }
    } else {
        std::cout << "not testing overflow detection as smoothing is turned off (lambda = 0)" << std::endl;
        return true;
    }
    return false;
}


bool test_pipeline_u16_blocked(TestData& test_data) {

    /// Checks that the blocked pipeline (APRConverterBatch) and sampling give the same result as
    /// the standard methods, given enough ghost slices to cover the entire image (then the Bsplines,
    /// gradients and local scales should be identical)

    bool success = true;

    const int z_block_size = test_data.img_original.z_num / 4; // process the image in 4 blocks
    const int z_ghost = test_data.img_original.z_num;          // use the entire image for the computations

    //the apr datastructure
    APR apr;
    APR aprBatch;
    APRConverterBatch<uint16_t> converterBatch;
    APRConverter<uint16_t> converter;

    converterBatch.set_generate_linear(true);
    converterBatch.set_sparse_pulling_scheme(false);
    converter.set_generate_linear(true);
    converter.set_sparse_pulling_scheme(false);

    APRParameters readPars = test_data.apr.get_apr_parameters();
    readPars.input_image_name = test_data.filename;
    readPars.input_dir = "";
    readPars.name = test_data.output_name;
    readPars.output_dir = test_data.output_dir;
    readPars.auto_parameters = false;
    readPars.output_steps = false;
    readPars.neighborhood_optimization = true;

    converter.par = readPars;
    converterBatch.par = readPars;

    converterBatch.z_block_size = z_block_size;
    converterBatch.ghost_z = z_ghost;

    converter.get_apr(apr, test_data.img_original);
    converterBatch.get_apr(aprBatch);

    if(aprBatch.total_number_particles() != apr.total_number_particles()){
        std::cout << "Number of particles do not match! Expected " << apr.total_number_particles() <<
                  " but received " << aprBatch.total_number_particles() << std::endl;

        return false;
    }

    ParticleData<uint16_t> parts;
    ParticleData<uint16_t> partsBatch;

    parts.sample_image(apr, test_data.img_original);
    partsBatch.sample_parts_from_img_blocked(aprBatch, test_data.filename, z_block_size, z_ghost);

    for(size_t i = 0; i < apr.total_number_particles(); ++i) {
        if(parts[i] != partsBatch[i]) {
            success = false;
        }
    }
    return success;
}

bool test_pipeline_bound(TestData& test_data,float rel_error){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    bool success = true;

    //the apr datastructure
    APR apr;
    APRConverter<uint16_t> aprConverter;

    //read in the command line options into the parameters file
    aprConverter.par.Ip_th = 0;
    aprConverter.par.rel_error = rel_error;
    aprConverter.par.lambda = 0;
    aprConverter.par.mask_file = "";

    aprConverter.par.sigma_th_max = 50;
    aprConverter.par.sigma_th = 100;

    aprConverter.par.auto_parameters = false;

    aprConverter.par.output_steps = true;

    //where things are
    aprConverter.par.input_image_name = test_data.filename;
    aprConverter.par.input_dir = "";
    aprConverter.par.name = test_data.output_name;
    aprConverter.par.output_dir = test_data.output_dir;

    //Gets the APR

    ParticleData<uint16_t> particles_intensities;

    aprConverter.get_apr(apr,test_data.img_original);

    particles_intensities.sample_image(apr, test_data.img_original);

    PixelData<uint16_t> pc_recon;
    APRReconstruction::reconstruct_constant(apr,pc_recon,particles_intensities);

    //read in used scale

    PixelData<float> scale = TiffUtils::getMesh<float>(test_data.output_dir + "local_intensity_scale_step.tif");

    TestBenchStats benchStats;

    benchStats =  compare_gt(test_data.img_original,pc_recon,scale);

    std::cout << "Inf norm: " << benchStats.inf_norm << " Bound: " << rel_error << std::endl;

    if(benchStats.inf_norm > rel_error){
        success = false;
    }

    return success;
}

bool test_pipeline_bound_blocked(TestData& test_data, float rel_error){

    /// Checks the reconstruction condition (for piecewise constant reconstruction) for the blocked pipeline
    /// using a local scale computed by the standard APR pipeline

    APRParameters par;
    par.rel_error = rel_error;
    par.sigma_th_max = 50;
    par.sigma_th = 100;
    par.lambda = 0;
    par.Ip_th = 0;
    par.auto_parameters = false;
    par.output_steps = false;
    par.neighborhood_optimization = true;

    //where things are
    par.input_image_name = test_data.filename;
    par.input_dir = "";
    par.name = test_data.output_name;
    par.output_dir = test_data.output_dir;

    // standard APR pipeline with output steps, to get the local intensity scale
    APRConverter<uint16_t> converter;
    converter.par = par;
    converter.par.output_steps = true;

    APR apr;
    converter.get_apr(apr, test_data.img_original);

    // batch APR converter for blocked conversion
    APRConverterBatch<uint16_t> converterBatch;
    converterBatch.par = par;
    converterBatch.z_block_size = 32;
    converterBatch.ghost_z = 16;

    // Get the APR by block
    APR aprBatch;
    converterBatch.get_apr(aprBatch);

    // Sample particles by block
    ParticleData<uint16_t> particles_intensities;
    particles_intensities.sample_parts_from_img_blocked(aprBatch, test_data.filename, 32, 32);

    // Piecewise constant reconstruction
    PixelData<uint16_t> pc_recon;
    APRReconstruction::reconstruct_constant(aprBatch,pc_recon,particles_intensities);

    //read in intensity scale computed for the entire image by the standard converter
    PixelData<float> scale = TiffUtils::getMesh<float>(test_data.output_dir + "local_intensity_scale_step.tif");

    TestBenchStats benchStats;
    benchStats =  compare_gt(test_data.img_original, pc_recon, scale);

    std::cout << "Inf norm: " << benchStats.inf_norm << " Bound: " << rel_error << std::endl;

    return benchStats.inf_norm <= rel_error;
}


bool test_reconstruct_level(TestData &test_data) {

    PixelData<uint8_t> level_recon;
    APRReconstruction::reconstruct_level(test_data.apr, level_recon);

    auto apr_it = test_data.apr.iterator();
    VectorData<uint8_t> level_parts;
    level_parts.resize((apr_it.total_number_particles()));

    for(uint8_t level = apr_it.level_min(); level <= apr_it.level_max(); ++level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(level, level_parts) firstprivate(apr_it) collapse(2)
#endif
        for(int z = 0; z < apr_it.z_num(level); ++z) {
            for(int x = 0; x < apr_it.x_num(level); ++x) {
                for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
                    level_parts[apr_it] = level;
                }
            }
        }
    }

    PixelData<uint8_t> level_recon_gt;
    APRReconstruction::reconstruct_constant(test_data.apr, level_recon_gt, level_parts);

    for(uint64_t idx = 0; idx < level_recon.mesh.size(); ++idx) {
        if(level_recon.mesh[idx] != level_recon_gt.mesh[idx]) {
            std::cout << "reconstruct_level failed at index " << idx << ". Expected " << level_recon_gt.mesh[idx] <<
                         " but got " << level_recon.mesh[idx] << std::endl;
            return false;
        }
    }
    return true;
}



bool test_reconstruct_patch(TestData &test_data, const int level_delta = 0) {

    ReconPatch patch;
    patch.z_begin = 13;
    patch.z_end = 119;
    patch.x_begin = 21;
    patch.x_end = 58;
    patch.y_begin = 24;
    patch.y_end = 104;
    patch.level_delta = level_delta;

    APRTimer timer(true);
    PixelData<float> recon_patch;
    PixelData<float> recon_full;

    ParticleData<float> tree_data;
    APRTreeNumerics::fill_tree_mean(test_data.apr, test_data.particles_intensities, tree_data);

    /// full reconstruction
    APRReconstruction::reconstruct_constant(test_data.apr, recon_full, test_data.particles_intensities);

    if (level_delta < 0) {
        std::vector<PixelData<float>> img_pyramid;
        const int level = test_data.apr.level_max() + patch.level_delta;
        downsamplePyramid(recon_full, img_pyramid, test_data.apr.level_max(), level);
        recon_full.swap(img_pyramid[level]);
    }

    /// patch reconstruction
    APRReconstruction::reconstruct_constant(test_data.apr, recon_patch, test_data.particles_intensities, tree_data, patch);

    size_t failures = 0;
    float tol = 0.001;
    size_t max_print = 10;

    for(int z = patch.z_begin; z < patch.z_end; ++z) {
        for(int x = patch.x_begin; x < patch.x_end; ++x) {
            for(int y = patch.y_begin; y < patch.y_end; ++y) {
                float gt = recon_full.at(y, x, z);
                float est = recon_patch.at(y-patch.y_begin, x-patch.x_begin, z-patch.z_begin);

                if(std::abs(gt-est) > tol) {
                    if(failures < max_print) {
                        std::cout << "Expected " << gt << " but received " << est << " at (z,x,y) = (" << z << ", " << x << ", " << y << ")" << std::endl;
                    }
                    failures++;
                }
            }
        }
    }

    return failures == 0;
}


bool test_reconstruct_patch_smooth(TestData &test_data) {

    ReconPatch patch;
    patch.z_begin = 13;
    patch.z_end = 119;
    patch.x_begin = 21;
    patch.x_end = 58;
    patch.y_begin = 24;
    patch.y_end = 104;
    patch.level_delta = 0;

    APRTimer timer(true);
    PixelData<float> recon_patch;
    PixelData<float> recon_full;

    /// full reconstruction
    APRReconstruction::reconstruct_smooth(test_data.apr, recon_full, test_data.particles_intensities);

    /// patch reconstruction
    APRReconstruction::reconstruct_smooth(test_data.apr, recon_patch, test_data.particles_intensities, patch);

    size_t failures = 0;
    float tol = 0.1;
    size_t max_print = 4;
    const int offset = 8; // values may be different at boundaries
    for(int z = patch.z_begin+offset; z < patch.z_end-offset; ++z) {
        for(int x = patch.x_begin+offset; x < patch.x_end-offset; ++x) {
            for(int y = patch.y_begin+offset; y < patch.y_end-offset; ++y) {
                float gt = recon_full.at(y, x, z);
                float est = recon_patch.at(y-patch.y_begin, x-patch.x_begin, z-patch.z_begin);

                if(std::abs(gt-est) > tol) {
                    if(failures < max_print) {
                        std::cout << "Expected " << gt << " but received " << est << " at (z,x,y) = (" << z << ", " << x << ", " << y << ")" << std::endl;
                    }
                    failures++;
                }
            }
        }
    }

    const size_t tot_compared = (recon_patch.z_num - 2*offset) * (recon_patch.x_num - 2*offset) * (recon_patch.y_num - 2*offset);
    std::cout << "test_reconstruct_patch_smooth: " << failures << " failures out of " << tot_compared << std::endl;

    return failures == 0;
}





bool test_convolve_pencil(TestData &test_data, const bool boundary = false, const std::vector<int>& stencil_size = {3, 3, 3}) {

    auto it = test_data.apr.iterator();

    PixelData<double> stenc(stencil_size[0], stencil_size[1], stencil_size[2]);

    double sz = stenc.mesh.size();
    double sum = sz * (sz-1)/2;
    for(size_t i = 0; i < stenc.mesh.size(); ++i){
        stenc.mesh[i] = i / sum;
    }

    std::vector<PixelData<double>> stencils;
    APRStencil::get_downsampled_stencils(stenc, stencils, it.level_max()-it.level_min(), true);

    ParticleData<double> output;
    APRFilter::convolve_pencil(test_data.apr, stencils, test_data.particles_intensities, output, boundary);

    ParticleData<double> output_gt;
    APRFilter::create_test_particles_equiv(test_data.apr, stencils, test_data.particles_intensities, output_gt, boundary);


    if(output.size() != output_gt.size()) {
        std::cerr << "output sizes differ" << std::endl;
        return false;
    }

    double eps = 1e-2;
    size_t failures = 0;

    for(uint64_t x=0; x < output.size(); ++x) {
        if(std::abs(output[x] - output_gt[x]) > eps) {
            std::cout << "discrepancy of " << std::abs(output[x] - output_gt[x]) << " at particle " << x << " (output = " << output[x] << ", ground_truth = " << output_gt[x] << ")" << std::endl;
            failures++;
        }
    }
    std::cout << failures << " failures out of " << it.total_number_particles() << std::endl;
    return (failures==0);
}


bool test_convolve(TestData &test_data, const bool boundary = false, const std::vector<int>& stencil_size = {3, 3, 3}) {

    auto it = test_data.apr.iterator();

    PixelData<double> stenc(stencil_size[0], stencil_size[1], stencil_size[2]);

    double sz = stenc.mesh.size();
    double sum = sz * (sz-1)/2;
    for(size_t i = 0; i < stenc.mesh.size(); ++i){
        stenc.mesh[i] = i / sum;
    }

    std::vector<PixelData<double>> stencils;
    APRStencil::get_downsampled_stencils(stenc, stencils, it.level_max()-it.level_min(), true);

    ParticleData<double> output;
    APRFilter::convolve(test_data.apr, stencils, test_data.particles_intensities, output, boundary);

    ParticleData<double> output_gt;
    APRFilter::create_test_particles_equiv(test_data.apr, stencils, test_data.particles_intensities, output_gt, boundary);

    if(output.size() != output_gt.size()) {
        std::cout << "output sizes differ" << std::endl;
        return false;
    }

    double eps = 1e-2;
    uint64_t failures = 0;

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(std::abs(output[it] - output_gt[it]) > eps) {
                        std::cout << "Expected " << output_gt[it] << " but received " << output[it] <<
                                  " at particle index " << it << " (level, z, x, y) = (" << level << ", " << z << ", " << x << ", " << it.y() << ")" << std::endl;
                        failures++;
                    }
                }
            }
        }
    }
    std::cout << failures << " failures out of " << it.total_number_particles() << std::endl;
    return (failures==0);
}


bool test_iterator_methods(TestData &test_data){
    //
    //  Testing consistency across iterators for methods
    //
    //


    bool success = true;

    auto it_lin = test_data.apr.iterator();
    auto it_tree = test_data.apr.tree_iterator();

    auto it_rand = test_data.apr.random_iterator();
    auto it_tree_rand = test_data.apr.random_tree_iterator();

    //regular iterators
    for (int i = it_lin.level_min(); i <= it_lin.level_max(); ++i) {

        uint64_t lin_begin = it_lin.particles_level_begin(i);
        uint64_t rand_beign = it_rand.particles_level_begin(i);

        uint64_t begin = it_lin.begin(i,0,0);

        if(it_rand.particles_level_end(i) != it_lin.particles_level_end(i)){
            success = false;
        }


        if(lin_begin != rand_beign){
            success = false;
        }

        if(lin_begin != begin){
            success = false;
        }

        if(it_rand.x_num(i) != it_lin.x_num(i)){
            success = false;
        }

        if(it_rand.y_num(i) != it_lin.y_num(i)){
            success = false;
        }

        if(it_rand.z_num(i) != it_lin.z_num(i)){
            success = false;
        }

    }

    //regular iterators
    for (int i = it_tree.level_min(); i <= it_tree.level_max(); ++i) {

        uint64_t lin_begin = it_tree.particles_level_begin(i);
        uint64_t rand_begin = it_tree_rand.particles_level_begin(i);

        uint64_t begin = it_tree.begin(i,0,0);

        if(it_tree_rand.particles_level_end(i) != it_tree.particles_level_end(i)){
            success = false;
        }

        if(lin_begin != rand_begin){
            success = false;
        }

        if(lin_begin != begin){
            success = false;
        }


        if(it_tree_rand.x_num(i) != it_tree.x_num(i)){
            success = false;
        }

        if(it_tree_rand.y_num(i) != it_tree.y_num(i)){
            success = false;
        }

        if(it_tree_rand.z_num(i) != it_tree.z_num(i)){
            success = false;
        }
    }

    return success;

}

bool test_apr_copy(TestData &test_data){

    bool success = true;

    APR aprCopy(test_data.apr);

    auto it_org = test_data.apr.iterator();
    auto it_copy = aprCopy.iterator();

    success = compare_two_iterators(it_org,it_copy,success);

    return success;
}





void CreateGTSmallTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_small/";
}

void CreateGTSmall2DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_2D/";
}

void CreateGTSmall1DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_1D/";
}

void CreateBigBigData::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/large_low_cr/large_low_cr.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"parts",test_data.particles_intensities);
    aprFile.close();
}



void CreateSmallSphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_120/sphere_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_120/sphere_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name,false);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
    test_data.output_name = "sphere_small";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_120/";
}

void CreateDiffDimsSphereTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name,false);

    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name,false);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_original.tif";
    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_diff_dims/";
    test_data.output_name = "sphere_210";
}

void Create210SphereTestAPROnly::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_apr.h5";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_210/sphere_original.tif";
    test_data.output_name = "sphere_210";
}


void CreateGTSmall2DTestProperties::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name,false);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D_original.tif";
    test_data.output_name = "sphere_2D";
    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_2D/";
}

void CreateGTSmall1DTestProperties::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name,false);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D_original.tif";
    test_data.output_name = "sphere_1D";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_1D/";
}

//1D tests

#ifndef APR_USE_CUDA




TEST_F(CreateGTSmall1DTestProperties, APR_ITERATION) {

//test iteration
    ASSERT_TRUE(test_apr_random_iterate(test_data));
    ASSERT_TRUE(test_linear_iterate(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, PULLING_SCHEME_SPARSE) {
    //tests the linear access geneartions and io

    ASSERT_TRUE(test_pulling_scheme_sparse(test_data));


}

TEST_F(CreateGTSmall1DTestProperties, LINEAR_ACCESS_CREATE) {

    ASSERT_TRUE(test_linear_access_create(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, LINEAR_ACCESS_IO) {

    ASSERT_TRUE(test_linear_access_io(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, PARTIAL_READ) {

    ASSERT_TRUE(test_read_upto_level(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, LAZY_PARTICLES) {

    ASSERT_TRUE(test_lazy_particles(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, COMPRESS_PARTICLES) {

    ASSERT_TRUE(test_particles_compress(test_data));

}


TEST_F(CreateGTSmall1DTestProperties, RANDOM_ACCESS) {

    ASSERT_TRUE(test_random_access_it(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, APR_NEIGHBOUR_ACCESS) {

    //test iteration
    ASSERT_TRUE(test_apr_neighbour_access(test_data));

}


TEST_F(CreateGTSmall1DTestProperties, APR_TREE) {

    //test iteration
    ASSERT_TRUE(test_apr_tree(test_data));

}



TEST_F(CreateGTSmall1DTestProperties, APR_INPUT_OUTPUT) {

    //test iteration
    // ASSERT_TRUE(test_apr_input_output(test_data));

    ASSERT_TRUE(test_apr_file(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, APR_PARTICLES) {

    ASSERT_TRUE(test_particle_structures(test_data));
}

TEST_F(CreateGTSmall1DTestProperties, APR_FILTER) {
    ASSERT_TRUE(test_convolve(test_data, false, {5, 1, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, false, {5, 1, 1}));

}

TEST_F(CreateGTSmall1DTestProperties, PIPELINE_COMPARE) {

    ASSERT_TRUE(test_pipeline_u16(test_data));
    ASSERT_TRUE(test_u16_overflow_detection(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, ITERATOR_METHODS) {

    ASSERT_TRUE(test_iterator_methods(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, TEST_COPY) {
    ASSERT_TRUE(test_apr_copy(test_data));
}



//2D tests

TEST_F(CreateGTSmall2DTestProperties, APR_ITERATION) {

//test iteration
    ASSERT_TRUE(test_apr_random_iterate(test_data));
    ASSERT_TRUE(test_linear_iterate(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, PULLING_SCHEME_SPARSE) {
    //tests the linear access geneartions and io
    ASSERT_TRUE(test_pulling_scheme_sparse(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, LINEAR_ACCESS_CREATE) {

    ASSERT_TRUE(test_linear_access_create(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, LINEAR_ACCESS_IO) {

    ASSERT_TRUE(test_linear_access_io(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, PARTIAL_READ) {

    ASSERT_TRUE(test_read_upto_level(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, LAZY_PARTICLES) {

    ASSERT_TRUE(test_lazy_particles(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, COMPRESS_PARTICLES) {

    ASSERT_TRUE(test_particles_compress(test_data));

}


TEST_F(CreateGTSmall2DTestProperties, RANDOM_ACCESS) {

    ASSERT_TRUE(test_random_access_it(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, APR_NEIGHBOUR_ACCESS) {

    //test iteration
    ASSERT_TRUE(test_apr_neighbour_access(test_data));

}


TEST_F(CreateGTSmall2DTestProperties, APR_TREE) {

    //test iteration
    ASSERT_TRUE(test_apr_tree(test_data));

}



TEST_F(CreateGTSmall2DTestProperties, APR_INPUT_OUTPUT) {

    //test iteration
    // ASSERT_TRUE(test_apr_input_output(test_data));

    ASSERT_TRUE(test_apr_file(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, APR_PARTICLES) {

    ASSERT_TRUE(test_particle_structures(test_data));
}

TEST_F(CreateGTSmall2DTestProperties, APR_FILTER) {
    ASSERT_TRUE(test_convolve(test_data, false, {5, 5, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, false, {5, 5, 1}));
}

TEST_F(CreateGTSmall2DTestProperties, PIPELINE_COMPARE) {
    ASSERT_TRUE(test_pipeline_u16(test_data));
    ASSERT_TRUE(test_u16_overflow_detection(test_data));
}

TEST_F(CreateGTSmall2DTestProperties, ITERATOR_METHODS) {
    ASSERT_TRUE(test_iterator_methods(test_data));
}


TEST_F(CreateGTSmall2DTestProperties, TEST_COPY) {
    ASSERT_TRUE(test_apr_copy(test_data));
}


#endif

//3D Big Big Data


TEST_F(CreateBigBigData, LINEAR_ACCESS_IO) {

    ASSERT_TRUE(test_linear_access_io(test_data));

}


TEST_F(CreateBigBigData, PARTIAL_READ) {

    ASSERT_TRUE(test_read_upto_level(test_data));

}

TEST_F(CreateBigBigData, LAZY_PARTICLES) {

    ASSERT_TRUE(test_lazy_particles(test_data));

}

TEST_F(CreateBigBigData, COMPRESS_PARTICLES) {

    ASSERT_TRUE(test_particles_compress(test_data));

}


TEST_F(CreateBigBigData, RANDOM_ACCESS) {

    ASSERT_TRUE(test_random_access_it(test_data));

}

TEST_F(CreateBigBigData, APR_PARTICLES) {

    ASSERT_TRUE(test_particle_structures(test_data));
}

TEST_F(CreateBigBigData, ITERATOR_METHODS) {

    ASSERT_TRUE(test_iterator_methods(test_data));

}




//3D tests

TEST_F(CreateSmallSphereTest, ITERATOR_METHODS) {
    ASSERT_TRUE(test_iterator_methods(test_data));
}

TEST_F(CreateSmallSphereTest, AUTO_PARAMETERS) {

//test iteration
ASSERT_TRUE(test_auto_parameters(test_data));

}

TEST_F(CreateDiffDimsSphereTest, AUTO_PARAMETERS) {

//test iteration
ASSERT_TRUE(test_auto_parameters(test_data));

}

TEST_F(CreateGTSmall2DTestProperties, AUTO_PARAMETERS) {

//test iteration
ASSERT_TRUE(test_auto_parameters(test_data));

}

TEST_F(CreateGTSmall1DTestProperties, AUTO_PARAMETERS) {

//test iteration
    ASSERT_TRUE(test_auto_parameters(test_data));

}


TEST_F(CreateSmallSphereTest, APR_ITERATION) {

//test iteration
ASSERT_TRUE(test_apr_random_iterate(test_data));
ASSERT_TRUE(test_linear_iterate(test_data));

}

TEST_F(CreateSmallSphereTest, TEST_COPY) {

//test iteration
    ASSERT_TRUE(test_apr_copy(test_data));

}

TEST_F(CreateDiffDimsSphereTest, PULLING_SCHEME_SPARSE) {
    //tests the linear access geneartions and io
    ASSERT_TRUE(test_pulling_scheme_sparse(test_data));

}

TEST_F(CreateDiffDimsSphereTest, ITERATOR_METHODS) {

    ASSERT_TRUE(test_iterator_methods(test_data));

}

TEST_F(CreateSmallSphereTest, PULLING_SCHEME_SPARSE) {

    ASSERT_TRUE(test_pulling_scheme_sparse(test_data));

}

TEST_F(CreateSmallSphereTest, LINEAR_ACCESS_CREATE) {

    ASSERT_TRUE(test_linear_access_create(test_data));

}

TEST_F(CreateSmallSphereTest, LINEAR_ACCESS_IO) {

    ASSERT_TRUE(test_linear_access_io(test_data));

}

TEST_F(CreateSmallSphereTest, TEST_RECONSTRUCT) {
    ASSERT_TRUE(test_reconstruct_patch(test_data, 0));
    ASSERT_TRUE(test_reconstruct_patch(test_data, -1));
    ASSERT_TRUE(test_reconstruct_patch(test_data, -2));

    ASSERT_TRUE(test_reconstruct_patch_smooth(test_data));
}

TEST_F(CreateDiffDimsSphereTest, TEST_RECONSTRUCT) {
    ASSERT_TRUE(test_reconstruct_patch(test_data, 0));
    ASSERT_TRUE(test_reconstruct_patch(test_data, -1));
    ASSERT_TRUE(test_reconstruct_patch(test_data, -2));

    ASSERT_TRUE(test_reconstruct_patch_smooth(test_data));
}


TEST_F(CreateDiffDimsSphereTest, LINEAR_ACCESS_CREATE) {

    ASSERT_TRUE(test_linear_access_create(test_data));

}

TEST_F(CreateDiffDimsSphereTest, LINEAR_ACCESS_IO) {

    ASSERT_TRUE(test_linear_access_io(test_data));

}

TEST_F(CreateSmallSphereTest, PARTIAL_READ) {

    ASSERT_TRUE(test_read_upto_level(test_data));

}


TEST_F(CreateDiffDimsSphereTest, PARTIAL_READ) {

    ASSERT_TRUE(test_read_upto_level(test_data));

}

TEST_F(CreateSmallSphereTest, LAZY_PARTICLES) {

    ASSERT_TRUE(test_lazy_particles(test_data));

}

TEST_F(CreateDiffDimsSphereTest, LAZY_PARTICLES) {

    ASSERT_TRUE(test_lazy_particles(test_data));

}

TEST_F(CreateSmallSphereTest, COMPRESS_PARTICLES) {

    ASSERT_TRUE(test_particles_compress(test_data));

}

TEST_F(CreateDiffDimsSphereTest, COMPRESS_PARTICLES) {

    ASSERT_TRUE(test_particles_compress(test_data));

}


TEST_F(CreateSmallSphereTest, RANDOM_ACCESS) {

    ASSERT_TRUE(test_random_access_it(test_data));

}

TEST_F(CreateDiffDimsSphereTest, RANDOM_ACCESS) {

    ASSERT_TRUE(test_random_access_it(test_data));

}


TEST_F(CreateDiffDimsSphereTest, RECONSTRUCT_LEVEL) {

    ASSERT_TRUE(test_reconstruct_level(test_data));

}


TEST_F(CreateDiffDimsSphereTest, APR_PIPELINE_3D_BLOCKED) {
    //test blocked pipeline for different error thresholds (E)
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.001));
}


#ifndef APR_USE_CUDA

TEST_F(CreateSmallSphereTest, PIPELINE_SIZE) {

//#TODO: need to explore these and add them back in, one seems to have some memory errors.

//    ASSERT_TRUE(test_symmetry_pipeline());
//
//    ASSERT_TRUE(test_pipeline_different_sizes(test_data));
//
//    ASSERT_TRUE(test_pipeline_mask(test_data));

}

#endif


TEST_F(CreateSmallSphereTest, APR_TREE) {

    //test iteration
    ASSERT_TRUE(test_apr_tree(test_data));

}

TEST_F(CreateSmallSphereTest, APR_NEIGHBOUR_ACCESS) {

    //test iteration
    ASSERT_TRUE(test_apr_neighbour_access(test_data));

}

TEST_F(CreateSmallSphereTest, APR_INPUT_OUTPUT) {
    //test iteration
   // ASSERT_TRUE(test_apr_input_output(test_data));
    ASSERT_TRUE(test_apr_file(test_data));
}

TEST_F(CreateSmallSphereTest, APR_PIPELINE_3D_BLOCKED) {
    //test blocked pipeline for different error thresholds (E)
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.001));
}


TEST_F(CreateGTSmallTest, APR_PIPELINE_3D) {

//test pipeline
    ASSERT_TRUE(test_pipeline_bound(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.001));

}


TEST_F(CreateGTSmallTest, APR_PIPELINE_3D_BLOCKED) {
    //test blocked pipeline for different error thresholds (E)
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound_blocked(test_data,0.001));
}


#ifndef APR_USE_CUDA

TEST_F(CreateGTSmall2DTest, APR_PIPELINE_2D) {

//test pipeline
    ASSERT_TRUE(test_pipeline_bound(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.001));

}

TEST_F(CreateGTSmall1DTest, APR_PIPELINE_1D) {

//test pipeline
    ASSERT_TRUE(test_pipeline_bound(test_data,0.2));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.1));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.01));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.05));
    ASSERT_TRUE(test_pipeline_bound(test_data,0.001));

}

#endif

TEST_F(CreateDiffDimsSphereTest, APR_ITERATION) {

//test iteration
    ASSERT_TRUE(test_apr_random_iterate(test_data));
    ASSERT_TRUE(test_linear_iterate(test_data));

}

TEST_F(CreateDiffDimsSphereTest, APR_TREE) {

//test iteration
    ASSERT_TRUE(test_apr_tree(test_data));

}

TEST_F(CreateDiffDimsSphereTest, APR_NEIGHBOUR_ACCESS) {

//test iteration
    ASSERT_TRUE(test_apr_neighbour_access(test_data));

}


TEST_F(CreateDiffDimsSphereTest, APR_PARTICLES) {

    ASSERT_TRUE(test_particle_structures(test_data));
}

TEST_F(CreateSmallSphereTest, APR_PARTICLES) {

    ASSERT_TRUE(test_particle_structures(test_data));
}


TEST_F(CreateDiffDimsSphereTest, APR_INPUT_OUTPUT) {

    ASSERT_TRUE(test_apr_file(test_data));

}

TEST_F(CreateSmallSphereTest, PIPELINE_COMPARE) {

    ASSERT_TRUE(test_pipeline_u16(test_data));
    ASSERT_TRUE(test_u16_overflow_detection(test_data));

}

TEST_F(CreateDiffDimsSphereTest, PIPELINE_COMPARE) {

    ASSERT_TRUE(test_pipeline_u16(test_data));
    ASSERT_TRUE(test_u16_overflow_detection(test_data));

}

TEST_F(CreateSmallSphereTest, PIPELINE_COMPARE_BLOCKED) {
    ASSERT_TRUE(test_pipeline_u16_blocked(test_data));
}


TEST_F(CreateSmallSphereTest, RUN_RICHARDSON_LUCY) {

    auto stenc = APRStencil::create_mean_filter<float>({5, 5, 5});

    ParticleData<float> output;
    APRNumerics::richardson_lucy(test_data.apr, test_data.particles_intensities, output, stenc, 10, true, true, false);
}


TEST_F(CreateSmallSphereTest, CHECK_DOWNSAMPLE_STENCIL) {

    PixelData<float> stencil_pd(5, 5, 5);
    VectorData<float> stencil_vd;
    stencil_vd.resize(125);

    float sum = 62.0f * 125.0f;
    for(int i = 0; i < 125; ++i) {
        stencil_pd.mesh[i] = ((float) i) / sum;
        stencil_vd[i] = ((float) i) / sum;
    }

    VectorData<float> stencil_vec_vd;
    VectorData<float> stencil_vec_pd;
    std::vector<PixelData<float>> pd_vec;

    int nlevels = 7;

    APRStencil::get_downsampled_stencils(stencil_pd, stencil_vec_pd, nlevels, false);
    APRStencil::get_downsampled_stencils(stencil_pd, pd_vec, nlevels, false);
    APRStencil::get_downsampled_stencils(stencil_vd, stencil_vec_vd, nlevels, false);

    // compare outputs for PixelData and VectorData inputs
    bool success = true;
    ASSERT_EQ(stencil_vec_vd.size(), stencil_vec_pd.size());

    std::cout << "comparing downsampled stencils for VectorData and PixelData inputs" << std::endl;

    for(size_t i = 0; i < stencil_vec_pd.size(); ++i) {
        if( std::abs( stencil_vec_pd[i] - stencil_vec_vd[i] ) > 1e-5 ) {
            std::cout << "stencil_vec_vd = " << stencil_vec_vd[i] << " stencil_vec_pd = " << stencil_vec_pd[i] << " at index " << i << std::endl;
            success = false;
        }
    }

    if(success) {
        std::cout << "OK!" << std::endl;
    }

    std::cout << "comparing downsampeld stencils for VectorData and std::vector<PixelData> output" << std::endl;
    success = true;
    int c = 0;
    for(size_t dlvl = 0; dlvl < pd_vec.size(); ++dlvl) {
        for(size_t i = 0; i < pd_vec[dlvl].mesh.size(); ++i) {
            if( std::abs( pd_vec[dlvl].mesh[i] - stencil_vec_pd[c] ) > 1e-5 ) {
                std::cout << "pd_vec = " << pd_vec[dlvl].mesh[i] << " stencil_vec_pd = " << stencil_vec_pd[c] <<
                          " at dlvl = " << dlvl << " and i = " << i << std::endl;
                success = false;
            }
            c++;
        }
    }

    ASSERT_EQ(c, stencil_vec_pd.size());
    if(success) {
        std::cout << "OK!" << std::endl;
    }

}


TEST_F(CreateSmallSphereTest, APR_FILTER) {
    // 3D filters
    ASSERT_TRUE(test_convolve(test_data, false, {7, 7, 7}));
    ASSERT_TRUE(test_convolve(test_data, true, {7, 7, 7}));
    ASSERT_TRUE(test_convolve(test_data, false, {3, 3, 3}));
    ASSERT_TRUE(test_convolve(test_data, true, {3, 3, 3}));

    ASSERT_TRUE(test_convolve_pencil(test_data, false, {9, 9, 9}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {9, 9, 9}));
    ASSERT_TRUE(test_convolve_pencil(test_data, false, {3, 3, 3}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {3, 3, 3}));

    // 2D filters
    ASSERT_TRUE(test_convolve(test_data, true, {13, 13, 1}));
    ASSERT_TRUE(test_convolve(test_data, true, {13, 1, 13}));
    ASSERT_TRUE(test_convolve(test_data, true, {1, 13, 13}));

    ASSERT_TRUE(test_convolve_pencil(test_data, true, {13, 13, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {13, 1, 13}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {1, 13, 13}));

    // 1D filters
    ASSERT_TRUE(test_convolve(test_data, true, {17, 1, 1}));
    ASSERT_TRUE(test_convolve(test_data, true, {1, 17, 1}));
    ASSERT_TRUE(test_convolve(test_data, true, {1, 1, 17}));

    ASSERT_TRUE(test_convolve_pencil(test_data, true, {17, 1, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {1, 17, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {1, 1, 17}));
}


TEST_F(CreateDiffDimsSphereTest, APR_FILTER) {

    // 3D filters
    ASSERT_TRUE(test_convolve(test_data, false, {13, 13, 13}));
    ASSERT_TRUE(test_convolve(test_data, true, {13, 13, 13}));
    ASSERT_TRUE(test_convolve(test_data, false, {3, 3, 3}));
    ASSERT_TRUE(test_convolve(test_data, true, {3, 3, 3}));

    ASSERT_TRUE(test_convolve_pencil(test_data, false, {13, 13, 13}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {13, 13, 13}));
    ASSERT_TRUE(test_convolve_pencil(test_data, false, {3, 3, 3}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {3, 3, 3}));

    // 2D filters
    ASSERT_TRUE(test_convolve(test_data, true, {13, 13, 1}));
    ASSERT_TRUE(test_convolve(test_data, true, {13, 1, 13}));
    ASSERT_TRUE(test_convolve(test_data, true, {1, 13, 13}));

    ASSERT_TRUE(test_convolve_pencil(test_data, true, {13, 13, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {13, 1, 13}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {1, 13, 13}));

    // 1D filters
    ASSERT_TRUE(test_convolve(test_data, true, {13, 1, 1}));
    ASSERT_TRUE(test_convolve(test_data, true, {1, 13, 1}));
    ASSERT_TRUE(test_convolve(test_data, true, {1, 1, 13}));

    ASSERT_TRUE(test_convolve_pencil(test_data, true, {13, 1, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {1, 13, 1}));
    ASSERT_TRUE(test_convolve_pencil(test_data, true, {1, 1, 13}));
}

TEST_F(CreateAPRTest, READ_PARTICLE_TYPE){
    ASSERT_TRUE(test_apr_file_particle_type());
}


bool compare_lazy_iterator(LazyIterator& lazy_it, LinearIterator& apr_it) {

    if(lazy_it.level_min() != apr_it.level_min()) { return false; }
    if(lazy_it.level_max() != apr_it.level_max()) { return false; }
    if(lazy_it.total_number_particles() != apr_it.total_number_particles()) { return false; }

    for(int level = lazy_it.level_min(); level <= lazy_it.level_max(); ++level) {
        if(lazy_it.z_num(level) != apr_it.z_num(level)) { return false; }
        if(lazy_it.x_num(level) != apr_it.x_num(level)) { return false; }
        if(lazy_it.y_num(level) != apr_it.y_num(level)) { return false; }
    }

    // loading rows of data
    lazy_it.set_buffer_size(lazy_it.y_num(lazy_it.level_max()));
    uint64_t counter = 0;
    for(int level = apr_it.level_max(); level >= apr_it.level_min(); --level) {
        for(int z = 0; z < apr_it.z_num(level); ++z) {
            for(int x = 0; x < apr_it.x_num(level); ++x) {
                lazy_it.load_row(level, z, x);
                if(lazy_it.begin(level, z, x) != apr_it.begin(level, z, x)) { return false; }
                if(lazy_it.end() != apr_it.end()) { return false; }

                for(; lazy_it < lazy_it.end(); ++lazy_it, ++apr_it) {
                    if(lazy_it.y() != apr_it.y()) { return false; }
                    counter++;
                }
            }
        }
    }
    if(counter != apr_it.total_number_particles()) { return false; }

    // loading slices of data
    lazy_it.set_buffer_size(lazy_it.x_num(lazy_it.level_max()) * lazy_it.y_num(lazy_it.level_max()));
    counter = 0;
    for(int level = lazy_it.level_max(); level >= lazy_it.level_min(); --level) {
        for(int z = 0; z < lazy_it.z_num(level); ++z) {
            lazy_it.load_slice(level, z);
            for(int x = 0; x < lazy_it.x_num(level); ++x) {
                if(lazy_it.begin(level, z, x) != apr_it.begin(level, z, x)) { return false; }
                if(lazy_it.end() != apr_it.end()) { return false; }

                for(; lazy_it < lazy_it.end(); ++lazy_it, ++apr_it) {
                    if(lazy_it.y() != apr_it.y()) { return false; }
                    counter++;
                }
            }
        }
    }
    if(counter != lazy_it.total_number_particles()) { return false; }

    return true;
}


bool test_lazy_iterators(TestData& test_data) {
    // write APR to file using new format (required by LazyAccess) #TODO: update the test files
    APRFile aprFile;
    std::string file_name = "lazy_access_test.apr";
    aprFile.open(file_name,"WRITE");
    aprFile.write_apr(test_data.apr);
    aprFile.write_particles("particles",test_data.particles_intensities);
    aprFile.close();

    // open file
    aprFile.open(file_name, "READ");

    // initialize LazyAccess and LazyIterator
    LazyAccess access;
    access.init(aprFile);
    access.open();
    LazyIterator lazy_it(access);

    // linear iterator
    auto apr_it = test_data.apr.iterator();

    // compare lazy and linear iterators
    bool success_apr = compare_lazy_iterator(lazy_it, apr_it);

    // initialize LazyAccess and LazyIterator for tree data
    LazyAccess tree_access;
    tree_access.init_tree(aprFile);
    tree_access.open();
    LazyIterator lazy_tree_it(tree_access);

    // linear iterator
    auto tree_it = test_data.apr.tree_iterator();

    // compare lazy and linear iterators
    bool success_tree = compare_lazy_iterator(lazy_tree_it, tree_it);

    access.close();
    tree_access.close();
    aprFile.close();

    return success_apr && success_tree;
}

TEST_F(CreateSmallSphereTest, TEST_LAZY_ITERATOR) {
    ASSERT_TRUE(test_lazy_iterators(test_data));
}

TEST_F(CreateDiffDimsSphereTest, TEST_LAZY_ITERATOR) {
    ASSERT_TRUE(test_lazy_iterators(test_data));
}


bool test_reconstruct_lazy(TestData& test_data, ReconPatch& patch) {

    // fill interior tree
    ParticleData<uint16_t> tree_data;
    APRTreeNumerics::fill_tree_mean(test_data.apr, test_data.particles_intensities, tree_data);

    // write APR and tree data to file
    APRFile aprFile;
    std::string file_name = "lazy_recon_test.apr";
    aprFile.open(file_name, "WRITE");
    aprFile.write_apr(test_data.apr);
    aprFile.write_particles("particles", test_data.particles_intensities, true);
    aprFile.write_particles("particles", tree_data, false);
    aprFile.close();

    // open file
    aprFile.open(file_name, "READ");

    // initialize lazy access and iterator for APR
    LazyAccess access;
    access.init(aprFile);
    access.open();
    LazyIterator apr_it(access);

    // intialize lazy access and iterator for tree
    LazyAccess tree_access;
    tree_access.init_tree(aprFile);
    tree_access.open();
    LazyIterator tree_it(tree_access);

    LazyData<uint16_t> lazy_parts;
    lazy_parts.init(aprFile, "particles");
    lazy_parts.open();

    LazyData<uint16_t> lazy_tree_parts;
    lazy_tree_parts.init_tree(aprFile, "particles");
    lazy_tree_parts.open();

    PixelData<uint16_t> lazy_constant;
    APRReconstruction::reconstruct_constant_lazy(apr_it, tree_it, lazy_constant, lazy_parts, lazy_tree_parts, patch);

    PixelData<uint16_t> lazy_level;
    APRReconstruction::reconstruct_level_lazy(apr_it, tree_it, lazy_level, patch);

    PixelData<uint16_t> lazy_smooth;
    APRReconstruction::reconstruct_smooth_lazy(apr_it, tree_it, lazy_smooth, lazy_parts, lazy_tree_parts, patch);

    // close files
    lazy_parts.close();
    lazy_tree_parts.close();
    access.close();
    tree_access.close();
    aprFile.close();

    /// ground truth
    PixelData<uint16_t> gt_constant;
    APRReconstruction::reconstruct_constant(test_data.apr, gt_constant, test_data.particles_intensities, tree_data, patch);

    PixelData<uint16_t> gt_level;
    APRReconstruction::reconstruct_level(test_data.apr, gt_level, patch);

    PixelData<uint16_t> gt_smooth;
    APRReconstruction::reconstruct_smooth(test_data.apr, gt_smooth, test_data.particles_intensities, tree_data, patch);

    return (compareMeshes(gt_constant, lazy_constant) +
            compareMeshes(gt_level, lazy_level) +
            compareMeshes(gt_smooth, lazy_smooth)) == 0;
}


TEST_F(CreateSmallSphereTest, TEST_RECONSTRUCT_LAZY) {

    ReconPatch patch;

    // upsampled full reconstruction
    patch.level_delta = 1;
    ASSERT_TRUE(test_reconstruct_lazy(test_data, patch));

    // full reconstruction at original resolution
    patch.level_delta = 0;
    ASSERT_TRUE(test_reconstruct_lazy(test_data, patch));

    // downsampled full reconstruction
    patch.level_delta = -2;
    ASSERT_TRUE(test_reconstruct_lazy(test_data, patch));

    // arbitrarily set patch region
    patch.z_begin = 17; patch.z_end = 118;
    patch.x_begin = 19; patch.x_end = 63;
    patch.y_begin = 3; patch.y_end = 111;

    // upsampled patch reconstruction
    patch.level_delta = 2;
    ASSERT_TRUE(test_reconstruct_lazy(test_data, patch));

    // patch reconstruction at original resolution
    patch.level_delta = 0;
    ASSERT_TRUE(test_reconstruct_lazy(test_data, patch));

    // downsampled patch reconstruction
    patch.level_delta = -1;
    ASSERT_TRUE(test_reconstruct_lazy(test_data, patch));
}


int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
