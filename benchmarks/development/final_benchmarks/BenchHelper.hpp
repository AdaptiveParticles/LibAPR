//
// Created by cheesema on 25.01.18.
//

#ifndef PARTPLAY_BENCHHELPER_HPP
#define PARTPLAY_BENCHHELPER_HPP

#include <arrayfire.h>
#include "MeshDataAF.h"
#include "src/data_structures/Mesh/MeshData.hpp"

#include <string>
#include <cmath>
#include <SynImageClasses.hpp>
#include <benchmarks/development/old_io/parameters.h>

#include <cstdio>
#include <dirent.h>
#include <iostream>
#include <fstream>

class BenchHelper {

public:

    inline bool check_file_exists(const std::string& name) {
        std::ifstream f(name.c_str());
        return f.good();
    }

    struct benchmark_settings{
        //benchmark settings and defaults
        int x_num = 128;
        int y_num = 128;
        int z_num = 128;

        float voxel_size = 0.1;
        float sampling_delta = 0.1;

        float image_sampling = 200;

        float dom_size_y = 0;
        float dom_size_x = 0;
        float dom_size_z = 0;

        std::string noise_type = "poisson";

        float shift = 1000;

        float linear_shift = 0;

        float desired_I = sqrt(1000)*10;
        float N_repeats = 1;
        float num_objects = 1;
        float sig = 1;

        float int_scale_min = 1;
        float int_scale_max = 10;

        float rel_error = 0.1;

        float obj_size = 4;

        float lambda = 0;

    };

    template<typename S>
    void copy_mesh_data_structures(MeshDataAF<S>& input_syn,MeshData<S>& input_img){
        //copy across metadata
        input_img.y_num = input_syn.y_num;
        input_img.x_num = input_syn.x_num;
        input_img.z_num = input_syn.z_num;

        input_img.initialize(input_img.y_num ,input_img.x_num ,input_img.z_num );

        std::copy(input_syn.mesh.begin(),input_syn.mesh.end(),input_img.mesh.begin());

    }


    void gen_parameter_pars(SynImage& syn_image,Proc_par& pars,std::string image_name){
        //
        //
        //  Takes in the SynImage model parameters and outputs them to the APR parameter class
        //
        //
        //

        pars.name = image_name;

        pars.dy = syn_image.sampling_properties.voxel_real_dims[0];
        pars.dx = syn_image.sampling_properties.voxel_real_dims[1];
        pars.dz = syn_image.sampling_properties.voxel_real_dims[2];

        pars.psfy = syn_image.PSF_properties.real_sigmas[0];
        pars.psfx = syn_image.PSF_properties.real_sigmas[1];
        pars.psfz = syn_image.PSF_properties.real_sigmas[2];

        pars.ydim = syn_image.real_domain.size[0];
        pars.xdim = syn_image.real_domain.size[1];
        pars.zdim = syn_image.real_domain.size[2];

        pars.noise_sigma = sqrt(syn_image.noise_properties.gauss_var);
        pars.background = syn_image.global_trans.const_shift;

    }

    void set_up_benchmark_defaults(SynImage& syn_image,benchmark_settings& bs){
        /////////////////////////////////////////
        //////////////////////////////////////////
        // SET UP THE DOMAIN SIZE

        int x_num = bs.x_num;
        int y_num = bs.y_num;
        int z_num = bs.z_num;

        ///////////////////////////////////////////////////////////////////
        //
        //  sampling properties

        //voxel size
        float voxel_size = bs.voxel_size;
        float sampling_delta = bs.sampling_delta;

        syn_image.sampling_properties.voxel_real_dims[0] = voxel_size;
        syn_image.sampling_properties.voxel_real_dims[1] = voxel_size;
        syn_image.sampling_properties.voxel_real_dims[2] = voxel_size;

        //sampling rate/delta
        syn_image.sampling_properties.sampling_delta[0] = sampling_delta;
        syn_image.sampling_properties.sampling_delta[1] = sampling_delta;
        syn_image.sampling_properties.sampling_delta[2] = sampling_delta;

        //real size of domain
        float dom_size_y = y_num*sampling_delta;
        float dom_size_x = x_num*sampling_delta;
        float dom_size_z = z_num*sampling_delta;
        syn_image.real_domain.set_domain_size(0, dom_size_y, 0, dom_size_x, 0, dom_size_z);

        bs.dom_size_y = dom_size_y;
        bs.dom_size_x = dom_size_x;
        bs.dom_size_z = dom_size_z;

        ///////////////////////////////////////////////////
        //Noise properties

        syn_image.noise_properties.gauss_var = 50;
        syn_image.noise_properties.noise_type = bs.noise_type;
        //syn_image.noise_properties.noise_type = "none";

        ////////////////////////////////////////////////////
        // Global Transforms

        float shift = bs.shift;
        syn_image.global_trans.const_shift = shift;
        float background = shift;

        float max_dim = std::max(dom_size_y,std::max(dom_size_y,dom_size_z));

        float min_grad = .5*shift/max_dim; //stop it going negative
        float max_grad = 1.5*shift/max_dim;

        Genrand_uni gen_rand;

        syn_image.global_trans.grad_y = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);
        syn_image.global_trans.grad_x = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);
        syn_image.global_trans.grad_z = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);




        set_gaussian_psf(syn_image,bs);

    }
    void update_domain(SynImage& syn_image,benchmark_settings& bs){

        syn_image.sampling_properties.voxel_real_dims[0] = bs.voxel_size;
        syn_image.sampling_properties.voxel_real_dims[1] = bs.voxel_size;
        syn_image.sampling_properties.voxel_real_dims[2] = bs.voxel_size;

        //sampling rate/delta
        syn_image.sampling_properties.sampling_delta[0] = bs.sampling_delta;
        syn_image.sampling_properties.sampling_delta[1] = bs.sampling_delta;
        syn_image.sampling_properties.sampling_delta[2] = bs.sampling_delta;

        //real size of domain
        bs.dom_size_y = bs.y_num*bs.sampling_delta;
        bs.dom_size_x = bs.x_num*bs.sampling_delta;
        bs.dom_size_z = bs.z_num*bs.sampling_delta;
        syn_image.real_domain.set_domain_size(0, bs.dom_size_y, 0, bs.dom_size_x, 0, bs.dom_size_z);

    }


    void set_gaussian_psf(SynImage& syn_image_loc,benchmark_settings& bs){
        ///////////////////////////////////////////////////////////////////
        //PSF properties
        syn_image_loc.PSF_properties.real_sigmas[0] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[0];
        syn_image_loc.PSF_properties.real_sigmas[1] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[1];
        syn_image_loc.PSF_properties.real_sigmas[2] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[2];

        syn_image_loc.PSF_properties.I0 = 1/(pow(2*3.14159265359,1.5)*syn_image_loc.PSF_properties.real_sigmas[0]*syn_image_loc.PSF_properties.real_sigmas[1]*syn_image_loc.PSF_properties.real_sigmas[2]);

        syn_image_loc.PSF_properties.I0 = 1;

        syn_image_loc.PSF_properties.cut_th = 0.0000001;

        syn_image_loc.PSF_properties.set_guassian_window_size();

        syn_image_loc.PSF_properties.type = "gauss";

    }

    void generate_objects(SynImage& syn_image_loc,benchmark_settings& bs){

        Genrand_uni gen_rand;

        //remove previous objects
        syn_image_loc.real_objects.resize(0);

        // loop over the objects
        for(int id = 0;id < syn_image_loc.object_templates.size();id++) {

            //loop over the different template objects

            Real_object temp_obj;

            //set the template id
            temp_obj.template_id = 0;

            for (int q = 0; q < bs.num_objects; q++) {

                // place them randomly in the image

                temp_obj.template_id = id;

                Object_template curr_obj = syn_image_loc.object_templates[temp_obj.template_id];

                float min_dom = std::min(bs.dom_size_y,std::min(bs.dom_size_x,bs.dom_size_z));

                float max_obj = std::max(curr_obj.real_size[0],std::max(curr_obj.real_size[1],curr_obj.real_size[2]));

                if(max_obj < min_dom) {
                    //have them avoid the boundary, to avoid boundary effects
                    temp_obj.location[0] = gen_rand.rand_num(bs.dom_size_y * .02,
                                                             .98 * bs.dom_size_y - curr_obj.real_size[0]);
                    temp_obj.location[1] = gen_rand.rand_num(bs.dom_size_x * .02,
                                                             .98 * bs.dom_size_x - curr_obj.real_size[1]);
                    temp_obj.location[2] = gen_rand.rand_num(bs.dom_size_z * .02,
                                                             .98 * bs.dom_size_z - curr_obj.real_size[2]);
                } else {
                    temp_obj.location[0] = .5 * bs.dom_size_y - curr_obj.real_size[0]/2;
                    temp_obj.location[1] = .5 * bs.dom_size_x - curr_obj.real_size[1]/2;
                    temp_obj.location[2] = .5 * bs.dom_size_z - curr_obj.real_size[2]/2;
                }
                float obj_int =  bs.desired_I;

                if(bs.int_scale_min != bs.int_scale_max) {

                    obj_int = gen_rand.rand_num(bs.int_scale_min, bs.int_scale_max) * bs.desired_I;

                }
                // temp_obj.int_scale = (
                //   ((curr_obj.real_deltas[0] * curr_obj.real_deltas[1] * curr_obj.real_deltas[2]) * obj_int) /
                //   (curr_obj.max_sample * pow(bs.voxel_size, 3)));

                temp_obj.int_scale =  obj_int;
                syn_image_loc.real_objects.push_back(temp_obj);
            }
        }


    }


    struct obj_properties {

        float density = 1000000;
        float sample_rate = 200;
        std::vector<float> real_size_vec  = {0,0,0};
        float rad_ratio = 0;
        std::vector<float> obj_size_vec = {4,4,4};
        float obj_size = 4;
        float real_size = 0;
        float rad_ratio_template = 0;
        float img_del = 0.1;

        obj_properties(benchmark_settings& bs): obj_size(bs.obj_size) ,img_del(bs.sampling_delta), sample_rate(bs.image_sampling){
            sample_rate = 200;

            obj_size_vec = {obj_size,obj_size,obj_size};

            real_size = obj_size + 8*bs.sig*img_del;
            rad_ratio = (obj_size/2)/real_size;

            float density = 1000000;

            rad_ratio_template = (obj_size/2)/real_size;

        }

    };

    std::vector<std::string> listFiles(const std::string& path,const std::string& extenstion)
    {
        //
        //  Bevan Cheeseman 2017, adapted from Stack overflow code
        //
        //  For a particular folder, finds files with a certain string in their name and returns as a vector of strings, I don't think this will work on Windows.
        //


        DIR* dirFile = opendir( path.c_str() );

        std::vector<std::string> file_list;

        if ( dirFile )
        {
            struct dirent* hFile;
            errno = 0;
            while (( hFile = readdir( dirFile )) != NULL )
            {
                if ( !std::strcmp( hFile->d_name, "."  )) continue;
                if ( !std::strcmp( hFile->d_name, ".." )) continue;

                // in linux hidden files all start with '.'
                //if ( gIgnoreHidden && ( hFile->d_name[0] == '.' )) continue;

                // dirFile.name is the name of the file. Do whatever string comparison
                // you want here. Something like:
                if ( std::strstr( hFile->d_name, extenstion.c_str() )) {
                    printf(" found a .tiff file: %s", hFile->d_name);
                    std::cout << std::endl;
                    file_list.push_back(hFile->d_name);
                }

            }
            closedir( dirFile );
        }

        return file_list;
    }
};



#endif //PARTPLAY_BENCHHELPER_HPP
