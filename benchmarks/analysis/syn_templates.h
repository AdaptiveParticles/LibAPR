//
// Created by bevanc on 26.01.17.
//

#ifndef PARTPLAY_SYN_TEMPLATES_H
#define PARTPLAY_SYN_TEMPLATES_H

#include "GenerateTemplates.hpp"
#include "apr_analysis.h"
#include "SynImageClasses.hpp"
#include "MetaBall.h"
#include "benchmark_helpers.hpp"

struct obj_properties {

    float density = 1000000;
    float sample_rate = 200;
    std::vector<float> real_size_vec  = {0,0,0};
    float rad_ratio = 0;
    std::vector<float> obj_size_vec = {4,4,4};
    float obj_size = 4;
    float real_size = 0;
    float rad_ratio_template = 0;

    obj_properties(float obj_size,float sig): obj_size(obj_size){
        sample_rate = 200;

        obj_size_vec = {obj_size,obj_size,obj_size};

        real_size = obj_size + 3*sig;
        rad_ratio = (obj_size/2)/real_size;

        float density = 1000000;

        rad_ratio_template = (obj_size/2)/real_size;
    }

};

Object_template get_object_template(cmdLineOptionsBench& options,obj_properties& obj_prop) {
    //
    //  Bevan Cheeseman 2017
    //
    //  Generates the templates
    //

    Object_template gen_template;

    if (options.template_file) {

        create_template_from_file<uint8_t>(options.template_dir + options.template_name, gen_template , obj_prop.obj_size_vec, obj_prop.density, obj_prop.rad_ratio_template);

    } else {
        if(options.template_name == "sphere") {
            generate_sphere_template(gen_template, obj_prop.sample_rate, obj_prop.real_size, obj_prop.density,
                                     obj_prop.rad_ratio);
        } else if(options.template_name == "metaball") {
            generate_metaball_template(gen_template, obj_prop.sample_rate, obj_prop.real_size, obj_prop.density,
                                       obj_prop.rad_ratio);
        }
    }

    return gen_template;
}




#endif //PARTPLAY_SYN_TEMPLATES_H
