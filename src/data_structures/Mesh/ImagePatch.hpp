//
// Created by joel on 02.11.20.
//

#ifndef LIBAPR_IMAGEPATCH_HPP
#define LIBAPR_IMAGEPATCH_HPP

struct ImagePatch {
    int x_begin_global = 0;
    int x_end_global = -1;

    int y_begin_global = 0;
    int y_end_global = -1;

    int z_begin_global = 0;
    int z_end_global = -1;

    int x_begin = 0;
    int x_end = -1;
    int x_offset = 0;

    int y_begin = 0;
    int y_end = -1;
    int y_offset = 0;

    int z_begin = 0;
    int z_end = -1;
    int z_offset = 0;

    int x_ghost_l = 0;
    int x_ghost_r = 0;

    int y_ghost_l = 0;
    int y_ghost_r = 0;

    int z_ghost_l = 0;
    int z_ghost_r = 0;
};


void initPatchGlobal(ImagePatch& patch, int z_begin_global, int z_end_global, int x_begin_global, int x_end_global, int y_begin_global, int y_end_global) {
    patch.z_begin_global = z_begin_global;
    patch.x_begin_global = x_begin_global;
    patch.y_begin_global = y_begin_global;

    patch.z_end_global = z_end_global;
    patch.x_end_global = x_end_global;
    patch.y_end_global = y_end_global;

    patch.z_begin = 0;
    patch.x_begin = 0;
    patch.y_begin = 0;

    patch.z_end = z_end_global - z_begin_global;
    patch.x_end = x_end_global - x_begin_global;
    patch.y_end = y_end_global - y_begin_global;
}

#endif //LIBAPR_IMAGEPATCH_HPP
