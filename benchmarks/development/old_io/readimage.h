//////////////////////////////////////////////////////////////////////////////////
//
//	Bevan Cheeseman 2016
//
//	GenImage
//
//	readimage.h
//
//	Use the libtiff library to read in tiff images or hdf5 image into mesh_data structure
//
//
//////////////////////////////////////////////////////////////////////////////////

//
//  SPIMStackReader.h
//  dive
//
//  Created by Ulrik Guenther on 13/05/14.
//  Copyright (c) 2014 ulrik.is. All rights reserved.
//

#ifndef PARTPLAY_READIMAGE_H
#define PARTPLAY_READIMAGE_H
#include "benchmarks/development/old_io/hdf5functions.h"
#include <tiffio.h>
#include "src/io/TiffUtils.hpp"

template <typename T>
void load_image_tiff(MeshData<T>& mesh_data,std::string file_name,int z_start = 0, int z_end = -1) {
    mesh_data = TiffUtils::getMesh<T>(file_name);
}

#endif //PARTPLAY_READIMAGE_H

