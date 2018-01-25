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
    mesh_data.move(TiffUtils::getMesh<T>(file_name));
//    TIFF* tif = TIFFOpen(file_name.c_str(), "r");
//    int dircount = 0;
//    uint32 width;
//    uint32 height;
//    unsigned short nbits;
//    unsigned short samples;
//    void* raster;
//
//    if (tif) {
//        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
//        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
//        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &nbits);
//        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
//
//        do {
//            dircount++;
//        } while (TIFFReadDirectory(tif));
//    } else {
//        std::cout <<  "Could not open TIFF file." << std::endl;
//        return;
//    }
//
//
//    if (dircount < (z_end - z_start + 1)){
//        std::cout << "number of slices and start and finish inconsitent!!" << std::endl;
//    }
//
//    //Conditions if too many slices are asked for, or all slices
//    if (z_end > dircount) {
//        std::cout << "Not that many slices, using max number instead" << std::endl;
//        z_end = dircount-1;
//    }
//    if (z_end < 0) {
//        z_end = dircount-1;
//    }
//
//
//    dircount = z_end - z_start + 1;
//
//    long ScanlineSize=TIFFScanlineSize(tif);
//    long StripSize =  TIFFStripSize(tif);
//
//    int rowsPerStrip;
//    int nRowsToConvert;
//
//    raster = _TIFFmalloc(StripSize);
//    T *TBuf = (T*)raster;
//
//    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
//
//
//    for(int i = z_start; i < (z_end+1); i++) {
//        TIFFSetDirectory(tif, i);
//
//
//        for (int topRow = 0; topRow < height; topRow += rowsPerStrip) {
//            nRowsToConvert = (topRow + rowsPerStrip >height?height- topRow : rowsPerStrip);
//
//            TIFFReadEncodedStrip(tif, TIFFComputeStrip(tif, topRow, 0), TBuf, nRowsToConvert*ScanlineSize);
//
//            std::copy(TBuf, TBuf+nRowsToConvert*width, back_inserter(mesh_data.mesh));
//
//        }
//
//
//    }
//
//    _TIFFfree(raster);
//
//
//    mesh_data.z_num = dircount;
//    mesh_data.y_num = width;
//    mesh_data.x_num = height;
//
//
//    TIFFClose(tif);
}

#endif //PARTPLAY_READIMAGE_H

