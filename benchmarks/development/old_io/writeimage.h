//
//  ImageGen Library
//
//  Bevan Cheeseman 2016
//
//  Functions for writing meshclass to files, either hdf5 or tiff
//
//

#ifndef PARTPLAY_WRITEIMAGE_H
#define PARTPLAY_WRITEIMAGE_H

#include <tiffio.h>
#include "parameters.h"

#include "src/data_structures/Mesh/meshclass.h"

template <typename T>
void write_image_tiff(Mesh_data<T>& image,std::string filename){
    //
    //
    //  Bevan Cheeseman 2015
    //
    //
    //  Code for writing tiff image to file
    //
    //
    //
    //


    TIFF* tif = TIFFOpen(filename.c_str() , "w");
    uint32 width;
    uint32 height;
    unsigned short nbits;
    unsigned short samples;
    void* raster;


    //set the size
    width = image.y_num;
    height = image.x_num;
    samples = 1;
    //bit size
    nbits = sizeof(T)*8;

    int num_dir = image.z_num;

    //set up the tiff file
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, nbits);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,TIFFDefaultStripSize(tif, width*samples));


    int test_field;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &test_field);

    int ScanlineSize= (int)TIFFScanlineSize(tif);
    int StripSize =  (int)TIFFStripSize(tif);
    int rowsPerStrip;
    int nRowsToConvert;

    raster = _TIFFmalloc(StripSize);
    T *TBuf = (T*)raster;

    TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);

    int z_start = 0;
    int z_end = num_dir-1;

    int row_count = 0;

    for(int i = z_start; i < (z_end+1); i++) {
        if (i > z_start) {
            TIFFWriteDirectory(tif);

        }

        //set up the tiff file
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, nbits);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples);
        TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,TIFFDefaultStripSize(tif, width*samples));

        row_count = 0;

        for (int topRow = 0; topRow < height; topRow += rowsPerStrip) {
            nRowsToConvert = (topRow + rowsPerStrip >height?height- topRow : rowsPerStrip);

            std::copy(image.mesh.begin() + i*width*height + row_count,image.mesh.begin() + i*width*height + row_count + nRowsToConvert*width, TBuf);

            row_count += nRowsToConvert*width;

            TIFFWriteEncodedStrip(tif, TIFFComputeStrip(tif, topRow, 0), TBuf, nRowsToConvert*ScanlineSize);

        }

    }

    _TIFFfree(raster);


    TIFFClose(tif);

}

template <typename T>
void debug_write(Mesh_data<T> input,std::string img_name){
    
    
    std::string image_path;
    
    std::string name = "debug";
    
    image_path = get_path("IMAGE_GEN_PATH");
    
    
    //debug
    Mesh_data<uint16_t> output_int;
    output_int.initialize(input.y_num,input.x_num,input.z_num,0);
    std::copy(input.mesh.begin(),input.mesh.end(),output_int.mesh.begin());
    
    write_image_tiff(output_int, image_path + img_name + ".tif");
    
}


#endif //PARTPLAY_WRITEIMAGE_H