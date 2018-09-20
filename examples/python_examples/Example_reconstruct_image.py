import os, sys
import argparse
import numpy as np
from TIFFreadwrite import writeTiff

sys.path.insert(0, "../../cmake-build-release")  #change this to your build folder
import pyApr

"""
    Example script that reads an APR from a given HDF5 file, reconstructs the pixel image and writes it to .tif
    
    Usage: python Example_reconstruct_image.py -d /home/user/images/ -i myAPR.h5 -o reconstructedImage.tif
"""
def main(args):

    apr = pyApr.AprShort()              # assuming 16 bit integers

    # read in the APR file
    filePath = os.path.join(args.directory, args.input)
    apr.read_apr(filePath)

    # reconstruct image
    if args.smooth:
        arr = apr.reconstruct_smooth()     # smooth reconstruction
    else:
        arr = apr.reconstruct()            # piecewise constant reconstruction

    # convert PyPixelData object to numpy array without copy
    arr = np.array(arr, copy=False)

    # write image to file
    outPath = os.path.join(args.directory, args.output)
    writeTiff(outPath, arr, compression=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script: read an APR from a HDF5 file, reconstruct the pixel image"
                                                 " and save it as TIFF.")
    parser.add_argument('--input', '-i', type=str, help="Name of the input APR file")
    parser.add_argument('--directory', '-d', type=str, help="Directory of the input file")
    parser.add_argument('--output', '-o', type=str, help="Name of the output .tif file")
    parser.add_argument('--smooth', action='store_true', default=False, help="(Optional) use smooth reconstruction")
    args = parser.parse_args()

    main(args)