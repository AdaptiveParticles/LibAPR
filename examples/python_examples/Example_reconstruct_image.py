import os, sys
import argparse
import numpy as np
from TIFFreadwrite import writeTiff

sys.path.insert(0, "../../cmake-build-release")
import pyApr

"""
    Example script that reads an APR from a given HDF5 file, reconstructs the pixel image and writes it to .tif
    
    Usage: python Example_reconstruct_image.py -d /Users/foo/Documents/myDir/ -i myAPR.h5 -o reconstructedImage.tif
"""
def main():
    parser = argparse.ArgumentParser(description="Set input and output names and folder")
    parser.add_argument('--input', '-i', type=str, help="Name of the input APR file")
    parser.add_argument('--directory', '-d', type=str, help="Directory of the input file")
    parser.add_argument('--output', '-o', type=str, help="Name of the output .tif file")
    args = parser.parse_args()

    apr = pyApr.AprShort()              # assuming 16 bit integers

    # read in APR file
    filePath = args.directory + args.input
    apr.read_apr(filePath)

    # reconstruct image
    arr = apr.reconstruct()             # piecewise constant reconstruction
    #arr = apr.reconstruct_smooth()     # smooth reconstruction

    arr = np.array(arr, copy=False)

    outPath = args.directory + args.output
    writeTiff(outPath, arr, compression=None)


if __name__ == '__main__':
    main()