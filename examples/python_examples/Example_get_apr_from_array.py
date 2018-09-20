import os, sys
import argparse
import numpy as np

from TIFFreadwrite import readTiff

sys.path.insert(0, "../../cmake-build-release")  #change this to your build folder
import pyApr

"""
    Example script that reads in a .tif image to a numpy array and computes the APR from it. The resulting APR
    is written to a HDF5 file in directory of the input file.
    
    Usage: python Example_get_apr_from_file.py -d /home/user/images/ -i myImage.tif -o myAPR
"""
def main(args):

    filePath = os.path.join(args.directory, args.input)

    apr = pyApr.AprShort()         # assuming 16 bit integers

    # ----------------------- APR parameter settings ----------------------- #
    pars = pyApr.APRParameters()

    # Set some parameters manually
    pars.Ip_th = 1000
    pars.sigma_th = 100
    pars.sigma_th_max = 10
    pars.rel_error = 0.1
    pars.lmbda = 1

    # Or try using the auto_parameters
    pars.auto_parameters = False

    apr.set_parameters(pars)
    # ----------------------- APR Conversion ----------------------- #

    img = readTiff(filePath).astype(np.uint16)    # numpy.ndarray

    apr.get_apr_from_array(img)

    outPath = os.path.join(args.directory, args.output)

    apr.write_apr(outPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script: read in a TIFF image as a numpy array and compute its "
                                                 "APR. The resulting APR is saved as a HDF5 file in the given directory.")
    parser.add_argument('--input', '-i', type=str, help="Name of the input .tif image")
    parser.add_argument('--directory', '-d', type=str, help="Directory of the input (and output) file")
    parser.add_argument('--output', '-o', type=str, help="Name of the output HDF5 file")
    args = parser.parse_args()

    main(args)