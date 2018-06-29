import os, sys
import argparse


sys.path.insert(0, "../../cmake-build-release")
import pyApr

"""
    Example script that computes an APR from a .tif image and saves the APR to a HDF5 file in directory of the input file.
    
    Usage: python Example_get_apr_from_file.py -d /Users/foo/Documents/myDir/ -i myImage.tif -o myAPR

"""
def main():
    parser = argparse.ArgumentParser(description="Set input and output names and folder")
    parser.add_argument('--input', '-i', type=str, help="Name of the input .tif image")
    parser.add_argument('--directory', '-d', type=str, help="Directory of the input file")
    parser.add_argument('--output', '-o', type=str, help="Name of the output HDF5 file")
    args = parser.parse_args()

    filePath = args.directory + args.input

    apr = pyApr.AprShort()         # assuming 16 bit integers

    # ----------------------- Parameter settings ----------------------- #
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

    apr.get_apr_from_file(filePath)

    outPath = args.directory + args.output

    apr.write_apr(outPath)


if __name__ == '__main__':
    main()



