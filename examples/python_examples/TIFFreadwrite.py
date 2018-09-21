from libtiff import TIFF, TIFFfile
import numpy as np

def writeTiff(fileName, array, compression=None):
    """Write input (numpy) array to tiff file with the specified name or path"""

    outTiff = TIFF.open(fileName, mode='w')

    array = array.squeeze().astype(dtype=np.uint16) # cast array to uint16

    ndims = len(array.shape)

    if ndims == 3:

        for zInd in range(array.shape[2]):
            outTiff.write_image(array[:, :, zInd], compression=compression, write_rgb=False)

    elif 0 < ndims < 3:

        outTiff.write_image(array, compression=compression, write_rgb=False)

    else:
        raise ValueError('Input array must have between 1 and 3 dimensions')

    outTiff.close()

    return None

def readTiff(fileName):
    """
    Read a tiff file into a numpy array
    Usage: img = readTiff(fileName)
    """
    tiff = TIFFfile(fileName)
    samples, sample_names = tiff.get_samples()

    outList = []
    for sample in samples:
        outList.append(np.copy(sample))

    out = np.concatenate(outList, axis=-1)

    tiff.close()

    return out