from sdtfile import *
import numpy as np
import pandas as pd
import os

def read_sdt_file(sdtfile, channel=0, xpix=256, ypix=256, tpix=256):
    """
    Reads a sdtfile and returns the header and a data cube.
    Parameters
    ----------
    sdtfile : str
        Path to SdtFile
    channel : int
    xpix : int
    ypix : int
    tpix : int
    Returns
    -------
    3d ndarray
        Read dataset with shape (xpix, ypix, tpix)
    dict
        Header information
    """
    sdt = SdtFile(sdtfile)
    if np.shape(sdt.data)[0] == 0:
        print("There is an error with this file: {}".format(sdtfile))
    sdt_meta = pd.DataFrame.from_records(sdt.measure_info[0])
    sdt_meta = sdt_meta.append(
        pd.DataFrame.from_records(sdt.measure_info[1]), ignore_index=True
    )
    sdt_meta.append(pd.DataFrame.from_records(sdt.measure_info[2]), ignore_index=True)
    sdt_meta = sdt_meta.append(
        pd.DataFrame.from_records(sdt.measure_info[3]), ignore_index=True
    )
    header = {}
    header["flimview"] = {}
    header["flimview"]["sdt_info"] = sdt.info
    header["flimview"]["filename"] = os.path.basename(sdtfile)
    header["flimview"]["pathname"] = os.path.dirname(sdtfile)
    header["flimview"]["xpix"] = xpix
    header["flimview"]["ypix"] = ypix
    header["flimview"]["tpix"] = tpix
    header["flimview"]["tresolution"] = sdt.times[0][1] / 1e-12
    return np.reshape(sdt.data[channel], (xpix, ypix, tpix)), header
    