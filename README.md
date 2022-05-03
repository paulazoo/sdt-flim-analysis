# FLIM analysis

SDT (SPC Setup & Data File) and PTU FLIM analysis written in Python by Paula Zhu

for MCB68 class project

Largely inspired from examples:
- stdfile
  https://github.com/cgohlke/sdtfile
- flimview
  https://github.com/Biophotonics-COMI/flimview
- readPTU_FLIM
  https://github.com/SumeetRohilla/readPTU_FLIM
- PicoQuant demo codes
  https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos
- from a jupyter notebook by tritemio on GitHub:
  https://gist.github.com/tritemio/734347586bc999f39f9ffe0ac5ba0e66


## Requirements
- python3

```python
import dask.array as dr
import glob
import imageio as iio
from itertools import chain
from joblib import Parallel, delayed

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import napari
import numpy as np
import os
import pandas as pd
from pathlib import Path, PurePath
import seaborn as sns
from skimage.filters import gaussian
from skimage.measure import find_contours
import sys

from PIL import Image

import xarray as xr
xr.set_options(**{
    "display_expand_attrs": False,
    "display_expand_data": False,
})

import zarr
# pip install numcodecs==0.10.0a2 for python 3.10

sys.path.append(pc_dir + "pyTCSPC")
import pyTCSPC as pc
```
