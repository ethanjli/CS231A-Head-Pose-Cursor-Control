from os import path
import sys

import numpy as np

_PACKAGE_PATH = path.dirname(sys.modules[__name__].__file__)
_ROOT_PATH = path.dirname(_PACKAGE_PATH)
_CALIB_PATH = path.join(_ROOT_PATH, 'calib')

TRANSLATION = np.load(path.join(_CALIB_PATH, 'trans_vec.npy'))
ROTATION = np.load(path.join(_CALIB_PATH, 'rot_mat.npy'))
K_LEFT = np.load(path.join(_CALIB_PATH, 'cam_mats_left.npy'))
K_RIGHT = np.load(path.join(_CALIB_PATH, 'cam_mats_right.npy'))
