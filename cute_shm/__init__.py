"""
Cute Shm: A convenient Python package for manipulating shared memory numpy arrays.

This package provides functionality for transferring numpy arrays to the shared
memory, supporting nested dictionary structures and HDF5 files.
"""

import importlib.metadata

from .core import (ArrayDict, MetaArrayDict, Project2Toml, SharedArray,
                   SharedArrayDict, SharedArrayMeta, bytes_to_human, unlink)
from .hdf5_shm import hdf5_size, hdf5_to_shm, unlinked_hdf5_to_shm
from .numpy_shm import arrays_to_shm, shm_to_arrays, unlinked_arrays_to_shm
from .progress import ShmProgress

__version__ = importlib.metadata.version("cute-shm")
