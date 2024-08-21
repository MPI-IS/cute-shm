"""
This module provides functionality for converting numpy arrays to shared memory
and vice versa. It includes functions for recursively handling nested dictionaries
of arrays and managing the shared memory lifecycle.
"""

from contextlib import contextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Iterator, cast

import numpy as np
import tomli
import tomli_w

from .core import (ArrayDict, MetaArrayDict, Project2Toml, SharedArray,
                   SharedArrayDict, SharedArrayMeta, array_size,
                   bytes_to_human, generate_random_string, logger, read_shm,
                   unlink, unregister)


def create_shm_from_np(project: str, array: np.ndarray) -> SharedArrayMeta:
    """
    Create a shared memory array from a numpy array.

    Args:
        array (np.ndarray): The input numpy array to be converted to shared memory.

    Returns:
        SharedArrayMeta: Metadata for the created shared memory array.
    """
    name = f"{Project2Toml.prefix}{project}.{generate_random_string()}"

    shm = shared_memory.SharedMemory(name=name, create=True, size=array.nbytes)

    shm_array: np.ndarray = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)

    shm_array[:] = array[:]

    return SharedArrayMeta(
        shm_name=shm.name,
        shm_private_name=shm._name,  # type: ignore
        shape=array.shape,
        dtype=str(array.dtype),
        attrs={},
    )


def _recursive_arrays_to_shm(
    project: str,
    arrays: ArrayDict,
    dest: MetaArrayDict,
):
    # Recursively convert numpy arrays to shared memory.
    #
    # Args:
    #    arrays: Dictionary of numpy arrays or nested dictionaries.
    #    dest: Destination dictionary for shared memory metadata.

    for k, v in arrays.items():
        if isinstance(v, dict):
            d: MetaArrayDict = {}
            dest[k] = d
            v_ = cast(ArrayDict, v)
            _recursive_arrays_to_shm(project, v_, d)
        else:
            logger.debug(
                f"transferring key '{k}' to shared memory "
                f"({bytes_to_human(v.nbytes)})"
            )
            dest[k] = create_shm_from_np(project, v)
            logger.debug(
                f"transfer of key '{k}' to shared memory '{dest[k]['shm_name']}' finished"
            )


def arrays_to_shm(
    project: str,
    arrays: ArrayDict,
    persistent: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Transfer the arrays to shared memory and write corresponding toml metadata file.

    Args:
        project (str | Path): Name of the project or path to which the metadata should be dumped.
        arrays (SharedArrayDict): Dictionary of numpy arrays or nested dictionaries.
        persistent (bool, optional): Whether to unregister shared memory segments. Defaults to False.
        overwrite (bool, optional): Whether to overwrite any existing project of the same name. Defaults to False.

    Returns:
        Path: The path to the generated toml metadata file.
    """
    toml_path = Project2Toml.get_path(project)

    if toml_path.exists():
        if overwrite is False:
            raise FileExistsError(
                f"Project {project}: failed to transfer numpy arrays to shared memory. "
                f"The corresponding meta file {toml_path} already exists"
            )
        else:
            unlink(project)

    dest: MetaArrayDict = {}
    _recursive_arrays_to_shm(project, arrays, dest)
    if persistent:
        logger.debug("persistence: unregistering shared memory segments")
        unregister(dest)
    logger.debug(f"writing shared memory meta data to {toml_path}")
    with open(toml_path, "wb") as f:
        tomli_w.dump(dest, f)
    return toml_path


@contextmanager
def unlinked_arrays_to_shm(
    project: str, arrays: ArrayDict, overwrite: bool = False
) -> Iterator[Path]:
    """
    Context manager for arrays_to_shm, which cleans up the shared memory upon exit,
    avoiding inadvertent RAM memory leaks.

    Args:
        project (str | Path): Name of the project or path to which the metadata should be dumped.
        arrays (SharedArrayDict): Dictionary of numpy arrays or nested dictionaries.
        overwrite (bool, optional): Whether to overwrite any existing project of the same name. Defaults to False.

    Yields:
        Path: The path to the generated toml metadata file.
    """
    try:
        persistent = False
        path = arrays_to_shm(project, arrays, persistent, overwrite=overwrite)
        yield path
    finally:
        unlink(project)


def _recursive_shm_to_arrays(
    meta_dict: MetaArrayDict,
    dest: SharedArrayDict,
):
    # Recursively convert shared memory metadata to numpy arrays.
    #
    # Args:
    #    meta_dict: Dictionary of shared memory metadata.
    #    dest: Destination dictionary for numpy arrays.
    #    shms: List to store SharedMemory objects.

    for k, v in meta_dict.items():
        if "shm_name" in v:
            meta = cast(SharedArrayMeta, v)
            logger.debug(
                f"reading {k} from shared memory "
                f"({bytes_to_human(array_size(meta))})"
            )
            shared_array: SharedArray = read_shm(meta)
            dest[k] = shared_array
        else:
            d: SharedArrayDict = {}
            dest[k] = d
            _recursive_shm_to_arrays(v, d)  # type: ignore


def shm_to_arrays(project: str, persistent: bool = True) -> SharedArrayDict:
    """
    Parse the toml metadata and cast the corresponding shared memory to a dictionary
    of numpy arrays.

    Args:
        project (str | Path): Name of the project or path to a toml shared metadata file.
        persistent (bool, optional): Whether to unregister shared memory segments. Defaults to True.

    Returns:
        ArrayDict: A dictionary of numpy arrays or nested dictionaries.
    """
    toml_path = Project2Toml.get_path(project)
    logger.debug(f"opening shared memory meta data file {toml_path}")
    with open(toml_path, "rb") as f:
        d = tomli.load(f)

    meta_dict: MetaArrayDict = cast(MetaArrayDict, d)
    shared_array_dict: SharedArrayDict = {}
    _recursive_shm_to_arrays(meta_dict, shared_array_dict)
    logger.debug(f"finished parsing {toml_path}")

    if persistent:
        logger.debug("unregistering shared memory segments")
        unregister(meta_dict)

    return shared_array_dict
