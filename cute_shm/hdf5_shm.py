"""
This module provides functionality for working with HDF5 files and shared memory.

It includes functions for converting HDF5 datasets to shared memory arrays and
vice versa, as well as utilities for measuring and reporting on HDF5 file sizes
and transfer progress.
"""

import threading
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Generator, Iterator

import h5py
import numpy as np
import tomli_w

from .core import (MetaArrayDict, Project2Toml, SharedArrayMeta,
                   bytes_to_human, generate_random_string, logger, unlink,
                   unregister)
from .numpy_shm import create_shm_from_np
from .progress import ShmProgress


def _hdf5_size(h5node: h5py.Group, total_size: int) -> int:
    for item in h5node.values():
        if isinstance(item, h5py.Dataset) and len(item.shape) != 0:
            total_size += item.nbytes
        elif isinstance(item, h5py.Group):
            total_size = _hdf5_size(item, total_size)
    return total_size


def hdf5_size(h5: Path | h5py.Group) -> int:
    """
    Calculate the total size of datasets in an HDF5 file or group.

    Args:
        h5 (Path | h5py.Group): Path to an HDF5 file or an h5py.Group object.

    Returns:
        int: Total size of all datasets in bytes.
    """
    if isinstance(h5, Path) or isinstance(h5, str):
        with h5py.File(Path(h5), "r") as f:
            return _hdf5_size(f, 0)
    else:
        return _hdf5_size(h5, 0)


def create_shm_from_h5py(project: str, dataset: h5py.Dataset) -> SharedArrayMeta:
    """
    Create a shared memory array from an HDF5 dataset.

    Args:
        dataset (h5py.Dataset): The HDF5 dataset to convert to shared memory.

    Returns:
        SharedArrayMeta: Metadata for the created shared memory array.
    """
    required_size = np.prod(dataset.shape) * np.dtype(dataset.dtype).itemsize
    name = f"{Project2Toml.prefix}{project}.{generate_random_string()}"

    shm = shared_memory.SharedMemory(name=name, create=True, size=required_size)

    shm_array: np.ndarray = np.ndarray(
        dataset.shape, dtype=dataset.dtype, buffer=shm.buf
    )

    shm_array[:] = dataset[:]

    return SharedArrayMeta(
        shm_name=shm.name,
        shm_private_name=shm._name,  # type: ignore
        shape=shm_array.shape,
        dtype=str(shm_array.dtype),
        attrs={str(k): v for k, v in dataset.attrs.items()},
    )


def _recursively_load_datasets(project: str, h5node: h5py.Group, meta_dict):
    for key, item in h5node.items():
        if isinstance(item, h5py.Dataset) and len(item.shape) != 0:
            if item.dtype.names:
                logger.debug(f"transfering hdf5 structured dataset {key}")
                meta_dict[key] = {}
                for name in item.dtype.names:
                    logger.debug(
                        f"transfering {key}/{name} to shared memory {bytes_to_human(item[name].nbytes)}"
                    )
                    meta_dict[key][name] = create_shm_from_np(project, item[name])
                    logger.debug(
                        f"transfer of {key}/{name} to shared memory '{meta_dict[key][name]['shm_name']}' finished"
                    )
            else:
                logger.debug(
                    f"transfering {key} to shared memory "
                    f"({bytes_to_human(item.nbytes)})"
                )
                meta_dict[key] = create_shm_from_h5py(project, item)
                logger.debug(
                    f"transfer of {key} to shared memory '{meta_dict[key]['shm_name']}' finished"
                )
        elif isinstance(item, h5py.Group):
            meta_dict[key] = {}
            _recursively_load_datasets(project, item, meta_dict[key])


@contextmanager
def _hdf5_progress(
    h5: Path | h5py.Group,
) -> Generator[ShmProgress, None, None]:
    total_size = hdf5_size(h5)
    progress = ShmProgress(total_size)
    progress.start()
    try:
        yield progress
    finally:
        progress.stop()


def hdf5_to_shm(
    hdf5_path: Path,
    project: str,
    progress: bool = False,
    persistent: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Convert an HDF5 file to shared memory arrays.

    Args:
        hdf5_path (Path): Path to the HDF5 file.
        project (str | Path): Name of the project or path for metadata storage.
        progress (bool, optional): Whether to show progress. Defaults to False.
        persistent (bool, optional): Whether to unregister shared memory segments. Defaults to False.
        overwrite (bool, optional): Whether to overwrite any existing project of the same name. Defaults to False.

    Returns:
        None
    """
    toml_path = Project2Toml.get_path(project)

    if toml_path.exists():
        if overwrite is False:
            raise FileExistsError(
                f"Project {project}: failed to transfer {hdf5_path}. "
                f"The corresponding meta file {toml_path} already exists"
            )
        else:
            unlink(project)

    meta_dict: MetaArrayDict = {}
    with h5py.File(hdf5_path, "r") as f:
        logger.debug(
            f"transferring content of {hdf5_path} "
            f"to shared memory ({bytes_to_human(hdf5_size(f))})"
        )
        if progress:
            with _hdf5_progress(f):
                _recursively_load_datasets(project, f, meta_dict)
        else:
            _recursively_load_datasets(project, f, meta_dict)
        logger.debug(
            f"finished transferring content of " f"{hdf5_path} to shared memory"
        )
        if persistent:
            logger.debug("persistence: unregistering shared memory segments")
            unregister(meta_dict)
        logger.debug(f"writing shared memory meta data to {toml_path}")
        with open(toml_path, "wb") as f:
            tomli_w.dump(meta_dict, f)

    return toml_path


@contextmanager
def unlinked_hdf5_to_shm(
    hdf5_path: Path,
    project: str,
    progress: bool = False,
    overwrite: bool = False,
) -> Iterator[None]:
    """
    Context manager to convert HDF5 to shared memory without memory leaks.

    Args:
        hdf5_path (Path): Path to the HDF5 file.
        project (str | Path): Name of the project or path for metadata storage.
        progress (bool, optional): Whether to show progress. Defaults to False.
        overwrite (bool, optional): Whether to overwrite any existing project of the same name. Defaults to False.

    Yields:
        None
    """
    persistent = False
    try:
        hdf5_to_shm(
            hdf5_path,
            project,
            progress,
            persistent=persistent,
            overwrite=overwrite,
        )
        yield
    finally:
        unlink(project)
