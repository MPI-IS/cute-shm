"""
Core functionality for the cute_shm package.
"""

import glob
import logging
import math
import os
import random
import re
import string
from multiprocessing import Process, Value, resource_tracker, shared_memory
from pathlib import Path
from typing import Any, Optional, TypeAlias, TypedDict, Union, cast

import numpy as np
import tomli

logger = logging.getLogger("cute-shm")


class SharedArrayMeta(TypedDict, total=False):
    """
    A TypedDict representing metadata for a shared memory array.

    Attributes:
        shm_name (str): The name of the shared memory segment.
        shape (tuple[int, ...]): The shape of the array.
        dtype (str): The data type of the array.
        attrs (dict[str, Any]): Additional attributes of the array.
    """

    shm_name: str
    shm_private_name: str
    shm: shared_memory.SharedMemory
    shape: tuple[int, ...]
    dtype: str
    attrs: dict[str, Any]


MetaArrayDict: TypeAlias = dict[str, Union[SharedArrayMeta, "MetaArrayDict"]]
"""
A dictionary of SharedArrayMeta or nested dictionaries of metadata.
"""


class SharedArray(TypedDict):

    meta: SharedArrayMeta
    data: np.ndarray


SharedArrayDict: TypeAlias = dict[str, Union[SharedArray, "SharedArrayDict"]]
"""
dictionary of numpy arrays or nested dictionaries of arrays.
"""

ArrayDict: TypeAlias = dict[str, Union["ArrayDict", np.ndarray]]
"""
nested dictionary of numpy arrays.
"""


def array_size(meta: SharedArrayMeta) -> int:
    """
    Return the size (in bytes) of the corresponding numpy array.

    Args:
        meta (SharedArrayMeta): Metadata of the shared memory array.

    Returns:
        int: Size of the array in bytes.
    """
    numel = np.prod(meta["shape"])
    elsize = np.dtype(meta["dtype"]).itemsize
    return int(numel) * elsize


def bytes_to_human(size_bytes: int) -> str:
    """
    Convert a size in bytes to a human-readable string.

    Args:
        size_bytes (int): Size in bytes.

    Returns:
        str: Human-readable string representation of the size.
    """
    if size_bytes == 0:
        return "0 bytes"
    size_name = ("bytes", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def read_shm(
    meta: SharedArrayMeta,
) -> SharedArray:
    """
    Read a shared memory array based on its metadata.

    Args:
        meta (SharedArrayMeta): Metadata of the shared memory array.

    Returns:
        SharedMemoryMeta object and the numpy array.
    """
    shm = shared_memory.SharedMemory(name=meta["shm_name"])
    data: np.ndarray = np.ndarray(
        meta["shape"],
        np.dtype(meta["dtype"]),
        buffer=shm.buf,
    )
    meta["shm"] = shm
    return SharedArray(meta=meta, data=data)


def unlink(project: str) -> None:
    """
    Unlink (remove) shared memory segments associated with a project.

    Args:
        project (str): Name of the project or path to the metadata file.

    Returns:
        None
    """

    def _recursive_unlink(meta_dict: MetaArrayDict):
        for k, v in meta_dict.items():
            if "shm_name" in v:
                v_ = cast(SharedArrayMeta, v)
                shm: Optional[shared_memory.SharedMemory] = None
                try:
                    shm = v_["shm"]
                    shm = cast(shared_memory.SharedMemory, shm)
                except KeyError:
                    pass
                except FileNotFoundError:
                    logger.warning(f"shared memory '{k}' not found")
                    continue
                if shm is None:
                    try:
                        shm = shared_memory.SharedMemory(
                            name=str(v_["shm_name"])
                        )
                    except FileNotFoundError:
                        logger.warning(f"shared memory '{k}' not found")
                        continue
                logger.debug(f"closing and unlinking shared memory '{k}'")
                shm.close()
                shm.unlink()
            else:
                _recursive_unlink(v)  # type: ignore

    toml_path = Project2Toml.get_path(project)
    logger.debug(f"closing and unlinking shared memory based on {toml_path}")
    with open(toml_path, "rb") as f:
        d = tomli.load(f)

    meta_dict: MetaArrayDict = cast(MetaArrayDict, d)
    _recursive_unlink(meta_dict)
    toml_path.unlink()


def unregister(meta_dict: MetaArrayDict) -> None:
    """
    Unregister shared memory segments from the resource tracker.

    Args:
        meta_dict (MetaArrayDict): Dictionary of metadata.
    """
    for k, v in meta_dict.items():
        if "shm_name" in v:
            logger.debug(f"persistence: unregistering {v['shm_name']}")
            resource_tracker.unregister(v["shm_private_name"], "shared_memory")
        else:
            unregister(v)  # type: ignore


def generate_random_string(length=10):
    """
    Returns a random string
    """
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


class Project2Toml:
    """
    A class to manage paths for project TOML files.

    Attributes:
        root (Path): The root directory for TOML files.
        prefix (str): The prefix for TOML filenames.
    """

    root = Path("/tmp/cute-shm")
    prefix = "cute-shm."

    @classmethod
    def get_path(cls, project: str) -> Path:
        """
        Return the path to the TOML file for a project.

        Args:
            project (str): Name of the project.

        Returns:
            Path: Path to the TOML file.
        """
        if not cls.root.exists():
            cls.root.mkdir(parents=True)
        return Path(cls.root) / f"{cls.prefix}{project}.toml"

    @classmethod
    def list(cls) -> dict[str, Path]:
        """
        List all projects (i.e. corresponding meta data toml file exists)

        Returns:
            dict[str, Path]: Dictionary of projects with their names as keys and paths as values.
        """
        projects = {}
        for file in cls.root.glob(f"{cls.prefix}*.toml"):
            project_name = file.stem[len(cls.prefix) :]
            projects[project_name] = file
        return projects
