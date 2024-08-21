[![Python package](https://github.com/MPI-IS/cute-shm/actions/workflows/tests.yml/badge.svg)](https://github.com/MPI-IS/cute-shm/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/cute-shm.svg)](https://pypi.org/project/cute-shm/)


# cute-shm

cute-shm is a convenience wrapper over Python's multiprocessing shared memory. It provides an easy-to-use API for managing shared memory numpy arrays and HDF5 files.
Using the shared memory allows to share numpy arrays across multiple processes running on the same node.

## Table of Contents

  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [API](#api)
      - [sharing numpy arrays](#sharing-numpy-arrays)
      - [sharing content of hdf5 files](#sharing-content-of-hdf5-files)
      - [Logging](#logging)
      - [Typing hints](#typing-hints)
      - [Concurrent access](#concurrent-access)
    - [Under the hood](#under-the-hood)
    - [Command line executables](#command-line-executables)
    - ["Manual" cleaning of the shared memory](#manual-cleaning-of-the-shared-memory)
  - [Demos](#demos)
  - [Warnings](#warnings)
    - [Bus error](#bus-error)
    - [Garbage collection of the shared memory](#garbage-collection-of-the-shared-memory)
  - [Authorship, Copyright, and License](#authorship-copyright-and-license)


## Requirements

Python 3.10 or later.

## Installation

You can install cute-shm using pip:

```
pip install cute-shm
```

## Usage

### API

#### sharing numpy arrays

```python
import numpy as np
import cute_shm as cute

# Create some numpy arrays
a = np.array([[12, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=np.int64)
b1 = np.zeros(100, dtype=np.float32)
b2 = np.zeros(300, dtype=np.float32)

# Create a nested dictionary of arrays
arrays = {"a": a, "b": {"b1": b1, "b2": b2}}

# An arbitrary name for this projet
project_name = "myproject"

# set to True if the shared memory should not be cleaned upon exit
# i.e. another process may need to access it later
persistent = False 

# set to True if the shared memory should be overwritten if it already exists
# (if False and a project of the same name already exists, a FileExistsError will be raised)
overwrite = False # set to True if the shared memory should be overwritten if it already exists

# transfer arrays to shared memory
cute.arrays_to_shm(
    project_name,
    arrays,
    persistent=persistent,
    overwrite=overwrite,
)

# reading the arrays from the shared memory
# This could be done in a different process.
# (including processes spawned after this process exits if persistent is True)
shm_arrays = cute.shm_to_arrays(project_name, persistent=persistent)

# shm_arrays has the same structure as arrays.
# Each item has two keys: 
# - "data": the numpy array
# - "meta": related metadata
a: np.ndarray = shm_arrays["a"]["data"]

# meta data consists mostly of things you will certainly not need.
a_meta: cute.SharedArrayMeta = shm_arrays["a"]["meta"]
a_meta["shape"] # the shape of the array, same as a.shape
a_meta["dtype"] # the data type of the array, same as str(a.dtype)
a_meta["shm_name"] # the name of the shared memory segment
a_meta["shm_private_name"] # the private name of the shared memory segment
a_meta["shm"] # the shared memory segment (instance of shared_memory.SharedMemory)

# clean up the shared memory and related metadata
# (do not call this if persistent is True and you want the shared memory to be available for other processes)
cute.unlink(project_name)
```
You can also use the `unlinked_arrays_to_shm` context manager to 
ensure the shared memory and related metadata are cleaned up on exit.

(if persistent is False, the python multiprocessing resource tracker will cleanup
the shared memory automatically, but not the meta data).

```python

# Transfer arrays to shared memory
with cute.unlinked_arrays_to_shm(project_name, arrays):

    # Read arrays from shared memory
    # (this could also be done in a different process)
    shm_arrays = cute.shm_to_arrays(project_name)

# Shared memory and meta data is automatically cleaned up when the context manager exits
```

#### sharing content of hdf5 files

Content of hdf5 files can also be transferred to shared memory as a dictionary of numpy arrays.

```python
from pathlib import Path
import cute_shm as cute

hdf5_path = Path("path/to/your/file.hdf5")

project_name = "myproject"

# if True, a progress bar showing the progress of the transfer
# to the shared memory will be shown
progress = True
# for persistent and overwrite, same usage as when sharing numpy arrays
persistent = False
overwrite = False

# transfer to the shared memory
hdf5_to_shm(
    hdf5_path, project_name, progress=progress, persistent=persistent, overwrite=overwrite
)

# content of hdf5 is shared as a nested directories of nested arrays 
shm_arrays = cute.shm_to_arrays(project_name, persistent=persistent)

# dataset attributes are stored in the "meta" dictionary
a: np.ndarray = shm_arrays["a"]["data"]
a_meta: cute.SharedArrayMeta = shm_arrays["a"]["meta"]
a_meta["attrs"] # the attributes of the dataset

```

A context manager is also provided:

```python
with unlinked_hdf5_to_shm(
    hdf5_path, project_name, progress, overwrite
):
    shm_arrays = cute.shm_to_arrays(project_name, persistent=persistent)
```

#### Logging

If in your own software using the cute-sh API you set the logging to level `DEBUG`, information related to 
the creation/deletion of shared memory segments will be provided.

#### Typing hints

cute-shm provides and uses these type aliases:

```python
# a nested dictionary of numpy arrays. This is the data structure
# that can be transferred to the shared memory.
ArrayDict: TypeAlias = dict[str, Union["ArrayDict", np.ndarray]]

# usage
arrays: cute_shm.ArrayDict = {"a": np.zeros(10), "b": {"b1": np.zeros(10), "b2": np.zeros(10)}}
cute_shm.arrays_to_shm("myproject", arrays)

# a shared memory array: data and related metadata
class SharedArray(TypedDict):
    meta: SharedArrayMeta
    data: np.ndarray

# the metadata of a shared memory array
class SharedArrayMeta(TypedDict, total=False):
    shm_name: str
    shm_private_name: str
    shm: shared_memory.SharedMemory
    shape: tuple[int, ...]
    dtype: str
    attrs: dict[str, Any]

# a nested dictionary of shared memory arrays.
# This is the data structure that is returned by the API
# when reading the shared memory.
SharedArrayDict: TypeAlias = dict[str, Union[SharedArray, "SharedArrayDict"]]

# usage
shm_arrays: cute_shm.SharedArrayDict = cute_shm.shm_to_arrays("myproject")
a: cute_shm.SharedArray = shm_arrays["a"]
a_data: np.ndarray = a["data"]
a_meta: cute_shm.SharedArrayMeta = a["meta"]
```

#### Concurrent access 
Once numpy arrays are transferred to the shared memory and no longer updated, 
they can be accessed by multiple processes concurrently without lock protection.

If a process updates the values of the arrays, locking should be implemented using 
either the [`multiprocessing.Lock`](https://docs.python.org/3/library/multiprocessing.html#synchronization-between-processes)
or a [`filelock`](https://py-filelock.readthedocs.io/en/latest/).

See for example the `demo_server.py`  and the `demo_client.py` demos [here](demos/).

### Under the hood

- when the arrays are transferred to shared memory, a toml file is created in the `/tmp/cute-shm` directory. Its name is based on the project name.
- this toml file contains all the metadata required for other processes to "cast" the shared memory to the proper dictionary structure.

If you prefer to store the metadata in a different location, change the `root` attribute of the class `Project2Toml`:

```python
import cute_shm as cute

cute.Project2Toml.root = Path("/path/to/your/directory")
```

### Command line executables

To load the content of a hdf5 file to the shared memory via command line:

```bash
cute-shm-hdf5 <project_name> <hdf5_path> 
```
for example:

```bash
# transfer to the shared memory.
# file.hdf5 expected in the current directory
cute-shm-hdf5 myproject file.hdf5 

# overwrite if data corresponding to a project named myproject already exists
# 'o' for overwrite
cute-shm-hdf5 myproject file.hdf5 -o

# do not display a progress bar
cute-shm-hdf5 myproject file.hdf5 -no-progress

# display debug information instead of a progress bar
# 'v' for verbose
cute-shm-hdf5 myproject file.hdf5 -v

# any python process can now access the shared memory
# "myproject" via cute.shm_to_arrays.
```

You can display about data hosted in the shared memory:

```bash
# full information
cute-shm-list

# just an overview ('s' for short)
cute-shm-list -s
```

Note that `cute-shm-list` will not only display the content of the shared memory created via `cute-shm-hdf5`, but also the content of
the shared memory created via the python API (shared memory currently being transferred will not be listed).

Shared memory can be cleaned up via the command `cute-shm-unlink <project_name>`:

```bash
cute-shm-unlink myproject
```
### "Manual" cleaning of the shared memory

Alternatively to use the API or the command line to free the shared memory, you may either:

- reboot the computer
- delete files prefixed by `cute-shm` in the `/dev/shm` folder and related toml files in the `/tmp/cute-shm` folder.

## Demos

For examples: [demos](demos/).

## Warnings

### Bus error (hitting RAM limits)

If the RAM of the computer gets full, transfer to the shared memory will not only fail, the process will also crash with a bus error.
This is a system error that cannot be managed by the python exception handling.

It has also been observed that the process becomes stuck when the RAM limit is exceeded.

### Garbage collection of the shared memory

Shared memory numpy arrays buffers is a pointer to the buffer of a related instance of `shared_memory.SharedMemory`.
This related instance needs to be loaded in the heap, i.e. it should not be garbage collected. If it is garbage collected,
then a `SegmentationFault` will occur and the process will crash (not managed by python exception handling).

The instance of the `shared_memory.SharedMemory` is located in the `meta` dictionary of the `SharedArrayMeta` instance.

For example, one should not:

```python
# read the shared memory to a dictionary of numpy arrays and meta data
shm_arrays: cute_shm.SharedArrayDict = cute_shm.shm_to_arrays(project_name)

# access the data and meta data of 'a'
shm_array = shm_arrays["a"]

# the numpy array
data: np.ndarray = shm_array["data"]

# the meta data
meta: cute_shm.SharedArrayMeta = shm_array["meta"]

# deleting the pointer to the shared memory segment
# related to the data
del meta["shm"]

# this will crash: the shared memory segment has been garbage collected
print(data[0]) 
```

or:

```python
def get_np(project_name: str)->np.ndarray:

    # read the shared memory to a dictionary of numpy arrays and meta data
    shm_arrays: cute_shm.SharedArrayDict = cute_shm.shm_to_arrays(project_name)

    # access the data and meta data of 'a'
    shm_array = shm_arrays["a"]
    data: np.ndarray = shm_array["data"]
    meta: cute_shm.SharedArrayMeta = shm_array["meta"]

    # meta["shm"] is a reference to the shared memory segment.
    # It will be garbage collected, along with the meta dictionary,
    # when the function exits.
    return data

a: np.ndarray = get_np("myproject")
# this will crash: the shared memory segment has been garbage collected
print(a[0])  
```

> **Note:** When the shared memory instance is garbage collected:
> - The data is not removed from the shared memory.
> - Only the pointer to the data buffer is lost.
> - This loss of pointer affects only the current process.

## Authorship, Copyright, and License

**Author:** Vincent Berenz  
**Institution:** Max Planck Institute for Intelligent Systems, Tübingen, Germany  
**Copyright:** © 2024 Max Planck Gesellschaft  
**License:** [MIT License](https://opensource.org/licenses/MIT)