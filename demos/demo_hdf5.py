"""
Demo: sharing HDF5 file content in the shared memory.
"""

import logging
import multiprocessing
from pathlib import Path

import cute_shm as cute


def read_shared_memory(project_name):
    # reads the arrays from the shared memory
    # and displays related information

    shm_arrays = cute.shm_to_arrays(project_name)

    def print_dataset_info(name, data):
        print(f"{name}:")
        print(f"  Shape: {data['data'].shape}")
        print(f"  Dtype: {data['data'].dtype}")
        print(f"  Size in bytes: {data['data'].nbytes}")
        print(f"  Mean value: {data['data'].mean()}")
        print(f"  Standard deviation: {data['data'].std()}")
        if "attrs" in data:
            print(f"  Attributes: {data['attrs']}")
        print()

    for key, value in shm_arrays.items():
        if isinstance(value, dict) and "data" not in value:
            for subkey, subvalue in value.items():
                print_dataset_info(f"{key}/{subkey}", subvalue)
        else:
            print_dataset_info(key, value)


if __name__ == "__main__":

    # optional: printing debug information
    logging.basicConfig(level=logging.DEBUG)

    # path to the HDF5 file
    hdf5_path = Path("demo.h5")

    # arbirary name of the project
    project_name = "demo_project"

    # Transfer HDF5 file to shared memory
    with cute.unlinked_hdf5_to_shm(hdf5_path, project_name):

        # access the content of the hdf5 file from another process
        process = multiprocessing.Process(
            target=read_shared_memory, args=(project_name,)
        )
        process.start()
        process.join()

    # the shared memory and related meta toml file are cleaned up automatically
    # when the context manager exits
