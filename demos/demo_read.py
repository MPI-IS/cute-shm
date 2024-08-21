"""
This demo shows how to read from shared memory.
See demo_transfer.py for how to load an HDF5 file into shared memory and demo_clean.py
for how to clean up the shared memory.
"""

import logging
from typing import cast

import cute_shm

if __name__ == "__main__":

    # optional: printing debug information
    logging.basicConfig(level=logging.DEBUG)

    # arbirary name of the project
    # (should match the name used in demo_transfer.py)
    project_name = "demo_transfer"

    # read from shared memory
    # persistent=True: the shared memory is not cleaned up when the process exits !
    #  This may counter intuitive, but python default behavior will be to clean up
    #  the shared memory upon exit, even if this is not the process that created
    #  the shared memory. If this argument is ommitted, the shared memory will be
    #  cleaned up when the process exits.
    #  (persistant True is the default, here doing it only explicit)
    try:
        shm_arrays: cute_shm.SharedArrayDict = cute_shm.shm_to_arrays(
            project_name, persistent=True
        )
    except FileNotFoundError:
        print("Shared memory not found. Please run demo_load.py first.")
        exit(1)

    # print information about the datasets

    def print_dataset_info(name: str, shm_array: cute_shm.SharedArray):
        data = shm_array["data"]
        meta = shm_array["meta"]
        print(f"{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Size in bytes: {data.nbytes}")
        print(f"  Mean value: {data.mean()}")
        print(f"  Standard deviation: {data.std()}")
        if "attrs" in meta:
            print(f"  Attributes: {meta['attrs']}")
        print()

    for key, value in shm_arrays.items():
        if isinstance(value, dict) and "data" not in value:
            for subkey, subvalue in value.items():
                subvalue_ = cast(cute_shm.SharedArray, subvalue)
                print_dataset_info(f"{key}/{subkey}", subvalue_)
        else:
            value_ = cast(cute_shm.SharedArray, value)
            print_dataset_info(key, value_)
