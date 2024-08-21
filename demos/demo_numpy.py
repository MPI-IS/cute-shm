"""
Demo: sharing numpy arrays in the shared memory.
"""

import logging
import multiprocessing

import numpy as np

import cute_shm


def create_arrays():
    # creates and returns a nested dictionary of numpy arrays

    array1 = np.random.rand(1000).astype(np.float32)
    array2 = np.random.randint(0, 100, size=(50, 50), dtype=np.int32)
    array3 = np.random.choice([True, False], size=(100, 100))
    nested_array1 = np.random.rand(10, 10).astype(np.float32)
    nested_array2 = np.random.randint(0, 100, size=(20, 20), dtype=np.int32)

    arrays = {
        "array1": array1,
        "array2": array2,
        "array3": array3,
        "nested": {"nested_array1": nested_array1, "nested_array2": nested_array2},
    }

    return arrays


def read_shared_memory(project_name):
    # reads arrays from shared memory
    # and displays related information

    shm_arrays = cute_shm.shm_to_arrays(project_name)

    print()
    for key, value in shm_arrays.items():
        if isinstance(value, dict) and "data" in value:
            print(f"Array: {key}")
            data = value["data"]
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Size in bytes: {data.nbytes}")
            print(f"  Mean value: {data.mean()}")
            print(f"  Standard deviation: {data.std()}")
            print()
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if "data" in nested_value:
                    print(f"Nested Array: {key}/{nested_key}")
                    data = nested_value["data"]
                    print(f"  Shape: {data.shape}")
                    print(f"  Dtype: {data.dtype}")
                    print(f"  Size in bytes: {data.nbytes}")
                    print(f"  Mean value: {data.mean()}")
                    print(f"  Standard deviation: {data.std()}")
                    print()


if __name__ == "__main__":

    # optional: printing debug information
    logging.basicConfig(level=logging.DEBUG)

    # arbirary name of the project
    project_name = "demo_project"

    # creating a nested dictionary of numpy arrays
    arrays = create_arrays()

    # transfer arrays to shared memory
    with cute_shm.unlinked_arrays_to_shm(project_name, arrays):

        # accessing the shared memory array in another process
        process = multiprocessing.Process(
            target=read_shared_memory, args=(project_name,)
        )
        process.start()
        process.join()

    # the shared memory is automatically unlinked when the context manager exits
    # the related meta data toml file is also removed
