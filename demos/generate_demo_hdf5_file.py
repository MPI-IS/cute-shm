from typing import Literal

import h5py
import numpy as np


def generate_demo_hdf5_file(
    filename: str = "demo.h5", total_size: int = 2_000_000
) -> None:
    with h5py.File(filename, "w") as f:
        # Simple datasets of various shapes, sizes, and dtypes
        f.create_dataset(
            "simple/1d_float32", data=np.random.rand(10000).astype(np.float32)
        )
        f.create_dataset(
            "simple/2d_int16",
            data=np.random.randint(-32768, 32767, size=(100, 100), dtype=np.int16),
        )
        f.create_dataset(
            "simple/3d_uint8",
            data=np.random.randint(0, 255, size=(50, 50, 50), dtype=np.uint8),
        )
        f.create_dataset(
            "simple/4d_complex64",
            data=np.random.rand(20, 20, 20, 20).astype(np.complex64),
        )

        # Structured dataset
        dt = np.dtype([("id", np.int32), ("value", np.float32), ("flag", np.bool_)])
        structured_data = np.array(
            [(1, 55.5, True), (2, 70.2, False), (3, 65.8, True)], dtype=dt
        )
        structured_dataset: h5py.Dataset = f.create_dataset(
            "structured", data=structured_data
        )

        # Dataset with attributes
        dset_with_attrs: h5py.Dataset = f.create_dataset(
            "with_attributes", data=np.random.rand(500, 500)
        )
        dset_with_attrs.attrs["description"] = "Random 500x500 matrix"
        dset_with_attrs.attrs["created_by"] = "generate_demo_hdf5_file"

        # Compressed dataset
        f.create_dataset(
            "compressed",
            data=np.random.rand(1000, 1000),
            compression="gzip",
            compression_opts=9,
        )

        # Large dataset to reach total size
        remaining_size: int = total_size - sum(
            dset.nbytes for dset in f.values() if isinstance(dset, h5py.Dataset)
        )
        if remaining_size > 0:
            shape: tuple[int, int] = (
                int(np.sqrt(remaining_size / 8)),
            ) * 2  # Assuming 8 bytes per float64
            f.create_dataset("large", data=np.random.rand(*shape))

    print(f"Demo HDF5 file '{filename}' created successfully.")


if __name__ == "__main__":
    generate_demo_hdf5_file()
