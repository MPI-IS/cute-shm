import multiprocessing
import tempfile
import time
from pathlib import Path
from typing import Generator, no_type_check

import h5py
import numpy as np
import pytest

import cute_shm


@pytest.fixture
def tmp_dir(request, scope="function") -> Generator[Path, None, None]:
    """
    Fixture yielding a temp directory.
    """
    folder_ = tempfile.TemporaryDirectory()
    folder = Path(folder_.name)
    try:
        yield folder
    finally:
        folder_.cleanup()


@no_type_check
def test_shared_memory_arrays(tmp_dir: Path) -> None:
    a: np.ndarray = np.array(
        [[12, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=np.int64
    )
    b1: np.ndarray = np.zeros(100, dtype=np.float32)
    b1[13] = 23
    b1[33] = 89
    b2: np.ndarray = np.zeros(300, dtype=np.float32)
    b2[11] = 0.24
    b2[21] = 74

    arrays: cute_shm.ArrayDict = {"a": a, "b": {"b1": b1, "b2": b2}}

    cute_shm.Project2Toml.root = tmp_dir
    project = "cute-shm-tests"

    with cute_shm.unlinked_arrays_to_shm(project, arrays):
        shm_arrays: cute_shm.SharedArrayDict = cute_shm.shm_to_arrays(project)

        assert "a" in shm_arrays
        assert "b" in shm_arrays
        assert len(shm_arrays) == 2
        assert len(shm_arrays["b"]) == 2
        assert "b1" in shm_arrays["b"]
        assert "b2" in shm_arrays["b"]

        shared_a = shm_arrays["a"]["data"]
        shared_b1 = shm_arrays["b"]["b1"]["data"]
        shared_b2 = shm_arrays["b"]["b2"]["data"]

        assert shared_a.shape == a.shape
        assert shared_a.dtype == a.dtype
        assert shared_b1.shape == b1.shape
        assert shared_b1.dtype == b1.dtype
        assert shared_b2.shape == b2.shape
        assert shared_b2.dtype == b2.dtype

        assert np.array_equal(a, shared_a)
        assert np.array_equal(b1, shared_b1)
        assert np.array_equal(b2, shared_b2)


@no_type_check
def test_hdf5_to_shm(tmp_dir: Path) -> None:

    cute_shm.Project2Toml.root = tmp_dir
    project = "utesting"

    # Create a temporary HDF5 file
    hdf5_path = tmp_dir / "test.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("dataset1", data=np.random.rand(10, 5))
        group = f.create_group("group1")
        group.create_dataset("dataset2", data=np.random.rand(5, 5))

    with cute_shm.unlinked_hdf5_to_shm(hdf5_path, project, progress=False):
        shm_arrays = cute_shm.shm_to_arrays(project)

        assert "dataset1" in shm_arrays
        assert "group1" in shm_arrays
        assert "dataset2" in shm_arrays["group1"]

        with h5py.File(hdf5_path, "r") as f:
            assert np.array_equal(
                f["dataset1"][:], shm_arrays["dataset1"]["data"]
            )
            assert np.array_equal(
                f["group1/dataset2"][:],
                shm_arrays["group1"]["dataset2"]["data"],
            )


def test_concurrent_reading(tmp_dir: Path) -> None:
    """
    Test for checking that concurrent reading from shared memory does not lead to
    corrupted data (even without locking).
    """

    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    target_value: float = 1.62
    target_index = 2
    test_array[target_index] = target_value

    arrays: cute_shm.ArrayDict = {"test_array": test_array}

    cute_shm.Project2Toml.root = tmp_dir
    project_name = "concurrent_reading_test"

    with cute_shm.unlinked_arrays_to_shm(project_name, arrays):

        def _read_shared_memory_value(
            project_name, index, expected_value, result
        ):
            shm_arrays = cute_shm.shm_to_arrays(project_name)
            for _ in range(1500):
                if shm_arrays["test_array"]["data"][index] != expected_value:
                    result.value = False
                time.sleep(0.001)

        manager = multiprocessing.Manager()
        result = manager.Value("b", True)

        processes = []
        for _ in range(5):
            p = multiprocessing.Process(
                target=_read_shared_memory_value,
                args=(project_name, target_index, target_value, result),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        assert result.value
