import argparse
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import psutil
from rich.console import Console
from rich.table import Table

from .core import ArrayDict, bytes_to_human
from .hdf5_shm import unlinked_hdf5_to_shm
from .numpy_shm import shm_to_arrays


def hdf5_to_dict(hdf5_path: Path) -> ArrayDict:
    def recursively_load_datasets(h5node):
        data_dict = {}
        for key, item in h5node.items():
            if isinstance(item, h5py.Dataset) and len(item.shape) != 0:
                if item.dtype.names:
                    data_dict[key] = {}
                    for name in item.dtype.names:
                        data_dict[key][name] = item[name][:]
                else:
                    data_dict[key] = item[:]
            elif isinstance(item, h5py.Group):
                data_dict[key] = recursively_load_datasets(item)
        return data_dict

    with h5py.File(hdf5_path, "r") as f:
        return recursively_load_datasets(f)


def access_shared_memory(
    project_name: str, iterations: int, result: multiprocessing.Manager().dict
) -> None:
    def access_data(key, array):
        if "data" in array:
            _ = np.random.choice(array["data"].ravel())
        else:
            for key, sub_array in array.items():
                access_data(key, sub_array)

    shm_arrays = shm_to_arrays(project_name, persistent=True)
    start_time = time.time()
    for _ in range(iterations):
        for key, array in shm_arrays.items():
            access_data(key, array)
    end_time = time.time()
    try:
        result["frequency"] += iterations / (end_time - start_time)
    except KeyError:
        result["frequency"] = iterations / (end_time - start_time)


def access_directly(
    data_dict: ArrayDict,
    iterations: int,
    result: multiprocessing.Manager().dict,
) -> None:
    def access_data(key, dataset):
        if isinstance(dataset, dict):
            for key, sub_dataset in dataset.items():
                access_data(key, sub_dataset)
        else:
            # Access the data
            _ = np.random.choice(dataset.ravel())

    start_time = time.time()
    for _ in range(iterations):
        for key, dataset in data_dict.items():
            access_data(key, dataset)
    end_time = time.time()
    try:
        result["frequency"] += iterations / (end_time - start_time)
    except KeyError:
        result["frequency"] = iterations / (end_time - start_time)


def _monitor_ram_usage(
    stop_event: multiprocessing.Event, max_value: multiprocessing.Value
) -> None:
    max_ram_usage = 0
    while not stop_event.is_set():
        memory = psutil.virtual_memory()
        ram_usage = memory.total - memory.available
        if ram_usage > max_ram_usage:
            max_ram_usage = ram_usage
        time.sleep(0.01)
    max_value.value = max_ram_usage


from contextlib import contextmanager


class RAM_Monitor:

    @classmethod
    def start(cls):
        cls._manager = multiprocessing.Manager()
        cls._max_usage = multiprocessing.Value("Q", 0)
        cls._stop_event = multiprocessing.Event()
        cls._ram_monitor = multiprocessing.Process(
            target=_monitor_ram_usage, args=(cls._stop_event, cls._max_usage)
        )
        cls._ram_monitor.start()

    @classmethod
    def stop(cls) -> int:
        cls._stop_event.set()
        cls._ram_monitor.join()
        return cls._max_usage.value


def run_experiment(
    hdf5_path: Path,
    project_name: str,
    num_processes: int,
    iterations: int,
    use_shm: bool,
    data_dict: ArrayDict = None,
) -> dict:

    manager = multiprocessing.Manager()
    results = manager.dict()

    RAM_Monitor.start()

    ts1 = time.time()
    processes = []
    for _ in range(num_processes):
        if use_shm:
            p = multiprocessing.Process(
                target=access_shared_memory,
                args=(project_name, iterations, results),
            )
        else:
            p = multiprocessing.Process(
                target=access_directly, args=(data_dict, iterations, results)
            )
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    max_ram_usage = RAM_Monitor.stop()

    avg_frequency = results["frequency"] / num_processes

    return {
        "num_processes": num_processes,
        "avg_frequency": avg_frequency,
        "max_ram_usage": max_ram_usage,
    }


def evaluation():
    parser = argparse.ArgumentParser(
        description="Evaluate RAM usage of processes accessing shared memory arrays using cute-shm."
    )
    parser.add_argument("hdf5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("project_name", type=str, help="Project name for shared memory")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of iterations for each process",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    hdf5_path = Path(args.hdf5_file)
    project_name = args.project_name
    iterations = args.iterations

    console = Console()
    table = Table(title="Experiment Results")
    table.add_column("Number of Processes", justify="center")
    table.add_column("Average Iteration Frequency (Shared Memory)", justify="center")
    table.add_column("Average Iteration Frequency (Direct Access)", justify="center")
    table.add_column("Max RAM Usage (Shared Memory)", justify="center")
    table.add_column("Max RAM Usage (Direct Access)", justify="center")

    logging.info("Loading data from HDF5 file to the RAM (direct, no shared memory)")
    RAM_Monitor.start()
    data_dict = hdf5_to_dict(hdf5_path)
    logging.info(
        f"Max observed RAM usage during load: {bytes_to_human(RAM_Monitor.stop())}"
    )

    logging.info("Running experiments with direct access to the RAM")
    direct_results = []
    for num_processes in [1, 5, 10, 15]:
        logging.info(f"Running experiment with {num_processes} processes")
        result_direct = run_experiment(
            hdf5_path,
            project_name,
            num_processes,
            iterations,
            False,
            data_dict,
        )
        direct_results.append(result_direct)

    del data_dict
    data_dict = None

    logging.info("Loading data from HDF5 file to the shared memory")
    RAM_Monitor.start()
    with unlinked_hdf5_to_shm(hdf5_path, project_name, overwrite=True, progress=True):
        logging.info(
            f"Max observed RAM usage during load: {bytes_to_human(RAM_Monitor.stop())}"
        )
        logging.info("Running experiments with shared memory")
        shm_results = []
        for num_processes in [1, 5, 10, 15]:
            logging.info(f"Running experiment with {num_processes} processes")
            result_shm = run_experiment(
                hdf5_path,
                project_name,
                num_processes,
                iterations,
                True,
                data_dict,
            )
            shm_results.append(result_shm)

    for direct_result, shm_result in zip(direct_results, shm_results):
        table.add_row(
            str(direct_result["num_processes"]),
            f"{shm_result['avg_frequency']:.2f}",
            f"{direct_result['avg_frequency']:.2f}",
            bytes_to_human(shm_result["max_ram_usage"]),
            bytes_to_human(direct_result["max_ram_usage"]),
        )

    console.print(table)
