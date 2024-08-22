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

from .core import ArrayDict
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
    result["frequency"] = iterations / (end_time - start_time)


from .core import bytes_to_human


def monitor_ram_usage(
    stop_event: multiprocessing.Event, result: multiprocessing.Manager().dict
) -> None:
    max_ram_usage = 0
    while not stop_event.is_set():
        ram_usage = psutil.virtual_memory().used
        if ram_usage > max_ram_usage:
            max_ram_usage = ram_usage
        time.sleep(0.1)
    result["max_ram_usage"] = max_ram_usage


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
    stop_event = multiprocessing.Event()

    ram_monitor = multiprocessing.Process(
        target=monitor_ram_usage, args=(stop_event, results)
    )
    ram_monitor.start()

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
        p.start()

    for p in processes:
        p.join()

    stop_event.set()
    ram_monitor.join()

    avg_frequency = results["frequency"] / num_processes
    max_ram_usage = results["max_ram_usage"]

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
        default=1000,
        help="Number of iterations for each process",
    )
    args = parser.parse_args()

    # logging.basicConfig(level=logging.DEBUG)

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

    data_dict = hdf5_to_dict(hdf5_path)

    with unlinked_hdf5_to_shm(hdf5_path, project_name, overwrite=True, progress=False):

        for num_processes in [1, 5, 10, 15]:
            result_shm = run_experiment(
                hdf5_path,
                project_name,
                num_processes,
                iterations,
                True,
                data_dict,
            )
            result_direct = run_experiment(
                hdf5_path,
                project_name,
                num_processes,
                iterations,
                False,
                data_dict,
            )
            table.add_row(
                str(result_shm["num_processes"]),
                f"{result_shm['avg_frequency']:.2f}",
                f"{result_direct['avg_frequency']:.2f}",
                psutil._common.bytes2human(result_shm["max_ram_usage"]),
                psutil._common.bytes2human(result_direct["max_ram_usage"]),
            )

    console.print(table)
