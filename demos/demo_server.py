"""
Writing an array to the shared memory and keep updating it, until `ctrl+c` is pressed.
Run in another terminal demo_client.py.
"""

import logging
import signal
import sys
import time

import numpy as np
from filelock import FileLock
from rich.console import Console
from rich.table import Table

import cute_shm


def create_and_update_array(project_name: str, lock_path: str) -> None:

    # creating a numpy array
    array = np.zeros(10, dtype=np.int32)
    array[0] = 1

    # for displaying in the terminal
    console = Console()

    # exiting gracefully on SIGINT
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # transferring the array to shared memory
    with cute_shm.unlinked_arrays_to_shm(project_name, {"array": array}):
        index = 0
        while True:
            # updating the shared array
            array = np.zeros(10, dtype=np.int32)
            index = (index + 1) % 10
            array[index] = 1
            with FileLock(lock_path):
                shm_arrays = cute_shm.shm_to_arrays(project_name)
                shm_arrays["array"]["data"][:] = array[:]  # type: ignore
            display_array(console, array)
            time.sleep(0.2)


def display_array(console: Console, array: np.ndarray) -> None:
    table = Table(show_header=False)
    table.add_row("Array", str(array))
    console.clear()
    console.print(table)


if __name__ == "__main__":
    project_name = "demo_project"
    lock_path = "/tmp/demo_project.lock"
    create_and_update_array(project_name, lock_path)
