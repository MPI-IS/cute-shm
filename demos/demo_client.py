"""
Reading continuously the numpy array shared by the demo 
demo_server (expected to be running in another terminal)
"""

import logging
import signal
import sys
import time
from typing import cast

import numpy as np
from filelock import FileLock
from rich.console import Console
from rich.table import Table

import cute_shm


def read_and_display_array(project_name: str, lock_path: str) -> None:

    # for displaying in the terminal
    console = Console()

    # reading the shared array from the shared memory
    try:
        shm_arrays = cute_shm.shm_to_arrays(project_name, persistent=True)
    except FileNotFoundError:
        console.print("Shared memory not found. Start demo_server.py first.")
        sys.exit(1)

    # exiting gracefully on SIGINT
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        # reading and displaying the array
        with FileLock(lock_path):
            array = shm_arrays["array"]["data"]
        array_ = cast(np.ndarray, array)
        display_array(console, array_)
        time.sleep(0.1)


def display_array(console: Console, array: np.ndarray) -> None:
    table = Table(show_header=False)
    table.add_row("Array", str(array))
    console.clear()
    console.print(table)


if __name__ == "__main__":
    project_name = "demo_project"
    lock_path = "/tmp/demo_project.lock"
    read_and_display_array(project_name, lock_path)
