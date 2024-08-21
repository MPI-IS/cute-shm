"""
This demo shows how to clean up the shared memory created by demo_transfer.py.
Alternatively, you can also clean up the shared memory via the command line
executable cute-shm-unlink:

    cute-shm-unlink demo_project
"""

import logging

import cute_shm

if __name__ == "__main__":

    # optional: printing debug information
    logging.basicConfig(level=logging.DEBUG)

    # name of the project
    # (should match the name used in demo_transfer.py)
    project_name = "demo_transfer"

    # clearing the shared memory and deleting the toml
    # meta data file
    try:
        cute_shm.unlink(project_name)
    except FileNotFoundError:
        print("Shared memory not found. Please run demo_transfer.py first.")
        exit(1)
