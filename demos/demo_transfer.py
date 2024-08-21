"""
This demo shows how to load an HDF5 file into shared memory in a persistent way.
See demo_read.py for how to read from shared memory and demo_clean.py for how to
clean up the shared memory.

After running this demo, you can also see the content of the shared memory via 
the command line executable cute-shm-list.
"""

import logging
from pathlib import Path

import cute_shm

if __name__ == "__main__":

    # optional: printing debug information
    logging.basicConfig(level=logging.DEBUG)

    # path to the HDF5 file
    hdf5_path = Path("demo.h5")

    # arbirary name of the project
    project_name = "demo_transfer"

    # Transfer HDF5 file to shared memory
    # persistent=True: the shared memory is not cleaned up when the process of this
    #   demo exits. Warning: this could be considered as a (controlled) memory leak !
    #   The shared memory will not be freed until cleaned (see demo_clean.py) or
    #   the computer is rebooted.
    # overwrite=True: if a shared memory with the same name already exists, it is overwritten.
    #   (otherwise an error is raised).
    # progress=False: if True, a progress bar is displayed (for monitoring the transfer). Does not
    #   work well with the logging, which sould be turned off.
    cute_shm.hdf5_to_shm(
        hdf5_path, project_name, persistent=True, overwrite=True, progress=False
    )
