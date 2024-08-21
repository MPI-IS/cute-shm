
## hdf5 file

- demo.h5: dummy hdf5 file for demo purposes
- generate_demo_hdf5_file: for generating demo.h5


## demo_numpy

Transferring numpy arrays to the shared memory and reading them
back in another process.

## demo_hdf5

Transferring the content of `demo.h5` in the shared memory and
reading it as a dictionary of numpy arrays in another process.

## transfer, read, clean

### demo_transfer

Transfer numpy arrays to the shared memory in a persistent manner:
the shared memory is not freed when the process exits.

### demo_read

Numpy arrays written by demo_transfer are accessed.

### demo_clean

The data written by `demo_transfer` will live in the RAM until
this demo is called (or the cleaned up by deleting files
in `/dev/shm` and `/tmp/cute-shm`)

## server, client

### demo_server

Write an array to the shared memory and keep updating it, until `ctrl+c` is pressed.

### demo_client

Read the array from the shared memory and displays it in the terminal until `ctrl+c` is pressed.
Assumes that the demo `demo_server` is running in another terminal.