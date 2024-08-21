import argparse
import sys
import time
from multiprocessing import resource_tracker
from pathlib import Path
from typing import TypeAlias

import h5py
import tomli
from rich.console import Console
from rich.table import Table

import cute_shm

from .core import Project2Toml, array_size, logger


def hdf5():
    """
    Command-line interface function to transfer HDF5 data to shared memory.

    This function parses command-line arguments, reads an HDF5 file, and transfers its contents
    to shared memory. It provides options for progress display and verbose output.

    Command-line arguments:
    - hdf5_file: Path to the HDF5 file (absolute path or filename in current directory)
    - project: Project name or absolute path to the TOML metadata file
    - -no-progress: Disable progress bar
    - -v, --verbose: Enable verbose output

    The function outputs information about the transfer process and exits with status code 0 on success,
    or 1 on error.
    """
    parser = argparse.ArgumentParser(description="Transfer HDF5 data to shared memory")
    parser.add_argument(
        "project",
        type=str,
        help="Arbitrary project name",
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to the HDF5 file (absolute path or filename in current directory)",
    )
    parser.add_argument(
        "-no-progress", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing project of same name",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(logging.DEBUG)
    else:
        logger.disabled = True

    try:
        # Convert hdf5_file to Path, resolving relative paths
        hdf5_path = Path(args.hdf5_file).resolve()

        with h5py.File(hdf5_path, "r") as f:
            total_size = cute_shm.hdf5_size(f)
            dataset_count = sum(1 for _ in f.keys())

        show_progress = not (args.no_progress or args.verbose)

        toml_path = cute_shm.hdf5_to_shm(
            hdf5_path,
            args.project,
            progress=show_progress,
            persistent=True,
            overwrite=args.overwrite,
        )

        sys.stdout.write(f"TOML metadata file: {toml_path}\n")
        sys.stdout.write(f"Datasets transferred: {dataset_count}\n")
        sys.stdout.write(f"Total memory size: {cute_shm.bytes_to_human(total_size)}\n")
        sys.exit(0)
    except Exception as e:
        import traceback

        sys.stderr.write(f"Error: {str(e)}\n")
        sys.stderr.write(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


def unlink_shm():
    """
    Command-line interface function to clean up shared memory created by HDF5 transfer.

    This function parses command-line arguments and removes the shared memory associated
    with a specific project. It provides an option for verbose output.

    Command-line arguments:
    - project: Project name or absolute path to the TOML metadata file
    - -v, --verbose: Enable verbose output

    The function outputs a success message and exits with status code 0 on success,
    or 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Clean up shared memory created by HDF5 transfer"
    )
    parser.add_argument(
        "project",
        type=str,
        help="Project name",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(logging.DEBUG)
    else:
        logger.disabled = True

    try:
        cute_shm.unlink(args.project)
        sys.stdout.write(
            f"Shared memory for project '{args.project}' has been cleaned up.\n"
        )
        sys.exit(0)
    except Exception as e:
        import traceback

        sys.stderr.write(f"Error: {str(e)}\n")
        sys.stderr.write(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


def _project_info(project: str, toml_path: Path) -> tuple[int, int, bool]:
    has_missing_shm = False

    if not toml_path.exists():
        raise FileNotFoundError(f"TOML metadata file not found: {toml_path}")

    with open(toml_path, "rb") as f:
        meta_dict = tomli.load(f)

    def parse(d) -> tuple[int, int, bool]:
        missing_shm = False
        total_size = 0
        nb_arrays = 0
        for k, v in d.items():
            if "shm_name" in v:
                size = array_size(v)
                total_size += size
                nb_arrays += 1
                shm_name = v["shm_name"]
                if not Path(f"/dev/shm/{shm_name}").exists():
                    missing_shm = True
            else:
                ts, nb_a, ms = parse(v)
                total_size += ts
                nb_arrays += nb_a
                if ms:
                    missing_shm = True
        return total_size, nb_arrays, missing_shm

    total_size, nb_arrays, has_missing_shm = parse(meta_dict)

    return total_size, nb_arrays, has_missing_shm


def _display_projects_info():

    console = Console()

    projects = Project2Toml.list()

    if not projects:
        console.print("\n[red]No cute-shm project found.[/red]\n")
        return

    table = Table(title="[bold]Projects Information[/bold]")
    table.add_column("Project Name", style="cyan")
    table.add_column("TOML Path", style="magenta")
    table.add_column("Number of arrays", style="blue", justify="center")
    table.add_column("Size", style="green", justify="center")
    table.add_column("Integrity", style="yellow", justify="center")

    for project, toml_path in projects.items():
        try:
            total_size, nb_arrays, has_missing_shm = _project_info(project, toml_path)
            size_str = cute_shm.bytes_to_human(total_size)
            integrity = "[green]✓[/green]" if not has_missing_shm else "[red]✗[/red]"
            table.add_row(project, str(toml_path), str(nb_arrays), size_str, integrity)
        except Exception as e:
            table.add_row(
                project, str(toml_path), "[red]Error[/red]", "[red]Error[/red]"
            )
            console.print(f"[red]Error processing project '{project}': {str(e)}[/red]")

    console.print(table)


_row: TypeAlias = tuple[str, str, str, str, str, str]


def _display_project_full(project: str, toml_path: Path, console: Console) -> bool:

    has_missing_shm = False

    if not toml_path.exists():
        raise FileNotFoundError(f"TOML metadata file not found: {toml_path}")

    with open(toml_path, "rb") as f:
        meta_dict = tomli.load(f)

    rows: list[_row] = []

    def add_rows(prefix, d) -> tuple[int, bool]:
        missing_shm = False
        total_size = 0
        for k, v in d.items():
            key_path = f"{prefix}.{k}" if prefix else k
            if "shm_name" in v:
                shape = str(v["shape"])
                dtype = v["dtype"]
                size = array_size(v)
                total_size += size
                size_str = cute_shm.bytes_to_human(size)
                attrs = ", ".join(v["attrs"].keys())
                shm_name = v["shm_name"]
                if not Path(f"/dev/shm/{shm_name}").exists():
                    missing_shm = True
                    shm_name = "[red]* MISSING *[/red]"
                rows.append((str(key_path), shape, dtype, size_str, attrs, shm_name))
            else:
                ts, ms = add_rows(key_path, v)
                total_size += ts
                if ms:
                    missing_shm = True
        return total_size, missing_shm

    total_size, has_missing_shm = add_rows("", meta_dict)

    title = f"[bold]Project: [blue]{project}[/blue][/bold] - [italic]{toml_path}[/italic] - [red]{cute_shm.bytes_to_human(total_size)}[/red]"
    table = Table(title=title)
    table.add_column("Key Path", justify="left", style="cyan", no_wrap=True)
    table.add_column("Shape", justify="right", style="magenta")
    table.add_column("Dtype", justify="right", style="green")
    table.add_column("Size", justify="right", style="red")
    table.add_column("Attributes", justify="right", style="yellow")
    table.add_column("Shared Memory Name", justify="right", style="blue")
    for row_ in rows:
        table.add_row(*row_)

    console.print(table)

    return has_missing_shm


def _display_projects_full() -> None:
    """
    Display all projects and their metadata.
    """
    console = Console()

    projects = Project2Toml.list()

    if not projects:
        console.print("\n[red]No cute-shm project found.[/red]\n")
        return

    project_with_missing_shm: list[str] = []

    for project, toml_path in projects.items():
        console.print("")
        try:
            has_missing_shm = _display_project_full(project, toml_path, console)
            if has_missing_shm:
                project_with_missing_shm.append(project)
        except tomli.TOMLDecodeError as e:
            console.print(
                f"[red]cute-shm project error, failed to parse TOML metadata file {toml_path}: {e}[/red]"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    if project_with_missing_shm:
        console.print(
            f"[red]|\n| Warning ! some project(s) has missing shared memory segment(s): [bold]{', '.join(project_with_missing_shm)}[/bold]\n|[/red]"
        )


def display_projects():
    """
    Display projects based on the optional '-short' command line argument.
    If '-short' is provided, displays only project info.
    Otherwise, displays full project details.
    """

    parser = argparse.ArgumentParser(description="Display cute-shm projects")
    parser.add_argument("-short", action="store_true", help="Display only project info")
    args = parser.parse_args()

    if args.short:
        _display_projects_info()
    else:
        _display_projects_full()
