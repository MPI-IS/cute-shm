import os
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional

import psutil
from rich.progress import BarColumn, Progress, TextColumn

from .core import bytes_to_human


class _Thr:
    def __init__(self, period: float = 0.1) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop: Optional[threading.Event] = None
        self._period = period

    def start(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None and self._stop is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._stop = None
        self._on_exit()

    def _run(self) -> None:
        if self._stop is None:
            raise TypeError("the start function should heave been called")
        while not self._stop.is_set():
            with self._lock:
                self._iterate()
            time.sleep(self._period)

    def _iterate(self) -> None:
        raise NotImplementedError

    def _on_exit(self) -> None:
        return


class _ShmTransfer(_Thr):
    def __init__(self) -> None:
        super().__init__()
        self._process = psutil.Process(os.getpid())
        self._transferred: int = 0
        self._prv: Optional[int] = None

    def get(self) -> int:
        with self._lock:
            return self._transferred

    def _iterate(self) -> None:
        mem_info = self._process.memory_info()
        shm_memory = mem_info.rss
        if self._prv is None:
            self._prv = shm_memory
        if shm_memory < self._prv:
            self._prv = None
        else:
            incr = shm_memory - self._prv
            self._transferred += incr
            self._prv = shm_memory
        time.sleep(0.1)


class ShmProgress(_Thr):
    """
    A class to track and display progress of shared memory transfers.

    Args:
        total_bytes (int): The total number of bytes to be transferred.
    """

    def __init__(self, total_bytes: int) -> None:
        super().__init__(period=1)
        self._started = False
        self._shm_transfer = _ShmTransfer()
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[custom_text]}"),
        )
        self._total_bytes = total_bytes
        self._transferred = self._progress.add_task(
            "[green]Uploaded data:",
            total=total_bytes,
            custom_text="starting",
        )

    def _iterate(self) -> None:
        if not self._started:
            self._shm_transfer.start()
            self._progress.start()
            self._started = True
        shm_transfer = self._shm_transfer.get()
        transfer_progress = str(
            f"{bytes_to_human(shm_transfer)} / {bytes_to_human(self._total_bytes)}"
        )
        self._progress.update(
            self._transferred,
            completed=shm_transfer,
            custom_text=transfer_progress,
        )

    def _on_exit(self) -> None:
        self._shm_transfer.stop()
        self._progress.stop()


@contextmanager
def shm_progress(total_size: int) -> Generator[ShmProgress, None, None]:
    """
    A context manager to create and manage a ShmProgress instance.

    Args:
        total_size (int): The total size of the data to be transferred.

    Yields:
        ShmProgress: An instance of ShmProgress to track transfer progress.
    """
    progress = ShmProgress(total_size)
    progress.start()
    try:
        yield progress
    finally:
        progress.stop()
