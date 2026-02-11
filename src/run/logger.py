"""
Logging utilities with tqdm support and distributed awareness.

Mirrors the ICML codebase logger pattern.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
from tqdm.auto import tqdm

from src.run.distributed import is_main_process


class DistributedFilter(logging.Filter):
    """Filter that only allows logging from main process in distributed training."""

    def filter(self, record):
        if getattr(record, "_all_ranks", False):
            return True
        return is_main_process()


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that plays nicely with tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def _create_formatter(process_id: Optional[int] = None) -> logging.Formatter:
    """Create a logging formatter with optional process ID prefix."""
    if process_id is not None:
        format_str = f"[GPU {process_id}] [%(asctime)s] [%(levelname)s] %(message)s"
    else:
        format_str = "[%(asctime)s] [%(levelname)s] %(message)s"
    return logging.Formatter(format_str, datefmt="%H:%M:%S")


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    distributed_aware: bool = True,
    process_id: Optional[int] = None,
    multiprocessing_queue: Optional[mp.Queue] = None,
) -> logging.Logger:
    """
    Set up a logger instance with optional file output and multiprocessing support.

    In distributed training, automatically filters logs to only show from rank 0.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers = []

    if distributed_aware and multiprocessing_queue is None:
        logger.addFilter(DistributedFilter())

    formatter = _create_formatter(process_id)

    if multiprocessing_queue is not None:
        queue_handler = QueueHandler(multiprocessing_queue)
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)
    else:
        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None and (not distributed_aware or is_main_process()):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def setup_multiprocess_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
) -> tuple:
    """Set up multiprocessing-safe logging with a queue."""
    manager = mp.Manager()
    log_queue = manager.Queue()

    formatter = _create_formatter()
    handlers = []

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)

    return log_queue, queue_listener, manager


def get_tqdm_kwargs(logger: logging.Logger, **kwargs) -> dict:
    """Get tqdm kwargs based on logger level. Progress bars only in DEBUG mode."""
    kwargs.setdefault("file", sys.stderr)
    kwargs["disable"] = not (
        logger.isEnabledFor(logging.DEBUG) and is_main_process()
    )
    return kwargs

