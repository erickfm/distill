"""
Parallel GPU experiment execution.

Mirrors the ICML codebase parallel.py — runs multiple configs across GPUs
with automatic GPU assignment and progress tracking.
"""

import os
import time
import logging
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import multiprocessing as mp
import json

mp.set_start_method("spawn", force=True)


def update_configs_json(configs_file: Path, ref_str: str, config: Dict[str, Any]) -> None:
    """Thread-safe function to update configs.json with file locking."""
    configs_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if configs_file.exists() else "w+"
    with open(configs_file, mode) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            content = f.read()
            configs_dict = json.loads(content) if content.strip() else {}
        except json.JSONDecodeError:
            configs_dict = {}

        configs_dict[ref_str] = config
        f.seek(0)
        f.truncate()
        json.dump(configs_dict, f, indent=2, default=str)


def worker_run(task: Tuple) -> Tuple[Dict[str, Any], float]:
    """Worker function that runs a single configuration on a specific GPU."""
    config, gpu_id, log_level, log_queue, res_root = task

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from src.run.utils import get_timestamp
    from src.run.main import run
    from src.run.logger import setup_logger

    worker_logger = setup_logger(
        name=f"worker_gpu_{gpu_id}",
        level=log_level,
        process_id=gpu_id,
        multiprocessing_queue=log_queue,
    )

    start_time = time.time()

    try:
        config["log_level"] = log_level
        config["process_id"] = gpu_id

        main_res_root = Path(res_root)
        gpu_res_root = main_res_root / f"gpu_{gpu_id}"

        timestamp = get_timestamp()
        res_dir = gpu_res_root / f"results_{timestamp}"

        config["res_dir"] = res_dir
        config["timestamp"] = timestamp

        ref_str = f"gpu_{gpu_id}/{res_dir.name}"
        worker_logger.info(f"Starting run at {ref_str}")

        run(**config)

        duration = time.time() - start_time
        worker_logger.info(f"Completed run at {ref_str}, took {format_time(duration)}")

        configs_file = main_res_root / "configs.json"
        update_configs_json(configs_file, ref_str, config)

        return config, duration

    except Exception as e:
        worker_logger.error(f"ERROR in run: {e}")
        raise


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_available_gpus() -> int:
    """Detect number of available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1
    except ImportError:
        return 1


def run_parallel(
    configs: list[Dict[str, Any]],
    res_root: str,
    log_level: str = "INFO",
    num_gpus: int = -1,
) -> None:
    """
    Run multiple experiment configurations in parallel across GPUs.

    Mirrors the ICML codebase run_parallel function.
    """
    from src.run.logger import setup_logger, setup_multiprocess_logging

    res_root = Path(res_root)
    res_root.mkdir(parents=True, exist_ok=True)

    log_file = res_root / "orchestrate.log"
    log_queue, queue_listener, manager = setup_multiprocess_logging(log_file, log_level)
    queue_listener.start()

    main_logger = setup_logger(
        name="orchestrate_main",
        log_file=log_file,
        level=log_level,
    )

    if num_gpus == -1:
        num_gpus = get_available_gpus()
    else:
        available = get_available_gpus()
        if num_gpus > available:
            main_logger.warning(f"Requested {num_gpus} GPUs, only {available} available")
            num_gpus = min(num_gpus, available)

    main_logger.info(f"Total configurations: {len(configs)}")
    main_logger.info(f"Using {num_gpus} GPUs")
    main_logger.info("=" * 100)

    completed = 0
    failed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        gpu_queue = list(range(num_gpus))
        active_futures = {}
        pending_configs = configs.copy()

        for _ in range(min(num_gpus, len(pending_configs))):
            if pending_configs and gpu_queue:
                config = pending_configs.pop(0)
                gpu_id = gpu_queue.pop(0)
                future = executor.submit(
                    worker_run, (config, gpu_id, log_level, log_queue, res_root)
                )
                active_futures[future] = (config, gpu_id)

        while active_futures:
            for done in as_completed(active_futures):
                config, gpu_id = active_futures[done]
                del active_futures[done]
                break

            try:
                _, duration = done.result()
                completed += 1

                if pending_configs:
                    next_config = pending_configs.pop(0)
                    future = executor.submit(
                        worker_run, (next_config, gpu_id, log_level, log_queue, res_root)
                    )
                    active_futures[future] = (next_config, gpu_id)
                else:
                    gpu_queue.append(gpu_id)

            except Exception as e:
                failed += 1
                main_logger.error(f"Task failed on GPU {gpu_id}: {e}")

                if pending_configs:
                    next_config = pending_configs.pop(0)
                    future = executor.submit(
                        worker_run, (next_config, gpu_id, log_level, log_queue, res_root)
                    )
                    active_futures[future] = (next_config, gpu_id)
                else:
                    gpu_queue.append(gpu_id)

            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = len(pending_configs) + len(active_futures)
            est_remaining = avg_time * remaining if remaining > 0 and completed > 0 else 0

            main_logger.info(f"PROGRESS: {completed}/{len(configs)} done, {failed} failed")
            main_logger.info(f"  Elapsed: {format_time(elapsed)} | ETA: {format_time(est_remaining)}")
            main_logger.info("=" * 100)

    main_logger.info("EXECUTION COMPLETE")
    main_logger.info(f"  Total: {len(configs)} | OK: {completed} | Failed: {failed}")
    main_logger.info(f"  Total time: {format_time(time.time() - start_time)}")
    main_logger.info(f"  Output: {res_root}")

    queue_listener.stop()
    manager.shutdown()

