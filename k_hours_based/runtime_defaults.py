import os


BUILD_DEFAULT_WORKERS = max(1, int(os.cpu_count() or 1) - 3)
BUILD_DEFAULT_CHUNK_SIZE = 250
BUILD_CSV_WRITE_BATCH_SIZE = 1000

SPLIT_DEFAULT_CHUNK_SIZE = 160000
SPLIT_DEFAULT_ROOM_SPLIT_WORKERS = max(1, int(os.cpu_count() or 1) - 3)
SPLIT_UNSORTED_ACTIVE_ROOM_BUFFER = 20
SPLIT_CSV_WRITE_BATCH_SIZE = 160000

FEDERATED_DEFAULT_CHUNKSIZE = 50000
DEFAULT_OUTPUT_THRESHOLDS = {
    "y_temp_main": 1.0,
    "y_temp_toilet": 1.0,
    "y_light": 5.0,
    "y_sound": 5.0,
}
DEFAULT_CHANGE_THRESHOLD = 0.5
DEFAULT_AIRFLOW_THRESHOLD = 0.5


def reserved_cpu_worker_limit() -> int:
    return max(1, int(os.cpu_count() or 1) - 3)


def default_build_workers() -> int:
    return reserved_cpu_worker_limit()


def default_federated_workers() -> int:
    return reserved_cpu_worker_limit()


def default_federated_client_cpu() -> float:
    total_cpus = max(1, int(os.cpu_count() or 1))
    if total_cpus <= 1:
        return 1.0
    effective_workers = reserved_cpu_worker_limit()
    return total_cpus / float(effective_workers)
