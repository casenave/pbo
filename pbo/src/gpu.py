# Generic imports
import os

# Custom imports
import tensorflow as tf

# Environment variable used to cap the GPU footprint (in MiB)
MEMORY_LIMIT_ENV = 'PBO_GPU_MEMORY_LIMIT_MB'

########################
# Read the memory limit from the environment
########################
def memory_limit_from_env():
    """Return the per-GPU memory cap in MiB, or None when no cap is set.

    An empty or non-positive value means "no cap". Invalid values raise a
    ValueError instead of being silently ignored, so a typo cannot quietly
    reintroduce the full preallocation.
    """
    raw = os.environ.get(MEMORY_LIMIT_ENV, '').strip()
    if (not raw): return None

    try:
        limit = int(float(raw))
    except ValueError as err:
        raise ValueError(f'{MEMORY_LIMIT_ENV} must be an integer number of MiB, got {raw!r}') from err

    if (limit <= 0): return None

    return limit

########################
# Limit TensorFlow's GPU memory footprint
########################
def configure_gpus(memory_limit_mb=None):
    """Limit the GPU memory TensorFlow reserves.

    TensorFlow pre-allocates every byte of free GPU memory at the first
    operation, which starves any other process sharing the device (desktop,
    second training job, notebook). Two modes are available:

    - memory_limit_mb set (directly or through PBO_GPU_MEMORY_LIMIT_MB): each
      GPU only reserves that many MiB;
    - otherwise: memory growth, so allocation follows actual usage.

    The configuration only applies before devices are initialized. When called
    later (for instance from a test that already ran a kernel) the request is
    skipped with a warning rather than failing the run.

    Returns the logical GPU devices visible to TensorFlow.
    """
    if (memory_limit_mb is None):
        memory_limit_mb = memory_limit_from_env()

    gpus = tf.config.list_physical_devices('GPU')
    if (not gpus): return []

    try:
        if (memory_limit_mb):
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)])
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as err:
        print('Warning : GPU devices already initialized, memory config ignored ('+str(err)+')')
        return tf.config.list_logical_devices('GPU')

    return tf.config.list_logical_devices('GPU')
