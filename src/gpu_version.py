import time

try:
    import cupy as cp
except ImportError:
    cp = None

def check_gpu_availability():
    """Checks if CUDA GPU is available."""
    if cp is None:
        print("[GPU] CuPy module not found via import.")
        return False
    try:
        count = cp.cuda.runtime.getDeviceCount()
        if count > 0:
            print(f"[GPU] CUDA Device Found: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
            return True
        else:
            print("[GPU] No CUDA Device Found.")
            return False
    except Exception as e:
        print(f"[GPU] Error checking GPU: {e}")
        return False

def run_gpu_benchmark(size):
    """
    Performs Matrix Multiplication on GPU using CuPy.
    Args:
        size (int): Dimension of the square N x N matrix.
    Returns:
        float: Execution time in seconds.
    """
    print(f"[GPU] Allocating {size}x{size} matrices on Device...")
    
    # Allocating directly on GPU memory
    A_gpu = cp.random.rand(size, size, dtype=cp.float32)
    B_gpu = cp.random.rand(size, size, dtype=cp.float32)
    
    print(f"[GPU] Warming up GPU...")
    # Warm-up run (to compile kernels and initialize context)
    warmup = cp.dot(A_gpu[:100, :100], B_gpu[:100, :100])
    cp.cuda.Stream.null.synchronize() # Wait for warm-up to finish
    
    print(f"[GPU] Starting computation for N={size}...")
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    
    # Perform Matrix Multiplication
    C_gpu = cp.dot(A_gpu, B_gpu)
    
    end_event.record()
    end_event.synchronize() # Wait for computation to finish
    
    execution_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0 # Convert ms to seconds
    
    print(f"[GPU] Computation finished in {execution_time:.4f} seconds.")
    return execution_time
