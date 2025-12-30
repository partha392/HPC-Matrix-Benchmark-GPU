import time
import numpy as np

def run_cpu_benchmark(size):
    """
    Performs Matrix Multiplication on CPU using NumPy.
    Args:
        size (int): Dimension of the square N x N matrix.
    Returns:
        float: Execution time in seconds.
    """
    print(f"[CPU] Allocating {size}x{size} matrices...")
    # Initialize random matrices with float32 for fair comparison with GPU default
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    print(f"[CPU] Starting computation for N={size}...")
    start_time = time.time()
    
    # Perform Matrix Multiplication
    C = np.dot(A, B)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"[CPU] Computation finished in {execution_time:.4f} seconds.")
    return execution_time
