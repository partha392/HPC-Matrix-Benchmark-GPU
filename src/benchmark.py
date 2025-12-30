import os
import time
from cpu_version import run_cpu_benchmark
from gpu_version import run_gpu_benchmark, check_gpu_availability

def save_results(results_str, filepath="results/benchmark_results.txt"):
    """Saves benchmark results to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(results_str)
    print(f"\n[INFO] Results saved to {filepath}")

def main():
    print("="*60)
    print("      HPC PROJECT: GPU vs CPU Matrix Multiplication")
    print("="*60)

    # Project Constraint: Matrix Size
    # Use N=4096 or similar large number to show significant speedup
    # N=2000 is usually enough for Colab T4 vs CPU
    MATRIX_SIZE = 4000 
    
    print(f"Benchmark Configuration:")
    print(f" - Matrix Size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f" - Elements: {MATRIX_SIZE**2:,}")
    print(f" - Operations (approx): {2 * MATRIX_SIZE**3:,} FLOPs")
    print("-" * 60)

    # 1. Check Resources
    has_gpu = check_gpu_availability()
    if not has_gpu:
        print("[WARNING] GPU not found! Speedup test will fail or run on emulated CPU.")
        # We proceed anyway but results will be disappointing if on CPU-only.
    
    results = []
    results.append("HPC PROJECT BENCHMARK RESULTS")
    results.append("=============================")
    results.append(f"Matrix Size: {MATRIX_SIZE}")
    
    # 2. Run CPU Benchmark
    print("\n>>> Running CPU Benchmark (NumPy)...")
    try:
        cpu_time = run_cpu_benchmark(MATRIX_SIZE)
        results.append(f"CPU Time: {cpu_time:.4f} sec")
    except Exception as e:
        print(f"CPU Benchmark failed: {e}")
        cpu_time = float('inf')
        results.append(f"CPU Time: FAILED ({e})")

    # 3. Run GPU Benchmark
    if has_gpu:
        print("\n>>> Running GPU Benchmark (CuPy)...")
        try:
            gpu_time = run_gpu_benchmark(MATRIX_SIZE)
            results.append(f"GPU Time: {gpu_time:.4f} sec")
        except Exception as e:
            print(f"GPU Benchmark failed: {e}")
            gpu_time = float('inf')
            results.append(f"GPU Time: FAILED ({e})")
    else:
        print("\n[SKIP] Skipping GPU Benchmark (No GPU detected)")
        gpu_time = float('inf')
        results.append("GPU Time: N/A")

    # 4. Calculate Speedup
    print("\n" + "-"*60)
    if cpu_time != float('inf') and gpu_time != float('inf') and gpu_time > 0:
        speedup = cpu_time / gpu_time
        summary = f"SPEEDUP ACHIEVED: {speedup:.2f}x"
        print(summary)
        results.append("-" * 30)
        results.append(summary)
    else:
        print("Could not calculate valid speedup.")
    print("="*60)

    # 5. Save Report
    final_output = "\n".join(results)
    save_results(final_output, "../results/benchmark_results.txt")

if __name__ == "__main__":
    main()
