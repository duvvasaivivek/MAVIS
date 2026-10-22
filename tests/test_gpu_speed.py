"""
PyTorch GPU Acceleration Proof & Benchmark
==========================================
Verifies CUDA availability and compares CPU vs GPU GEMM performance.
"""

import time
import torch


def benchmark_device_speed(matrix_size: int = 8192):
    print("=" * 50)
    print("PyTorch GPU ACCELERATION BENCHMARK")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("[WARN] No CUDA GPU detected. Running on CPU only.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")
    print(f"\nPerforming {matrix_size}x{matrix_size} Matrix Multiplication...")

    # 1. CPU Run
    print("\n[1] Running on CPU...")
    a_cpu = torch.randn(matrix_size, matrix_size)
    b_cpu = torch.randn(matrix_size, matrix_size)

    start = time.time()
    _ = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"    CPU Time: {cpu_time:.4f} seconds")

    # 2. GPU Run
    print("\n[2] Running on GPU...")
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()

    # Warmup
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    start = time.time()
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"    GPU Time: {gpu_time:.4f} seconds")

    print("\n" + "=" * 50)
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"RESULT: GPU is {speedup:.1f}x FASTER than CPU!")
    print("=" * 50)


if __name__ == "__main__":
    benchmark_device_speed()
