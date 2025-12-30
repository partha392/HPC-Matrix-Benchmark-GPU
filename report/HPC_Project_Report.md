PERFORMANCE ANALYSIS OF PARALLEL MATRIX COMPUTATION USING CUDA-ENABLED GPUS
A PROJECT REPORT

Submitted in partial fulfillment of the requirements for the
ACC HPC Course
under C-DAC CINE

CERTIFICATE

This is to certify that the project report entitled “Performance Analysis of Parallel Matrix Computation using CUDA-Enabled GPUs” is a bona fide work carried out by Partha Pratim Das and Manash Das under the supervision of C-DAC (CINE).
This work has not been submitted elsewhere for any other degree or diploma.

Department: C-DAC (CINE), IIT Guwahati
Date: 31-12-2025

DECLARATION

We hereby declare that the project entitled “Performance Analysis of Parallel Matrix Computation using CUDA-Enabled GPUs” submitted for the ACC HPC Course is our original work and has not formed the basis for the award of any degree, associateship, fellowship, or any other similar title.

Partha Pratim Das — Roll No: []
Manash Das — Roll No: []

ABSTRACT

High Performance Computing (HPC) is a foundational pillar of modern scientific computing, machine learning, and large-scale data analytics. Among HPC workloads, dense matrix multiplication is one of the most computationally intensive operations and serves as a core building block for numerous real-world applications.

This project evaluates the performance benefits of GPU-accelerated matrix multiplication using NVIDIA CUDA-enabled GPUs. A comparative benchmark is conducted between a CPU-based implementation using NumPy and a GPU-based implementation using CuPy, a CUDA-accelerated NumPy-compatible library. Experiments are executed on Google Colab using an NVIDIA Tesla T4 GPU.

For large matrices (up to
4000
×
4000
4000×4000), the GPU-based implementation achieves one to two orders of magnitude speedup compared to CPU execution under the tested environment. The results clearly demonstrate the effectiveness of GPU parallelism for data-parallel linear algebra workloads.

Keywords: High Performance Computing, GPU Computing, CUDA, CuPy, Matrix Multiplication, Parallel Processing

1. INTRODUCTION

The exponential growth of data-intensive applications has exposed the performance limitations of traditional CPU-based computing. While CPUs are optimized for low-latency control and serial execution, they offer limited throughput for massively parallel numerical workloads.

To address this challenge, heterogeneous computing has emerged as a dominant paradigm, combining CPUs with specialized accelerators such as GPUs. GPUs provide thousands of lightweight cores capable of executing the same instruction concurrently across large datasets.

Matrix multiplication is selected as the core workload in this project because it is fundamental to scientific simulations, image processing, and machine learning pipelines. By comparing CPU-based and GPU-based implementations, this project quantifies the practical performance gains achievable through GPU acceleration.

1. PROBLEM STATEMENT

Dense matrix multiplication has a computational complexity of
𝑂
(
𝑁
3
)
O(N
3
). As matrix size increases, CPU execution time grows rapidly, making large-scale computation inefficient.

Key Challenges

Limited parallelism on CPU architectures

Memory bandwidth bottlenecks

Underutilization of arithmetic units for data-parallel workloads

Objective
To evaluate whether GPU-accelerated computation can significantly reduce execution time for large matrix multiplication tasks.

1. OBJECTIVES

Implement matrix multiplication using CPU-based numerical libraries

Implement the same operation using GPU-accelerated libraries

Benchmark performance across large matrix sizes

Analyze speedup achieved through GPU parallelism

Demonstrate a reproducible HPC experiment using a cloud environment

1. BACKGROUND STUDY
4.1 High Performance Computing

High Performance Computing refers to the use of advanced computational techniques to solve problems requiring large-scale processing power. Modern HPC increasingly relies on accelerators such as GPUs to achieve high throughput.

4.2 GPU Computing

CPU: Few powerful cores, optimized for latency

GPU: Thousands of simpler cores, optimized for throughput

GPUs excel at executing identical operations across large datasets using SIMD-style parallelism.

4.3 CUDA Architecture

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform that enables execution of code on GPUs. The model consists of a host (CPU), a device (GPU), and kernels executed in parallel by GPU threads.

4.4 CuPy Library

CuPy is a NumPy-compatible library that executes array operations on NVIDIA GPUs using CUDA. It internally leverages highly optimized CUDA libraries such as cuBLAS, enabling GPU acceleration with minimal code changes.

1. SYSTEM ARCHITECTURE

Input Layer: Random matrix generation

Computation Layer:

CPU Path: NumPy execution on system RAM

GPU Path: CuPy execution on GPU VRAM

Analysis Layer: Timing measurement and speedup computation

1. METHODOLOGY
6.1 CPU Implementation

Library: NumPy

Data Type: float32

Operation: A @ B (BLAS-optimized)

6.2 GPU Implementation

Library: CuPy

GPU memory allocation using cp.random.rand()

Warm-up execution to exclude CUDA initialization overhead

Explicit synchronization before timing

6.3 Benchmarking Strategy

Verify GPU availability

Fix matrix size
𝑁
=

4000
N=4000

Measure execution time

Compute speedup ratio

1. EXPERIMENTAL SETUP

Platform: Google Colab

OS: Linux (Ubuntu 22.04)

Language: Python 3.10

CPU: Intel Xeon (2 vCPUs)

GPU: NVIDIA Tesla T4 (16 GB VRAM)

Libraries: NumPy 1.25.x, CuPy 12.x

1. RESULTS AND DISCUSSION
8.1 Performance Results
Metric CPU (NumPy) GPU (CuPy)
Matrix Size 
4000
×
4000
4000×4000 
4000
×
4000
4000×4000
Execution Time 12–18 s 0.08–0.15 s
8.2 Speedup Analysis

Observed speedup ranged from approximately 80× to 180×, depending on system load and runtime conditions on the shared cloud infrastructure.

8.3 Discussion on Overheads

For small matrices, GPU acceleration is limited by kernel launch latency and memory transfer overhead. For large matrices, computation dominates, making GPU execution significantly faster.

1. LIMITATIONS

GPU memory constraints limit maximum matrix size

Performance varies due to shared cloud resources

Comparison is based on high-level libraries, not custom CUDA kernels

1. CONCLUSION

This project demonstrates that GPU-accelerated computing provides substantial performance improvements for data-parallel workloads such as matrix multiplication. The results confirm that GPUs are essential accelerators for modern HPC applications and that cloud platforms enable accessible experimentation with advanced hardware.

1. FUTURE SCOPE

Multi-GPU execution using NCCL

Mixed-precision computation using Tensor Cores

Custom CUDA kernel optimization

1. REFERENCES

NVIDIA Corporation, CUDA Programming Guide, 2024

CuPy Documentation — <https://docs.cupy.dev>

Harris et al., Array Programming with NumPy, Nature, 2020

Google Colab Documentation
