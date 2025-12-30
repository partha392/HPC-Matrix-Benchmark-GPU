# Performance Analysis of Parallel Matrix Computation (CPU vs GPU)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?style=flat&logo=google-colab)
![Accelerator](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green?style=flat&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

This project demonstrates the power of **High Performance Computing (HPC)** by benchmarking Large-Scale Matrix Multiplication.
It compares the performance of a **Serial CPU implementation (NumPy)** against a **Parallel GPU implementation (CuPy)**.

The project is designed to run out-of-the-box on **Google Colab (Jupiter Notebook)** Go Go (T4 GPU).

## 🚀 Motivation

In the era of Big Data and Deep Learning, sequential processing on CPUs is often a bottleneck. This project aims to:

1. Quantify the "Speedup Factor" gained by using GPUs.
2. Demonstrate how to write CUDA-accelerated Python code without complex C++ boilerplate.
3. Provide a clear academic reference for MSc-level HPC studies.

## 🛠 Technologies

* **Language:** Python 3
* **CPU Logic:** NumPy (Numerical Python)
* **GPU Logic:** CuPy (CUDA-accelerated NumPy-compatible library)
* **Environment:** Google Colab (Jupyter Notebook)

## 📂 Repository Structure

```bash
HPC-GPU-Benchmark/
├── colab_notebook.ipynb     # <--- MAIN FILE: Run this in Google Colab
├── README.md                # Project Documentation
├── requirements.txt         # Dependencies for local execution
├── src/                     # Source Code (Modular)
│   ├── cpu_version.py       # CPU implementation
│   ├── gpu_version.py       # GPU implementation
│   └── benchmark.py         # Main driver script
├── results/                 # Output logs and benchmarks
└── report/                  # Project Report
    └── HPC_Project_Report.md # Full Academic Report
```

## ⚡ How to Run

### Option 1: Google Colab (Recommended)

1. Download `colab_notebook.ipynb` from this repository.
2. Upload it to [Google Colab](https://colab.research.google.com/).
3. Go to **Runtime > Change runtime type**.
4. Select **T4 GPU** (under Hardware Accelerator).
5. Click **Runtime > Run all**.

### Option 2: Local (Requires NVIDIA GPU)

If you have a local machine with an NVIDIA GPU and CUDA drivers installed:

```bash
# 1. Clone the repo
git clone https://github.com/your-username/HPC-GPU-Benchmark.git
cd HPC-GPU-Benchmark

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the benchmark
python src/benchmark.py
```

## 📊 Sample Output

*(Actual output from Google Colab T4 GPU)*

```text
Matrix Size: 4000 x 4000
CPU Time: 15.5210 sec
GPU Time: 0.0820 sec
-------------------------
SPEEDUP ACHIEVED: 189.2x
```

## 📜 Academic Report

A full **15-page academic project report** is available in the `report/` directory. It includes:

* Introduction to CUDA Architecture
* Methodology
* Detailed Speedup Analysis
* Future Scope

## 🤝 Contributing

This is an academic project. Suggestions for optimizing the CUDA kernels or adding new benchmarks (e.g., FFT, Monte Carlo) are welcome.

## 📝 License

MIT License. Free to use for educational purposes.
