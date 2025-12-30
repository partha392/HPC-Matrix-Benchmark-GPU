# Performance Analysis of Parallel Matrix Computation (CPU vs GPU)

📌 **Project Overview**

This project demonstrates the practical benefits of **High Performance Computing (HPC)** by benchmarking large-scale matrix multiplication on **CPU vs GPU**.

A comparative study is performed between:

* **CPU-based** serial computation using **NumPy**
* **GPU-accelerated** parallel computation using **CuPy** (CUDA)

The project is designed to run out-of-the-box on **Google Colab** using an **NVIDIA Tesla T4 GPU**.

🚀 **Motivation**

In modern scientific computing, machine learning, and data analytics, sequential CPU execution often becomes a performance bottleneck. This project aims to:

1. Quantify the performance improvement achieved through GPU acceleration
2. Demonstrate CUDA-enabled parallel computing using high-level Python libraries
3. Provide an academically sound, reproducible reference for MSc-level HPC coursework

🛠 **Technologies Used**

* **Language:** Python 3.10
* **CPU Computation:** NumPy
* **GPU Computation:** CuPy
* **Parallel Platform:** NVIDIA CUDA
* **Execution Environment:** Google Colab (Jupyter Notebook)

📂 **Repository Structure**

```
HPC-Matrix-Benchmark-GPU/
├── colab_notebook.ipynb        # MAIN FILE (Run on Google Colab)
├── README.md                   # Project Documentation
├── requirements.txt            # Optional local dependencies
├── src/
│   ├── cpu_version.py          # CPU implementation (NumPy)
│   ├── gpu_version.py          # GPU implementation (CuPy)
│   └── benchmark.py            # Benchmark driver
├── results/                    # Benchmark logs / outputs
└── report/
    └── HPC_Project_Report.md   # Full academic project report
```

⚡ **How to Run**

**Option 1: Google Colab (Recommended)**

1. Download `colab_notebook.ipynb` from this repository
2. Upload it to [https://colab.research.google.com](https://colab.research.google.com)
3. Go to **Runtime → Change runtime type**
4. Select **GPU (T4)** as the hardware accelerator
5. Click **Runtime → Run all**

✅ *No local setup required*

**Option 2: Local Execution (Requires NVIDIA GPU)**

Requires:

* NVIDIA GPU
* CUDA drivers
* Compatible CuPy installation

```bash
# Clone the repository
git clone https://github.com/partha392/HPC-Matrix-Benchmark-GPU.git
cd HPC-Matrix-Benchmark-GPU

# Install dependencies
pip install -r requirements.txt

# Run benchmark
python src/benchmark.py
```

📊 **Sample Output**

*(Observed on Google Colab with NVIDIA Tesla T4 GPU)*

```text
Matrix Size: 4000 x 4000
CPU Time: 12.8 – 18.1 sec
GPU Time: 0.08 – 0.15 sec
-------------------------
Observed Speedup: ~80x – 180x
```

⚠️ *Exact performance may vary due to shared cloud infrastructure and runtime conditions.*

📜 **Academic Report**

A detailed academic report is available in the `report/` directory, covering:

* HPC and CUDA architecture overview
* Experimental methodology
* Performance benchmarking and speedup analysis
* Limitations and future scope

📄 File: [report/HPC_Project_Report.md](report/HPC_Project_Report.md)

🤝 **Contributions & Extensions**

This is an academic HPC benchmark project.
Possible extensions include:

* Multi-GPU benchmarking (NCCL)
* Mixed-precision computation using Tensor Cores
* Additional benchmarks (FFT, Monte Carlo, reductions)

Suggestions and improvements are welcome.

📝 **License**

This project is licensed under the MIT License and is free to use for educational and academic purposes.
