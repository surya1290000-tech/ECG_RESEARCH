# Table 2: Four-Model Architecture, Parameter Allocation, and Inference Complexity

### Table 2A: Architecture Specifications and Parameter Counts

| Model Name | Canonical Checkpoint Path | Architecture Type | Temporal Pooling Mechanism | Trainable Parameters | Parameter Delta vs. GAP | Checkpoint Disk Size (MB) | SHA-256 Checksum Verification |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **InceptionTime1D** | `checkpoints/model_inception_fold10_best.pth` | Multi-Scale Inception Blocks ($k \in \{9, 19, 39\}$) | Global Average Pooling (GAP) | **3,886,149** | −33,344 (−0.85%) | 14.88 MB | `18277a08339eeb00efaa9a734c2f466b8451e8e0ca7e780edb64b7174e3a582b` |
| **ECGResNet-Attention** | `checkpoints/model_b_fold10_best.pth` | 1D Residual Network (Stem + 4 Stages) | Lightweight Temporal Attention Pooling | **3,920,006** | **+513 (+0.013%)** | 15.01 MB | `8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa` |
| **ECGResNet-GAP** | `checkpoints/model_a_fold10_best.pth` | 1D Residual Network (Stem + 4 Stages) | Uniform Global Average Pooling (GAP) | **3,919,493** | Baseline (0) | 15.01 MB | `995eb2c70ba741842c70d2f2f0c5ba2710d80d7127b0d5cace38183dc87edc39` |
| **XResNet1D** | `checkpoints/model_xresnet_fold10_best.pth` | 1D ResNet with Computer Vision Tweaks | Anti-Aliased Average Pooling (AAP) | **3,931,525** | +12,032 (+0.31%) | 15.06 MB | `aee7bf2e9b37eb11956920113a056daaf56cb9960949c684a38626e79a50499b` |

---

### Table 2B: Measured Inference Complexity on Frozen Fold-10 ($N = 2,158$)

| Model Name | Evaluated Batch Size | Total Test Runtime (s) | Per-Record Latency (ms/ECG) | Inference Throughput (ECGs/s) | Relative Measured Throughput vs. Inception | Relative Measured Throughput vs. GAP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ECGResNet-Attention** | 64 | **34.5 s** | **16.0 ms** | **62.6 ECGs/s** | **+50.5%** | **+51.2%** |
| **InceptionTime1D** | 64 | 51.9 s | 24.1 ms | 41.6 ECGs/s | Baseline (1.00×) | +0.5% |
| **ECGResNet-GAP** | 64 | 52.1 s | 24.1 ms | 41.4 ECGs/s | −0.5% | Baseline (1.00×) |
| **XResNet1D** | 64 | 52.3 s | 24.2 ms | 41.3 ECGs/s | −0.7% | −0.2% |

*Hardware Configuration & Methodological Note*:
1. Measurements conducted on evaluation host CPU (x86_64, PyTorch 2.x, multi-threaded forward inference under `torch.no_grad()`).
2. Inference latency and throughput reflect execution under the evaluated software/hardware configuration and batch size (64). These values characterize implementation efficiency under identical evaluation conditions and should not be assumed to generalize identically to specialized accelerators (GPUs, TPUs) or low-power embedded hardware.
3. Data provenance: `results/phase9/four_model_benchmark/four_model_complexity.csv` and `results/phase10/research_freeze/CHECKPOINT_HASHES.csv`.
