# FreeToken & Throttle: Edge-Native MoE Inference Benchmarks on Consumer Hardware

This repository contains empirical benchmarks of **FreeToken** (arXiv:2608.16157), 
an edge-native Mixture-of-Experts serving engine from UC Berkeley, MIT, and UT Austin, 
running on consumer hardware (NVIDIA RTX 5060 8 GB), measured using **Throttle** 
(a rigorous statistical cost-per-token measurement tool).

---

## About FreeToken

FreeToken is an open-source MoE inference engine that treats a personal machine 
"not as a small GPU, but as a unified, elastic inference platform." Instead of 
statically assigning experts to devices at load time (as llama.cpp does), it 
continuously recomputes the optimal CPU/GPU split using a bandwidth-adaptive 
scheduling formula derived from each machine's measured PCIe and memory bandwidth.

**Authors:** Shuo Yang, Xiaoze Fan, Melissa Pan, Haocheng Xi, Zhe Wang, Shanlin Sun, 
Kurt Keutzer, Song Han, Matei Zaharia, Chenfeng Xu, Ion Stoica  
(UC Berkeley · MIT · UT Austin · FlashML)  
**Paper:** FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive 
Execution (arXiv:2608.16157, August 17, 2026)  
**Repo:** github.com/FlashML-org/FreeToken  
**License:** Apache 2.0

---

## About Throttle

Throttle is a statistical benchmarking tool for LLM serving cost. Rather than 
reporting a single $/M token number, it uses a **Williams Design** (balanced 
Latin square) with **Leave-One-Out (LOO) outlier validation** to produce 
**noise-bounded, reproducible cost estimates** with 95% confidence intervals.

If a run shows >5% noise between blocks, Throttle flags the result as 
**NOT CALIBRATED** — meaning the hardware environment is unstable and the 
numbers should not be trusted.

**PyPI:** `pip install throttle-pro`  
**Version used:** 0.6.2

---

## Hardware & Software Environment

| Component      | Detail                                              |
|----------------|-----------------------------------------------------|
| GPU            | NVIDIA GeForce RTX 5060 (8 GB VRAM)                |
| OS             | Windows 11 + WSL2 (Ubuntu)                         |
| FreeToken      | 0.2.0-beta                                          |
| Throttle       | 0.6.2                                               |
| Python         | 3.11                                                |
| FreeToken API  | OpenAI-compatible endpoint at http://localhost:1919 |

---

## Methodology

All benchmarks were run using `throttle check` against FreeToken's 
OpenAI-compatible API. Each configuration was measured over multiple blocks 
to capture run-to-run variance. Throttle's Williams Design ensures that 
measurement order effects are balanced across blocks.

**GPU hourly rate used:** $0.50/hr  
*(Representative of consumer-grade cloud GPU rental, e.g. RunPod RTX 4000-class)*

---

## Results

### Benchmark 1 — Dense Model Pipeline Validation
**Model:** cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF (Q4_K_S)  
**Purpose:** Validate the Throttle measurement pipeline before running the 
MoE test. Dolphin 3.0 is a **dense** (non-MoE) model — it does not exercise 
FreeToken's expert offloading scheduler. It was used here to confirm that 
the Throttle ↔ FreeToken API integration was working correctly and that 
baseline cost numbers were stable.

#### 1a. Baseline — Concurrency 2, Max 64 tokens

| Metric       | Value                        |
|--------------|------------------------------|
| Cost         | $1.30 / M output tokens      |
| 95% CI       | $1.17 – $1.42                |
| Latency      | ~2.26 s per block (256 tok)  |
| Status       | CALIBRATED                   |

**Observation:** At low concurrency, the GPU has high idle time between 
requests, inflating cost per token.

#### 1b. Optimized — Concurrency 4, Max 64 tokens

| Metric       | Value                        |
|--------------|------------------------------|
| Cost         | $0.81 / M output tokens      |
| 95% CI       | $0.76 – $0.86                |
| Latency      | ~1.50 s per block (256 tok)  |
| Status       | CALIBRATED                   |

**Observation:** Doubling concurrency reduced cost by **38%** by hiding 
memory-access latency with parallel requests. This validates the measurement 
pipeline and establishes a dense-model cost baseline.

---

### Benchmark 2 — MoE Model: gpt-oss-20b (MXFP4)
**Model:** openai/gpt-oss-20b  
**Architecture:** MoE, 20.91B total parameters, **3.61B active per token**, 
32 experts (top-4), 24 layers, 128K context window  
**Quantization:** Native MXFP4 (trained natively in this precision by OpenAI)  
**Official minimum VRAM:** 16 GB  
**Test hardware VRAM:** 8 GB  

> **Important context:** gpt-oss-20b officially requires 16 GB VRAM 
> (per OpenAI's model card). Running it on 8 GB forces FreeToken into full 
> expert-offload mode: MoE expert weights reside in system RAM, with an 
> LRU cache of hot experts on the GPU. Cache misses are streamed over PCIe 
> on demand. This is FreeToken's core design — the question this benchmark 
> answers is: **at what cost does "runs" translate to "cost-effective"?**

#### 2a. MoE Stress Test — Concurrency 4, Max 64 tokens

| Metric       | Value                         |
|--------------|-------------------------------|
| Cost         | $19.17 / M output tokens      |
| 95% CI       | $18.17 – $20.16               |
| Latency      | ~18 s per block (136 tok)     |
| Status       | CALIBRATED                    |

**Observation:** Cost is **~24× higher** than the dense 8B baseline at the 
same concurrency. The bottleneck is PCIe bandwidth: on every token, 
FreeToken must stream expert weights that are not in the GPU's LRU cache 
from system RAM over PCIe. On an 8 GB GPU, the cache hit rate is very low 
for a 20.91B model, making nearly every expert fetch a cache miss.

The model ran. The numbers are real. But **"runs" ≠ "cost-effective"** on 
this hardware configuration.

---

## Key Takeaways for Staff Engineers

**1. FreeToken's sweet spot is not 20B-class models on 8 GB VRAM.**  
FreeToken's paper benchmarks target Qwen3.6-35B-A3B (~3B active parameters) 
on 8 GB hardware, where the high sparsity ratio (3B active / 35B total) 
gives the LRU expert cache a meaningful hit rate. At 20.91B total / 3.61B 
active on 8 GB, the sparsity advantage is insufficient to offset the 
PCIe streaming overhead. The minimum viable VRAM for cost-effective 
gpt-oss-20b inference is **16 GB**, which matches OpenAI's own specification.

**2. Measurement rigor is non-negotiable.**  
A 12% run-to-run noise margin appeared in the dense baseline stress test 
(Concurrency 4, Max 256 tokens). Without a tool like Throttle — which uses 
Williams Design and LOO validation to distinguish real regressions from 
system noise — an engineer could easily mistake thermal throttling or OS 
scheduling jitter for a performance regression. Throttle correctly flagged 
this result as **NOT CALIBRATED** until the noise source was identified.

**3. The cost explosion is predictable and architectural, not a bug.**  
The $19.17/M result is not an anomaly. It is the expected consequence of 
running a model that exceeds VRAM capacity on hardware with a PCIe 4.0 ×16 
bus (~32 GB/s bidirectional). The MoE expert streaming overhead, which 
FreeToken is specifically designed to minimize, cannot be fully mitigated 
when the VRAM is 50% of the model's minimum requirement.

**4. FreeToken proves that edge MoE inference is an engineering problem, 
not a theoretical one.**  
The fact that gpt-oss-20b (21B MoE, MXFP4) ran at all on an 8 GB consumer 
GPU — producing valid, reproducible output — is itself a significant result. 
The right hardware (16 GB VRAM minimum, fast PCIe 5.0 bus) would change 
the cost picture substantially.

---

## Reproducibility

```bash
# Benchmark 1a — Dense Baseline (Concurrency 2)
throttle check \
  --url http://localhost:1919 \
  --model cognitivecomputations.Dolphin3.0-Llama3.1-8B-GGUF \
  --gpu-hourly-rate 0.50 \
  --label freetoken-dense-c2 \
  --concurrency 2 \
  --max-tokens 64

# Benchmark 1b — Dense Optimized (Concurrency 4)
throttle check \
  --url http://localhost:1919 \
  --model cognitivecomputations.Dolphin3.0-Llama3.1-8B-GGUF \
  --gpu-hourly-rate 0.50 \
  --label freetoken-dense-c4 \
  --concurrency 4 \
  --max-tokens 64

# Benchmark 2 — MoE Stress Test (gpt-oss-20b, MXFP4)
throttle check \
  --url http://localhost:1919 \
  --model gpt-oss-20b \
  --gpu-hourly-rate 0.50 \
  --label freetoken-moe-c4 \
  --concurrency 4 \
  --max-tokens 64

```

## Raw Terminal Outputs

### MoE Benchmark (gpt-oss-20b)
*Terminal output showing the $19.17/M cost and 18s latency on an 8GB RTX 5060.*
![MoE Benchmark](moe-gpt-oss-20b.png.PNG)

References
FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution — arXiv:2608.16157 (Yang et al., UC Berkeley / MIT / UT Austin, August 2026)
gpt-oss-20b Model Card — OpenAI Deployment Safety Hub
(20.91B total, 3.61B active, 32 experts top-4, 16 GB minimum VRAM)
Throttle: Statistical Cost-Per-Token Measurement for LLM Serving — github.com/JohnScheuer/throttle / PyPI: throttle-pro
Benchmarks conducted and authored by João Felipe de Souza (github.com/JohnScheuer) — November 2026.
