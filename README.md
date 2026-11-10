# FreeToken & Throttle: Edge-Native MoE Serving Benchmarks

This repository contains empirical benchmarks of **FreeToken** (an edge-native Mixture-of-Experts serving engine from UC Berkeley/MIT) running on consumer hardware (NVIDIA RTX 5060), measured using **Throttle** (a rigorous statistical cost-per-token measurement tool).

## Hardware & Software Environment
*   **GPU:** NVIDIA GeForce RTX 5060 (8 GB VRAM)
*   **OS:** Windows 11 + WSL2 (Ubuntu)
*   **Model:** `cognitivecomputations.Dolphin3.0-Llama3.1-8B-GGUF` (Q4_K_S quantization)
*   **Context:** 131,072 tokens (Offloaded to system RAM due to VRAM constraints)
*   **Tools:** FreeToken 0.2.0-beta, Throttle 0.6.2, Python 3.11

## Methodology
The benchmarks were run using `throttle check` against the FreeToken OpenAI-compatible API (`http://localhost:1919`). 
To ensure statistical rigor, Throttle was configured to measure the run-to-run noise (baseline calibration) and to enforce the same workload for valid comparisons.

## Results

### 1. Baseline: Concurrency 2 (Max 64 tokens)
*   **Cost:** **$1.30 / Million Output Tokens** (95% CI: $1.17 - $1.42)
*   **Latency:** ~2.26s per block of 256 tokens
*   **Observation:** At low concurrency, the GPU exhibits high idle time between requests, inflating the cost per token.

### 2. Optimized: Concurrency 4 (Max 64 tokens)
*   **Cost:** **$0.81 / Million Output Tokens** (95% CI: $0.76 - $0.86)
*   **Latency:** ~1.50s per block of 256 tokens
*   **Observation:** Increasing concurrency by 2x reduced the cost by **38%**. This proves that FreeToken efficiently utilizes the GPU when it has more parallel requests to hide the memory offloading latency.

### 3. Stress Test: Concurrency 4 (Max 256 tokens)
*   **Cost:** **$0.88 - $0.99 / Million Output Tokens**
*   **Latency:** ~4.70s per block of 700+ tokens
*   **Observation:** Increasing the output token limit introduced a **12% run-to-run noise**. The hardware began to show thermal throttling or Windows background task interference, which Throttle correctly flagged as `NOT CALIBRATED` until the noise bound was measured.

## Key Takeaways for Staff Engineers
1.  **Hardware constraints force architectural depth:** Running an 8B model with 131k context on an 8GB GPU forces the engine to offload to system RAM. FreeToken handles this gracefully, but the cost per token is highly sensitive to the concurrency level.
2.  **Measurement rigor is non-negotiable:** A 12% noise margin means that without a tool like Throttle (which uses Williams Design and LOO validation), an engineer would easily mistake a noisy environment for a performance regression.
3.  **Edge-native serving is viable:** FreeToken proves that consumer hardware can serve frontier MoE models at a predictable cost, provided the workload is properly batched.

## Reproducibility
```bash
# Baseline
throttle check --url http://localhost:1919 --model cognitivecomputations.Dolphin3.0-Llama3.1-8B-GGUF --gpu-hourly-rate 0.50 --label freetoken-baseline --concurrency 2 --max-tokens 64

# Optimized
throttle check --url http://localhost:1919 --model cognitivecomputations.Dolphin3.0-Llama3.1-8B-GGUF --gpu-hourly-rate 0.50 --label freetoken-opt --concurrency 4 --max-tokens 64




#Authored by João Felipe de Souza (github.com/JohnScheuer)
