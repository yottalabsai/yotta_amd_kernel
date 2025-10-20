# AMD Developer Challenge 2025: Distributed Inference Kernels

This repository contains our optimized implementations for the [AMD Developer Challenge 2025: Distributed Inference Kernels](https://amdchallenge2025.datamonsters.com). We developed high-performance implementations of three critical distributed GPU kernels for single-node 8× AMD MI300X configurations.

## Overview

The challenge focuses on optimizing three fundamental distributed primitives that are essential for modern large language model (LLM) training and inference:

1. **All-to-All Communication** for Mixture-of-Experts (MoE) models
2. **GEMM-ReduceScatter** for tensor parallelism
3. **AllGather-GEMM** for distributed inference


## Key Technical Concepts

### Symmetric Heap
Memory allocated with identical layout across multiple GPUs, enabling direct remote writes at the same relative offsets without complex addressing.

### IPC (Inter-Process Communication)
Direct GPU-to-GPU memory access using `hipIpcGetMemHandle`/`hipIpcOpenMemHandle` for zero-copy data transfer.

### XCD (eXtreme Compute Die)
Hardware component of AMD GPUs. Our optimizations remap thread blocks across MI300X's 8 XCDs for maximum parallelism.

### Cache Modifiers
Directives like `.cg` (cache global) and `.cv` (cache volatile) for controlling cache behavior in GPU memory operations.

## Performance Results

Our optimizations demonstrate significant performance improvements through:
- Communication-computation overlap
- Reduced memory allocations
- Hardware-aware optimizations
- Custom launchers and barriers

The geometric mean performance metric ensures solutions perform well across diverse workloads rather than being tuned for specific cases.

## References

1. [AMD Developer Challenge 2025](https://amdchallenge2025.datamonsters.com)
2. [AMD Instinct MI300X Accelerator](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
3. [ROCm Documentation](https://rocm.docs.amd.com/)
4. [Reference Kernels Repository](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_distributed)

## Acknowledgments

We thank **AMD** for organizing the GPU Optimization Challenge 2025 and providing access to MI300X hardware. We also thank **GPUMode** and all organizers for making this competition possible.
