# vLLM Setup for DGX Spark (Blackwell GB10)

This directory contains vendored scripts for building vLLM from source on NVIDIA DGX Spark with Blackwell GB10 GPUs (SM_121 architecture).

## Attribution

These scripts are based on [eelbaz/dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup) (MIT License), adapted for the spark-vtuber project.

## Why Build from Source?

The standard vLLM pip package doesn't support DGX Spark's ARM64 + Blackwell architecture. This script:

1. Builds Triton from source with SM_121a support
2. Applies critical patches for Blackwell MOE kernels
3. Builds vLLM with CUDA 13.0 support

## Requirements

- NVIDIA DGX Spark with GB10 GPU
- CUDA 13.0+ toolkit installed
- Python 3.12 with development headers (python3-dev, python3.12-dev)
- ~50GB free disk space during build (final installation ~10-15GB)
- ~20-30 minutes build time

**Note:** The installer will automatically install system dependencies including python3-dev if not present.

## Installation

From the spark-vtuber root directory:

```bash
# Run the vLLM installer
bash scripts/vllm/install_vllm.sh --install-dir ./vllm-install

# After installation, activate the environment
source ./vllm-install/vllm_env.sh
```

## Usage

After installation, you can start the vLLM OpenAI-compatible server:

```bash
cd vllm-install

# Start server with default model (Qwen2.5-0.5B-Instruct)
./vllm-serve.sh

# Or specify a local model path (after running download_models.sh)
# First load takes ~4-5 minutes to initialize the model
./vllm-serve.sh "../models/qwen3-30b-a3b-awq" 8000

# Check status
./vllm-status.sh

# Stop server
./vllm-stop.sh
```

## Integration with spark-vtuber

The spark-vtuber LLM module can connect to vLLM's OpenAI-compatible API:

```bash
# In .env
LLM__VLLM_API_URL=http://localhost:8000/v1
LLM__MODEL_NAME=./models/qwen3-30b-a3b-awq
```

## Files

- `install_vllm.sh` - Main installation script
- `patches/gpt_oss_triton_moe.patch` - Patch for Qwen3/MoE support
- `helpers/vllm-serve.sh` - Start vLLM server
- `helpers/vllm-status.sh` - Check server status
- `helpers/vllm-stop.sh` - Stop server

## Pinned Versions

- vLLM: `66a168a197ba214a5b70a74fa2e713c9eeb3251a`
- Triton: `4caa0328bf8df64896dd5f6fb9df41b0eb2e750a`
- PyTorch: 2.9.0+cu130
- Python: 3.12

## Troubleshooting

### CMake Error: "Could NOT find Python3 (missing: Development.Module)"

This error means Python development headers are not installed. The installer handles this automatically, but if you see this error:

```bash
sudo apt-get install -y python3-dev python3.12-dev
```

Then re-run the vLLM installation script.

### Other Issues

See the [NVIDIA DGX Spark vLLM Troubleshooting Guide](https://build.nvidia.com/spark/vllm/troubleshooting) for common issues.
