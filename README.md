# Spark VTuber

An AI-powered VTuber streaming system built for NVIDIA DGX Spark, inspired by Neuro-sama. This project aims to create an autonomous AI personality capable of streaming, gaming, and interacting with viewers in real-time.

## Project Overview

Spark VTuber is a comprehensive AI streaming system that combines:
- **Large Language Models (Qwen3 MoE)** for natural conversation and personality
- **Real-time Text-to-Speech** for voice synthesis (<150ms latency)
- **Live2D Avatar** with audio-driven lip synchronization
- **Game Integration** for autonomous gameplay (Minecraft, turn-based games, rhythm games)
- **Long-term Memory** using RAG (Retrieval-Augmented Generation)
- **Dual AI Personalities** for dynamic interactions ("main" + "evil twin")

### Hardware Target

**NVIDIA DGX Spark (GB10 Grace Blackwell)**
- 128GB unified LPDDR5x memory
- 1 petaFLOP FP4 AI performance
- 273 GB/s memory bandwidth
- NVLink-C2C cache-coherent CPU-GPU interconnect

### Design Goals

- **Sub-500ms response latency** from chat input to audio output
- **8+ hour continuous streaming** capability
- **Single-machine deployment** (no cloud dependencies)
- **Entertaining and engaging** personality and gameplay

## Repository Structure

```
spark-vtuber/
├── README.md                  # This file
├── research/                  # Research and technical analysis
│   ├── README.md              # Research documentation index
│   ├── prompts/               # Research prompts used
│   ├── reports/               # Technical feasibility studies
│   └── references/            # Papers, PDFs, external resources
├── docs/                      # Project documentation (future)
│   ├── architecture/          # System architecture designs
│   └── setup/                 # Setup and deployment guides
├── src/                       # Source code (future)
│   ├── llm/                   # LLM inference engine
│   ├── tts/                   # Text-to-speech system
│   ├── stt/                   # Speech-to-text system
│   ├── avatar/                # Avatar control and animation
│   ├── memory/                # RAG and memory systems
│   └── game/                  # Game integration agents
└── config/                    # Configuration files (future)
```

## Current Status

**Phase:** ⚠️ **Alpha Testing** - Implementation complete, requires critical fixes before production

### ✅ Completed
- Technical feasibility analysis
- Hardware requirement validation
- Model and technology selection
- Architecture design and implementation
- Core LLM inference pipeline (vLLM + transformers fallback)
- TTS integration (Fish Speech 1.5 local inference)
- STT integration (Parakeet TDT 0.6B V2)
- Memory system (ChromaDB + semantic search)
- Avatar control (VTube Studio API)
- Dual AI personality system with LoRA switching
- Chat integration (Twitch IRC)
- Main pipeline orchestration
- CLI interface with Typer
- Performance metrics and instrumentation

### ⚠️ Known Issues (See [Audit Report](docs/AUDIT_REPORT.md))
- Minecraft integration stubbed (not functional)
- Security improvements needed for credential handling

**Next Steps:**
- Test on DGX Spark hardware
- Benchmark actual latency and memory usage
- Production hardening

## Key Technical Decisions

### Architecture Approach

**LoRA-Based Dual AI System** (Recommended)
- Single Qwen3-30B-A3B base model (~15-20GB with AWQ)
- MoE architecture: 30B total params, 3B active per token
- Two LoRA personality adapters (~3GB total)
- Shared memory with personality-tagged entries
- <20ms personality switching latency

### Technology Stack

| Component | Technology | Memory | Latency |
|-----------|-----------|--------|---------|
| LLM | Qwen3-30B-A3B (AWQ) | 15-20GB | 200-400ms |
| TTS | Fish Speech 1.5 | ~12GB | 80-150ms |
| STT | Parakeet TDT 0.6B V2 | ~4GB | <100ms |
| Memory | Mem0 + ChromaDB | 5-10GB | <50ms |
| Avatar | VTube Studio + Live2D | ~2GB | 16-33ms |
| Vision | YOLO-World / Florence-2 | 4-8GB | <50ms |

### Game Integration Strategy

- **Turn-based games:** Direct LLM function calling
- **Minecraft:** Hierarchical control (LLM + Voyager + Baritone)
- **Rhythm games:** Specialized neural network (<50ms latency)

## Research Findings

### Memory Allocation (128GB Total)
- LLM: 15-20GB (Qwen3-30B-A3B MoE with AWQ)
- Dual AI (LoRA): +3GB
- TTS: 2-4GB
- STT: 3-5GB
- Vision: 4-8GB
- Memory/RAG: 5-10GB
- Game agents: 5-10GB
- System: 10-15GB
- **Total:** ~99-132GB (✅ Feasible with optimization)

### Performance Targets
- **Response Latency:** <500ms (chat → audio)
- **TTS Latency:** <150ms
- **Avatar Sync:** 16-33ms (60fps)
- **Uptime:** 8+ hours continuous
- **Memory Efficiency:** <120GB sustained

## Quick Start

### 5-Minute Setup (Using UV)

```bash
# Clone repository
git clone https://github.com/jhacksman/spark-vtuber.git
cd spark-vtuber

# Run automated setup (installs UV, creates venv, installs dependencies)
bash scripts/setup.sh

# Activate virtual environment
source .venv/bin/activate

# Run in test mode (LLM + TTS only, no external dependencies)
uv run spark-vtuber run --no-chat --no-avatar --no-game
```

**First run will download models automatically (~20GB, takes 30-60 minutes).**

### Prerequisites

- **Hardware:** NVIDIA DGX Spark with GB10 Grace Blackwell (or NVIDIA GPU with 20GB+ VRAM)
- **OS:** Ubuntu 22.04 LTS or later
- **CUDA:** 12.3+ with NVIDIA drivers 545+
- **Python:** 3.10+
- **Storage:** 2TB+ NVMe (for models)
- **Network:** Fast connection for model downloads

### vLLM on DGX Spark

DGX Spark uses ARM64 architecture (Grace CPU). The standard vLLM pip package doesn't support ARM64, so we include a vendored build script that compiles vLLM from source with Blackwell SM_121 support:

```bash
# Build vLLM for DGX Spark (~20-30 minutes)
bash scripts/vllm/install_vllm.sh --install-dir ./vllm-install

# Download models first (if not already done)
bash scripts/download_models.sh

# Start vLLM server
source ./vllm-install/vllm_env.sh
./vllm-install/vllm-serve.sh "./models/qwen3-30b-a3b-awq" 8000
```

See [scripts/vllm/README.md](scripts/vllm/README.md) for details.

### Detailed Setup

For complete setup instructions including:
- VTube Studio avatar configuration
- Twitch chat integration
- Model selection and download
- Performance tuning
- Troubleshooting

**See [docs/SETUP.md](docs/SETUP.md)**

### Testing Components

```bash
# Test LLM generation (downloads model on first run)
uv run spark-vtuber test-llm "Hello, who are you?"

# Test TTS synthesis
uv run spark-vtuber test-tts "Testing text to speech" --output test.wav

# Show configuration
uv run spark-vtuber status

# Run with avatar (requires VTube Studio running)
uv run spark-vtuber run --no-chat --no-game

# Run with Twitch chat (requires .env configuration)
uv run spark-vtuber run --no-avatar --no-game

# Full system (all components)
uv run spark-vtuber run
```

## Documentation

### Setup & Usage
- **[Setup Guide](docs/SETUP.md)** - Complete installation and configuration instructions
- **[Audit Report](docs/AUDIT_REPORT.md)** - Technical audit of current implementation (READ THIS FIRST!)

### Research & Analysis
- **[Research Overview](research/README.md)** - Technical research and feasibility studies
- **[Technical Feasibility Analysis](research/reports/technical_feasibility_analysis.md)** - Original feasibility study
- **[Implementation Roadmap](research/reports/technical_feasibility_analysis.md#8-implementation-roadmap)** - Development phases and timeline

### Key Documents
- **[pyproject.toml](pyproject.toml)** - Project dependencies and configuration
- **[.env.example](.env.example)** - Environment configuration template

## Contributing

This is currently a personal research project. Contributions, suggestions, and feedback are welcome via issues and pull requests.

### Development Priorities

1. **Phase 1:** Core LLM inference and chat pipeline
2. **Phase 2:** Avatar integration and lip sync
3. **Phase 3:** Memory and long-term persistence
4. **Phase 4:** Dual AI personality system
5. **Phase 5:** Game integration (text-based)
6. **Phase 6:** Advanced game integration (Minecraft)
7. **Phase 7:** Optimization and production readiness

## Inspiration

This project is inspired by:
- **Neuro-sama** - AI VTuber created by Vedal
- **NVIDIA's AI streaming demos** - Real-time AI interactions
- **Voyager** - LLM-based Minecraft agent
- **MemGPT** - Long-term conversational memory

## License

TBD (Will be determined once core implementation begins)

## Acknowledgments

- NVIDIA for DGX Spark hardware, TensorRT-LLM, and Parakeet TDT
- Alibaba/Qwen team for Qwen3 models
- Fish Audio team for Fish Speech 1.5
- Open-source communities behind ChromaDB and Mem0
- VTube Studio and Live2D for avatar technologies

---

**Project Status:** Alpha Testing
**Last Updated:** January 10, 2026
**Hardware:** NVIDIA DGX Spark (GB10)
