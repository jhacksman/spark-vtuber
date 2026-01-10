# Spark VTuber

An AI-powered VTuber streaming system built for NVIDIA DGX Spark, inspired by Neuro-sama. This project aims to create an autonomous AI personality capable of streaming, gaming, and interacting with viewers in real-time.

## Project Overview

Spark VTuber is a comprehensive AI streaming system that combines:
- **Large Language Models (70B+)** for natural conversation and personality
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

**Phase:** Research & Planning

The project is currently in the research phase. We have completed:
- ✅ Technical feasibility analysis
- ✅ Hardware requirement validation
- ✅ Model and technology selection
- ✅ Architecture design
- ✅ Implementation roadmap

**Next Steps:**
- Set up development environment on DGX Spark
- Implement core LLM inference pipeline
- Integrate TTS and avatar systems
- Build memory and persistence layer

See [research/reports/technical_feasibility_analysis.md](research/reports/technical_feasibility_analysis.md) for detailed findings.

## Key Technical Decisions

### Architecture Approach

**LoRA-Based Dual AI System** (Recommended)
- Single Llama 3.1 70B base model (~35GB)
- Two LoRA personality adapters (~3GB total)
- Shared memory with personality-tagged entries
- <20ms personality switching latency

**Alternative:** Dual Qwen2.5-32B models (safer, 32GB total)

### Technology Stack

| Component | Technology | Memory | Latency |
|-----------|-----------|--------|---------|
| LLM | Llama 3.1 70B (4-bit) | 35-40GB | 200-400ms |
| TTS | StyleTTS2 / Fish Speech | 2-4GB | 80-150ms |
| STT | Faster-Whisper (Large-v3) | 3-5GB | 200-400ms |
| Memory | Mem0 + ChromaDB | 5-10GB | <50ms |
| Avatar | VTube Studio + Live2D | ~2GB | 16-33ms |
| Vision | YOLO-World / Florence-2 | 4-8GB | <50ms |

### Game Integration Strategy

- **Turn-based games:** Direct LLM function calling
- **Minecraft:** Hierarchical control (LLM + Voyager + Baritone)
- **Rhythm games:** Specialized neural network (<50ms latency)

## Research Findings

### Memory Allocation (128GB Total)
- LLM: 35-40GB
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

## Getting Started

### Prerequisites
- NVIDIA DGX Spark with GB10 Grace Blackwell
- Ubuntu 22.04 or later
- CUDA 12.3+
- Python 3.10+
- 2TB+ NVMe storage

### Installation

```bash
# Repository setup
git clone https://github.com/jhacksman/spark-vtuber.git
cd spark-vtuber

# (Future) Install dependencies
# pip install -r requirements.txt

# (Future) Download models
# ./scripts/download_models.sh
```

> **Note:** Implementation is in progress. Installation instructions will be added as components are developed.

## Documentation

- **[Research Overview](research/README.md)** - Technical research and feasibility studies
- **[Technical Feasibility Analysis](research/reports/technical_feasibility_analysis.md)** - Comprehensive technical analysis
- **[Implementation Roadmap](research/reports/technical_feasibility_analysis.md#8-implementation-roadmap)** - Development phases and timeline

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

- NVIDIA for DGX Spark hardware and TensorRT-LLM
- Meta for Llama 3.1
- Open-source communities behind StyleTTS2, Faster-Whisper, ChromaDB, and Mem0
- VTube Studio and Live2D for avatar technologies

---

**Project Status:** Research Phase
**Last Updated:** January 10, 2026
**Hardware:** NVIDIA DGX Spark (GB10)
