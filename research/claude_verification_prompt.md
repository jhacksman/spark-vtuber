# Claude Research Verification Prompt

Use this prompt with Claude to verify and expand upon the AI VTuber technology stack research.

---

## PROMPT

```
I'm building an AI VTuber streaming system similar to Neuro-sama that will run on a single NVIDIA DGX Spark (128GB unified memory, GB10 Grace Blackwell superchip). I need you to help verify the technical feasibility and identify any gaps in my proposed architecture.

## System Components

1. **Core LLM**: 70B parameter model (4-bit quantized) for conversational AI
2. **TTS**: Neural text-to-speech with streaming output (<200ms latency target)
3. **STT**: Whisper-based speech recognition for collaborator audio
4. **Avatar**: Live2D model with audio-driven lip sync via VTube Studio API
5. **Game Integration**: Hierarchical control (LLM for high-level decisions, specialized agents for low-level actions)
6. **Memory**: RAG-based long-term memory with vector store
7. **Dual AI**: Support for two AI personalities (main + "evil twin")

## Hardware Constraints

- 128GB unified LPDDR5x memory (shared between CPU and GPU)
- 273 GB/s memory bandwidth
- NVLink-C2C for cache-coherent CPU-GPU access
- 1 petaFLOP FP4 AI performance
- Single machine deployment (no cloud dependencies)

## Questions for Verification

1. **Memory Feasibility**: Is my proposed memory allocation realistic?
   - LLM: 80-100GB
   - TTS: 5-10GB
   - STT: 2-5GB
   - Vision: 5-10GB
   - Memory/RAG: 5-10GB
   - System: 10-15GB

2. **Latency Analysis**: Can we achieve sub-second response times with this pipeline?
   - Chat input -> LLM inference -> TTS streaming -> Avatar lip sync

3. **Game Playing**: What are the best approaches for:
   - Turn-based games (text API)
   - Real-time games like Minecraft (hierarchical control)
   - Rhythm games (specialized neural networks)

4. **Dual AI Coordination**: How should two AI personalities share resources and coordinate dialogue?

5. **Known Limitations**: What are the biggest technical risks or limitations I should be aware of?

6. **Alternative Approaches**: Are there better architectures or models I should consider for any component?

7. **Recent Research**: What recent papers (2024-2025) should I review for:
   - Low-latency streaming TTS
   - Real-time avatar animation
   - LLM-based game agents
   - Long-term conversational memory

Please provide detailed technical analysis with specific model recommendations, memory estimates, and latency projections where possible.
```

---

## FOLLOW-UP QUESTIONS

After the initial response, consider asking these follow-up questions:

### For LLM Selection
```
What specific open-source LLMs would you recommend for this use case? Consider:
- Conversational ability and personality consistency
- Inference speed on Blackwell architecture
- 4-bit quantization quality
- Context window requirements for streaming chat
```

### For TTS Optimization
```
Compare these TTS options for real-time streaming on GB10:
- Coqui TTS / XTTS
- StyleTTS2
- VITS/VITS2
- Bark
- Commercial APIs as fallback

What's the best balance of quality, latency, and memory usage?
```

### For Game Integration
```
For Minecraft specifically, analyze:
- Baritone mod for pathfinding
- MineRL/MineDojo for RL-based control
- Voyager-style LLM planning
- Vision-based approaches vs API-based

What hybrid approach would work best for entertaining gameplay?
```

### For Memory Systems
```
Compare memory architectures for long-term conversational AI:
- Mem0
- MemGPT
- LangChain memory modules
- Custom RAG with ChromaDB/Pinecone

What's optimal for maintaining personality consistency across multi-hour streams?
```

### For Avatar Control
```
Beyond VTube Studio, what options exist for:
- More expressive facial animations
- Body gesture control
- 3D avatar support
- AI-driven expression selection based on dialogue sentiment
```

---

## EXPECTED OUTPUTS

Claude should provide:

1. **Validation or corrections** to memory estimates
2. **Specific model recommendations** with version numbers
3. **Latency projections** for each pipeline stage
4. **Risk assessment** with mitigation strategies
5. **Architecture diagrams** or pseudocode where helpful
6. **Paper citations** for cutting-edge techniques
7. **Implementation priorities** and suggested development order

---

## NOTES FOR INTERPRETATION

- If Claude suggests cloud APIs, note that we prefer local-only solutions
- If memory estimates exceed 128GB, we need to reconsider model sizes
- Latency targets: <500ms for chat response, <100ms for avatar sync
- The system should support 8+ hour continuous streaming
