# Claude Code Audit Prompt: Spark VTuber Implementation

## Context

You are auditing a complete AI VTuber streaming system implementation for NVIDIA DGX Spark (128GB unified memory). The implementation was created by Devin and needs thorough review before production use.

**Repository**: https://github.com/jhacksman/spark-vtuber
**PR**: https://github.com/jhacksman/spark-vtuber/pull/3
**Branch**: `devin/1768009836-full-implementation`

## Implementation Overview

The system consists of 10 integrated components:

1. **LLM Engine** (`src/spark_vtuber/llm/`) - Llama 3.1 70B with vLLM/transformers, LoRA switching
2. **TTS Pipeline** (`src/spark_vtuber/tts/`) - Coqui TTS with streaming synthesis
3. **STT Pipeline** (`src/spark_vtuber/stt/`) - Faster-Whisper with VAD
4. **Memory System** (`src/spark_vtuber/memory/`) - ChromaDB with semantic search
5. **Avatar Control** (`src/spark_vtuber/avatar/`) - VTube Studio API, lip sync
6. **Personality System** (`src/spark_vtuber/personality/`) - Dual AI with LoRA switching
7. **Chat Integration** (`src/spark_vtuber/chat/`) - Twitch IRC client
8. **Game Integration** (`src/spark_vtuber/game/`) - Minecraft + Watch Mode
9. **Main Pipeline** (`src/spark_vtuber/pipeline.py`) - Orchestration
10. **CLI** (`src/spark_vtuber/main.py`) - Typer-based interface

## Audit Checklist

### 1. Architecture Review

- [ ] Review overall module structure and separation of concerns
- [ ] Verify dependency injection patterns are used correctly
- [ ] Check for circular imports between modules
- [ ] Assess if the pipeline architecture supports the <500ms latency target
- [ ] Verify streaming patterns are implemented correctly throughout
- [ ] Review the builder pattern in `PipelineBuilder`

### 2. Memory Budget Verification (Critical - 128GB Limit)

The DGX Spark has 128GB unified LPDDR5x memory. Verify the implementation respects this constraint:

**Expected Memory Allocation:**
- LLM (Llama 3.1 70B 4-bit AWQ): ~38GB
- TTS (Coqui): ~2GB
- STT (Whisper large-v3): ~3GB
- Memory/RAG (ChromaDB + embeddings): ~5GB
- Avatar/Vision: ~2GB
- System overhead: ~10GB
- **Total Target: <100GB** (leaving headroom)

**Audit Tasks:**
- [ ] Review `gpu_memory_utilization` settings in LLM config
- [ ] Verify quantization is correctly applied (4-bit)
- [ ] Check if models can be loaded/unloaded dynamically
- [ ] Verify `get_memory_usage()` methods are accurate
- [ ] Look for memory leaks in async operations
- [ ] Check if ChromaDB persistence is configured correctly

### 3. Latency Analysis (<500ms Target)

**Target Breakdown:**
- LLM first token: <200ms
- TTS first audio chunk: <100ms
- Avatar update: <50ms
- Total pipeline: <500ms

**Audit Tasks:**
- [ ] Review streaming implementation in `LlamaLLM.generate_stream()`
- [ ] Verify sentence-based TTS streaming in `StreamingTTS`
- [ ] Check if lip sync processing adds latency
- [ ] Review async patterns for parallelization opportunities
- [ ] Verify no blocking calls in the main pipeline loop
- [ ] Check WebSocket handling in VTube Studio client

### 4. API Correctness

**vLLM Integration (`llm/llama.py`):**
- [ ] Verify `AsyncLLMEngine` usage is correct
- [ ] Check LoRA request handling
- [ ] Verify sampling parameters are passed correctly
- [ ] Review fallback to transformers backend

**VTube Studio API (`avatar/vtube_studio.py`):**
- [ ] Verify WebSocket protocol implementation
- [ ] Check authentication flow
- [ ] Verify parameter injection format
- [ ] Review hotkey triggering

**Twitch IRC (`chat/twitch.py`):**
- [ ] Verify IRC protocol compliance
- [ ] Check capability requests (tags, commands, membership)
- [ ] Verify message parsing handles all edge cases
- [ ] Review SSL connection handling

**ChromaDB (`memory/chroma.py`):**
- [ ] Verify collection creation and persistence
- [ ] Check embedding generation
- [ ] Review query filtering logic
- [ ] Verify metadata handling

### 5. Error Handling & Edge Cases

- [ ] Review exception handling in all async methods
- [ ] Check for proper cleanup in `__aexit__` methods
- [ ] Verify reconnection logic for WebSocket connections
- [ ] Review timeout handling
- [ ] Check for race conditions in concurrent operations
- [ ] Verify graceful degradation when components fail

### 6. Async/Await Patterns

- [ ] Verify all I/O operations are async
- [ ] Check for blocking calls wrapped in `run_in_executor`
- [ ] Review task cancellation handling
- [ ] Verify proper use of `asyncio.Queue`
- [ ] Check for potential deadlocks
- [ ] Review signal handling in main.py

### 7. Type Safety

- [ ] Review type hints for completeness
- [ ] Check for `Any` usage that should be more specific
- [ ] Verify dataclass field types
- [ ] Review Optional vs required fields
- [ ] Check generic type usage (AsyncIterator, etc.)

### 8. Test Coverage Analysis

**Current Tests (`tests/unit/`):**
- test_config.py - Configuration settings
- test_context.py - Conversation context
- test_personality.py - Personality system
- test_chat.py - Chat message handling
- test_memory.py - Memory entries
- test_avatar.py - Avatar/lip sync
- test_game.py - Game state/actions

**Audit Tasks:**
- [ ] Identify untested code paths
- [ ] Review mock usage in tests
- [ ] Check for missing edge case tests
- [ ] Verify async test patterns
- [ ] Identify integration test gaps
- [ ] Review test fixtures in conftest.py

### 9. Security Considerations

- [ ] Review OAuth token handling for Twitch
- [ ] Check for credential exposure in logs
- [ ] Verify no hardcoded secrets
- [ ] Review input sanitization for chat messages
- [ ] Check WebSocket security
- [ ] Review file path handling

### 10. Performance Optimizations

- [ ] Identify opportunities for caching
- [ ] Review batch processing opportunities
- [ ] Check for unnecessary object creation in hot paths
- [ ] Review numpy array handling efficiency
- [ ] Identify potential GPU memory fragmentation issues
- [ ] Review ChromaDB query optimization

### 11. Integration Points

- [ ] Review data flow between LLM → TTS → Avatar
- [ ] Check memory retrieval integration with LLM context
- [ ] Verify personality switching affects all components
- [ ] Review chat → pipeline → response flow
- [ ] Check game state integration with LLM prompts

### 12. Configuration & Deployment

- [ ] Review pyproject.toml dependencies
- [ ] Check for version pinning issues
- [ ] Verify environment variable handling
- [ ] Review CLI argument parsing
- [ ] Check for missing configuration options

## Specific Code Review Requests

### High Priority Files

1. **`src/spark_vtuber/pipeline.py`** - Main orchestration logic
   - Review `process_message()` flow
   - Check streaming coordination
   - Verify error recovery

2. **`src/spark_vtuber/llm/llama.py`** - LLM implementation
   - Review vLLM engine initialization
   - Check LoRA adapter handling
   - Verify streaming token generation

3. **`src/spark_vtuber/personality/coordinator.py`** - Dual AI coordination
   - Review turn-taking logic
   - Check interjection probability handling
   - Verify personality switching

4. **`src/spark_vtuber/tts/streaming.py`** - Streaming TTS
   - Review sentence boundary detection
   - Check audio chunking logic
   - Verify buffer management

5. **`src/spark_vtuber/avatar/vtube_studio.py`** - Avatar control
   - Review WebSocket protocol
   - Check parameter injection
   - Verify authentication flow

### Questions to Answer

1. **Memory**: Will all components fit within 128GB when running simultaneously?
2. **Latency**: Can the pipeline achieve <500ms end-to-end latency?
3. **Stability**: Are there any patterns that could cause crashes during long streams?
4. **Scalability**: Can the system handle high chat volume (1000+ messages/minute)?
5. **Recovery**: How does the system recover from component failures?

## Output Format

Please provide your audit in the following format:

```markdown
## Audit Summary

### Critical Issues (Must Fix)
- Issue 1: [Description, File, Line, Severity]
- Issue 2: ...

### Major Issues (Should Fix)
- Issue 1: [Description, File, Line, Recommendation]
- Issue 2: ...

### Minor Issues (Nice to Fix)
- Issue 1: [Description, File, Line, Suggestion]
- Issue 2: ...

### Positive Observations
- What was done well

### Memory Budget Assessment
- Estimated total: X GB
- Feasibility: [Pass/Fail/Marginal]
- Recommendations: ...

### Latency Assessment
- Estimated pipeline latency: X ms
- Feasibility: [Pass/Fail/Marginal]
- Bottlenecks: ...

### Test Coverage Assessment
- Current coverage estimate: X%
- Critical gaps: ...
- Recommended additional tests: ...

### Recommendations
1. Priority 1: ...
2. Priority 2: ...
3. Priority 3: ...
```

## Reference Documents

- Original Design Document: `research/references/Design+Document_+AI+VTuber+Streamer+Neuro-sama+Clone.pdf`
- Technical Feasibility Analysis: `research/reports/technical_feasibility_analysis.md`
- Implementation Strategy: `docs/IMPLEMENTATION_STRATEGY.md`

## Commands to Run

```bash
# Clone and checkout the PR branch
git clone https://github.com/jhacksman/spark-vtuber.git
cd spark-vtuber
git checkout devin/1768009836-full-implementation

# View the implementation
find src -name "*.py" | head -20
wc -l src/spark_vtuber/**/*.py

# Review specific files
cat src/spark_vtuber/pipeline.py
cat src/spark_vtuber/llm/llama.py
cat src/spark_vtuber/personality/coordinator.py
```

---

**Note**: This is a production-critical system that will run for extended periods (8+ hour streams). Please be thorough in identifying any issues that could cause instability, memory leaks, or performance degradation over time.
