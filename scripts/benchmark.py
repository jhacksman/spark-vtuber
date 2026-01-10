#!/usr/bin/env python3
"""
Benchmark script for Spark VTuber pipeline.

Runs test messages through the pipeline and outputs performance metrics.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spark_vtuber.metrics import (
    BenchmarkResults,
    MetricsCollector,
    PipelineMetrics,
    get_gpu_memory,
    get_cpu_memory,
)


TEST_MESSAGES = [
    "Hello! How are you doing today?",
    "What's your favorite video game?",
    "Tell me a joke!",
    "What do you think about the weather?",
    "Can you sing a song?",
    "What's the meaning of life?",
    "Do you have any hobbies?",
    "What's your opinion on cats vs dogs?",
    "Tell me something interesting!",
    "What would you do with a million dollars?",
    "What's your favorite food?",
    "Do you believe in aliens?",
    "What's the best movie you've seen?",
    "Can you tell me a story?",
    "What makes you happy?",
    "What's your biggest fear?",
    "If you could travel anywhere, where would you go?",
    "What's your favorite color and why?",
    "Do you have any advice for someone feeling down?",
    "What's the most important thing in life?",
]


class MockLLM:
    """Mock LLM for benchmarking without actual model."""

    def __init__(self, tokens_per_response: int = 50, delay_ms: float = 5.0):
        self.tokens_per_response = tokens_per_response
        self.delay_ms = delay_ms
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate_stream(self, prompt: str, **kwargs):
        for i in range(self.tokens_per_response):
            await asyncio.sleep(self.delay_ms / 1000)
            yield f"token{i} "

    def get_memory_usage(self) -> dict:
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}


class MockTTS:
    """Mock TTS for benchmarking without actual model."""

    def __init__(self, chunks_per_sentence: int = 5, delay_ms: float = 10.0):
        self.chunks_per_sentence = chunks_per_sentence
        self.delay_ms = delay_ms
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def synthesize_stream(self, text: str):
        import numpy as np
        for _ in range(self.chunks_per_sentence):
            await asyncio.sleep(self.delay_ms / 1000)
            yield np.zeros(1024, dtype=np.float32)

    def get_memory_usage(self) -> dict:
        return {"allocated_gb": 0.0}


class MockMemory:
    """Mock memory for benchmarking."""

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def search(self, query: str, top_k: int = 5):
        await asyncio.sleep(0.005)
        return []

    async def add(self, content: str, category: str, personality: str) -> None:
        pass


def print_results(results: BenchmarkResults) -> None:
    """Print benchmark results to console."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nMessages: {results.total_messages} total, "
          f"{results.successful_messages} successful, "
          f"{results.failed_messages} failed")

    print(f"\nLatency (ms):")
    print(f"  p50:  {results.latency_p50_ms:>8.1f}")
    print(f"  p95:  {results.latency_p95_ms:>8.1f}")
    print(f"  p99:  {results.latency_p99_ms:>8.1f}")
    print(f"  mean: {results.latency_mean_ms:>8.1f}")
    print(f"  min:  {results.latency_min_ms:>8.1f}")
    print(f"  max:  {results.latency_max_ms:>8.1f}")

    print(f"\nThroughput:")
    print(f"  LLM tokens/sec: {results.llm_tokens_per_sec_mean:.1f}")
    print(f"  TTS first chunk: {results.tts_first_chunk_mean_ms:.1f}ms")

    print(f"\nMemory (peak):")
    print(f"  GPU: {results.gpu_memory_peak_gb:.1f} GB")
    print(f"  CPU: {results.cpu_memory_peak_gb:.1f} GB")

    print(f"\nDuration: {results.duration_seconds:.1f}s")

    status = "PASS" if results.passes_target else "FAIL"
    color = "\033[92m" if results.passes_target else "\033[91m"
    reset = "\033[0m"

    print(f"\n{color}Target (<{results.target_ms}ms): {status}{reset}")
    print("=" * 60)


async def run_benchmark(
    num_messages: int = 20,
    target_ms: float = 500.0,
    output_path: Path | None = None,
    use_mock: bool = True,
    llm_delay_ms: float = 5.0,
    tts_delay_ms: float = 10.0,
) -> BenchmarkResults:
    """
    Run benchmark with specified parameters.

    Args:
        num_messages: Number of test messages to run
        target_ms: Latency target in milliseconds
        output_path: Optional path to write JSON results
        use_mock: Use mock components (True) or real pipeline (False)
        llm_delay_ms: Simulated LLM delay per token (mock only)
        tts_delay_ms: Simulated TTS delay per chunk (mock only)

    Returns:
        BenchmarkResults with all metrics
    """
    print(f"Running benchmark with {num_messages} messages...")
    print(f"Target latency: <{target_ms}ms")

    if use_mock:
        print("Using mock components (no GPU required)")
    else:
        print("Using real pipeline components")

    collector = MetricsCollector(
        log_dir=Path("logs"),
        summary_interval=num_messages + 1,
    )

    await collector.start_memory_monitor(interval_seconds=1.0)

    messages = (TEST_MESSAGES * ((num_messages // len(TEST_MESSAGES)) + 1))[:num_messages]

    if use_mock:
        llm = MockLLM(tokens_per_response=50, delay_ms=llm_delay_ms)
        tts = MockTTS(chunks_per_sentence=5, delay_ms=tts_delay_ms)
        memory = MockMemory()

        await llm.load()
        await tts.load()
        await memory.initialize()

        for i, msg in enumerate(messages, 1):
            print(f"\rProcessing message {i}/{num_messages}...", end="", flush=True)

            metrics = PipelineMetrics()
            pipeline_start = time.perf_counter()

            mem_start = time.perf_counter()
            await memory.search(msg)
            metrics.memory_retrieval_ms = (time.perf_counter() - mem_start) * 1000

            llm_start = time.perf_counter()
            llm_first_token = None
            token_count = 0

            async for token in llm.generate_stream(msg):
                if llm_first_token is None:
                    llm_first_token = time.perf_counter()
                token_count += 1

            llm_end = time.perf_counter()

            tts_start = time.perf_counter()
            tts_first_chunk = None

            async for chunk in tts.synthesize_stream("response"):
                if tts_first_chunk is None:
                    tts_first_chunk = time.perf_counter()

            tts_end = time.perf_counter()

            pipeline_end = time.perf_counter()

            metrics.llm_first_token_ms = (
                (llm_first_token - llm_start) * 1000 if llm_first_token else 0
            )
            metrics.llm_total_ms = (llm_end - llm_start) * 1000
            metrics.llm_tokens_generated = token_count
            metrics.llm_tokens_per_sec = (
                token_count / (metrics.llm_total_ms / 1000) if metrics.llm_total_ms > 0 else 0
            )
            metrics.tts_first_chunk_ms = (
                (tts_first_chunk - tts_start) * 1000 if tts_first_chunk else 0
            )
            metrics.tts_total_ms = (tts_end - tts_start) * 1000
            metrics.total_pipeline_ms = (pipeline_end - pipeline_start) * 1000

            collector.record_metrics(metrics)

        await llm.unload()
        await tts.unload()
        await memory.close()

    else:
        from spark_vtuber.config.settings import Settings
        from spark_vtuber.llm.llama import LlamaLLM
        from spark_vtuber.tts.coqui import CoquiTTS
        from spark_vtuber.memory.chroma import ChromaMemory
        from spark_vtuber.pipeline import StreamingPipeline
        from spark_vtuber.chat.base import ChatMessage

        settings = Settings()
        llm = LlamaLLM(settings.llm)
        tts = CoquiTTS(settings.tts)
        memory = ChromaMemory(settings.memory)

        pipeline = StreamingPipeline(
            llm=llm,
            tts=tts,
            memory=memory,
            settings=settings,
            enable_metrics=True,
        )

        await pipeline.initialize()

        for i, msg in enumerate(messages, 1):
            print(f"\rProcessing message {i}/{num_messages}...", end="", flush=True)

            chat_msg = ChatMessage(
                username="benchmark_user",
                content=msg,
                timestamp=time.time(),
            )
            await pipeline.process_message(chat_msg)

        await pipeline.shutdown()

        if pipeline.metrics_collector:
            collector = pipeline.metrics_collector

    print("\n")

    await collector.stop_memory_monitor()

    results = collector.get_benchmark_results(target_ms=target_ms)

    if results:
        print_results(results)

        if output_path:
            results.to_json(output_path)
            print(f"\nResults written to: {output_path}")

        return results

    print("No results collected!")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Spark VTuber pipeline performance"
    )
    parser.add_argument(
        "--messages", "-m",
        type=int,
        default=20,
        help="Number of test messages to run (default: 20)"
    )
    parser.add_argument(
        "--target", "-t",
        type=float,
        default=500.0,
        help="Latency target in milliseconds (default: 500)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (optional)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real pipeline components (requires GPU and models)"
    )
    parser.add_argument(
        "--llm-delay",
        type=float,
        default=5.0,
        help="Simulated LLM delay per token in ms (mock only, default: 5)"
    )
    parser.add_argument(
        "--tts-delay",
        type=float,
        default=10.0,
        help="Simulated TTS delay per chunk in ms (mock only, default: 10)"
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    results = asyncio.run(run_benchmark(
        num_messages=args.messages,
        target_ms=args.target,
        output_path=output_path,
        use_mock=not args.real,
        llm_delay_ms=args.llm_delay,
        tts_delay_ms=args.tts_delay,
    ))

    if results and not results.passes_target:
        sys.exit(1)


if __name__ == "__main__":
    main()
