"""
Performance metrics and instrumentation for Spark VTuber.

Provides latency tracking, memory monitoring, and benchmark utilities
for validating performance on DGX Spark hardware.
"""

import asyncio
import csv
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from spark_vtuber.utils.logging import LoggerMixin


@dataclass
class PipelineMetrics:
    """Metrics for a single pipeline execution."""

    llm_first_token_ms: float = 0.0
    llm_total_ms: float = 0.0
    llm_tokens_generated: int = 0
    llm_tokens_per_sec: float = 0.0

    tts_first_chunk_ms: float = 0.0
    tts_total_ms: float = 0.0

    avatar_sync_ms: float = 0.0
    memory_retrieval_ms: float = 0.0

    total_pipeline_ms: float = 0.0

    gpu_memory_gb: float = 0.0
    cpu_memory_gb: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return asdict(self)

    def passes_latency_target(self, target_ms: float = 500.0) -> bool:
        """Check if total pipeline latency is under target."""
        return self.total_pipeline_ms < target_ms


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: float
    gpu_memory_gb: float
    gpu_memory_percent: float
    cpu_memory_gb: float
    cpu_memory_percent: float

    def to_dict(self) -> dict:
        """Convert snapshot to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    total_messages: int
    successful_messages: int
    failed_messages: int

    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float

    llm_tokens_per_sec_mean: float
    tts_first_chunk_mean_ms: float

    gpu_memory_peak_gb: float
    cpu_memory_peak_gb: float

    passes_target: bool
    target_ms: float

    duration_seconds: float
    timestamp: str

    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Write results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_gpu_memory() -> tuple[float, float]:
    """
    Get current GPU memory usage.

    Returns:
        Tuple of (memory_gb, memory_percent)
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            percent = (allocated / total) * 100 if total > 0 else 0
            return allocated, percent
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            used, total = map(float, result.stdout.strip().split(","))
            used_gb = used / 1024
            total_gb = total / 1024
            percent = (used / total) * 100 if total > 0 else 0
            return used_gb, percent
    except Exception:
        pass

    return 0.0, 0.0


def get_cpu_memory() -> tuple[float, float]:
    """
    Get current CPU memory usage.

    Returns:
        Tuple of (memory_gb, memory_percent)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        percent = mem.percent
        return used_gb, percent
    except ImportError:
        pass

    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
            total = int(lines[0].split()[1]) / (1024**2)
            available = int(lines[2].split()[1]) / (1024**2)
            used = total - available
            percent = (used / total) * 100 if total > 0 else 0
            return used, percent
    except Exception:
        pass

    return 0.0, 0.0


class MetricsCollector(LoggerMixin):
    """
    Collects and logs pipeline metrics.

    Writes metrics to CSV files and provides summary statistics.
    """

    def __init__(
        self,
        log_dir: Path = Path("logs"),
        metrics_file: str = "metrics.csv",
        memory_file: str = "memory.csv",
        summary_interval: int = 10,
        memory_alert_gb: float = 115.0,
    ):
        """
        Initialize metrics collector.

        Args:
            log_dir: Directory for log files
            metrics_file: Filename for metrics CSV
            memory_file: Filename for memory CSV
            summary_interval: Print summary every N messages
            memory_alert_gb: Alert threshold for GPU memory (GB)
        """
        self.log_dir = log_dir
        self.metrics_path = log_dir / metrics_file
        self.memory_path = log_dir / memory_file
        self.summary_interval = summary_interval
        self.memory_alert_gb = memory_alert_gb

        self._metrics: list[PipelineMetrics] = []
        self._memory_snapshots: list[MemorySnapshot] = []
        self._message_count = 0

        self._memory_monitor_task: asyncio.Task | None = None
        self._memory_monitor_running = False

        self._ensure_log_dir()
        self._init_csv_files()

    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "llm_first_token_ms",
                    "llm_total_ms",
                    "llm_tokens_generated",
                    "llm_tokens_per_sec",
                    "tts_first_chunk_ms",
                    "tts_total_ms",
                    "avatar_sync_ms",
                    "memory_retrieval_ms",
                    "total_pipeline_ms",
                    "gpu_memory_gb",
                    "cpu_memory_gb",
                ])

        if not self.memory_path.exists():
            with open(self.memory_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "gpu_memory_gb",
                    "gpu_memory_percent",
                    "cpu_memory_gb",
                    "cpu_memory_percent",
                ])

    def record_metrics(self, metrics: PipelineMetrics) -> None:
        """
        Record pipeline metrics.

        Args:
            metrics: Metrics to record
        """
        gpu_mem, _ = get_gpu_memory()
        cpu_mem, _ = get_cpu_memory()
        metrics.gpu_memory_gb = gpu_mem
        metrics.cpu_memory_gb = cpu_mem

        self._metrics.append(metrics)
        self._message_count += 1

        self._write_metrics_csv(metrics)

        if self._message_count % self.summary_interval == 0:
            self._print_summary()

    def _write_metrics_csv(self, metrics: PipelineMetrics) -> None:
        """Write metrics to CSV file."""
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(metrics.timestamp).isoformat(),
                f"{metrics.llm_first_token_ms:.2f}",
                f"{metrics.llm_total_ms:.2f}",
                metrics.llm_tokens_generated,
                f"{metrics.llm_tokens_per_sec:.2f}",
                f"{metrics.tts_first_chunk_ms:.2f}",
                f"{metrics.tts_total_ms:.2f}",
                f"{metrics.avatar_sync_ms:.2f}",
                f"{metrics.memory_retrieval_ms:.2f}",
                f"{metrics.total_pipeline_ms:.2f}",
                f"{metrics.gpu_memory_gb:.2f}",
                f"{metrics.cpu_memory_gb:.2f}",
            ])

    def _print_summary(self) -> None:
        """Print summary of recent metrics."""
        if not self._metrics:
            return

        recent = self._metrics[-self.summary_interval:]

        avg_pipeline = sum(m.total_pipeline_ms for m in recent) / len(recent)
        avg_llm_first = sum(m.llm_first_token_ms for m in recent) / len(recent)
        avg_tts_first = sum(m.tts_first_chunk_ms for m in recent) / len(recent)
        avg_tokens_sec = sum(m.llm_tokens_per_sec for m in recent) / len(recent)

        passes = sum(1 for m in recent if m.passes_latency_target())
        pass_rate = (passes / len(recent)) * 100

        self.logger.info(
            f"[Metrics Summary - Last {len(recent)} messages]\n"
            f"  Pipeline: {avg_pipeline:.0f}ms avg | "
            f"LLM first token: {avg_llm_first:.0f}ms | "
            f"TTS first chunk: {avg_tts_first:.0f}ms\n"
            f"  Tokens/sec: {avg_tokens_sec:.1f} | "
            f"<500ms target: {pass_rate:.0f}% pass rate"
        )

    async def start_memory_monitor(self, interval_seconds: float = 5.0) -> None:
        """
        Start background memory monitoring.

        Args:
            interval_seconds: Interval between memory checks
        """
        if self._memory_monitor_running:
            return

        self._memory_monitor_running = True
        self._memory_monitor_task = asyncio.create_task(
            self._memory_monitor_loop(interval_seconds)
        )
        self.logger.info(f"Memory monitor started (interval: {interval_seconds}s)")

    async def stop_memory_monitor(self) -> None:
        """Stop background memory monitoring."""
        self._memory_monitor_running = False
        if self._memory_monitor_task:
            self._memory_monitor_task.cancel()
            try:
                await self._memory_monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Memory monitor stopped")

    async def _memory_monitor_loop(self, interval: float) -> None:
        """Background loop for memory monitoring."""
        while self._memory_monitor_running:
            try:
                snapshot = self._take_memory_snapshot()
                self._memory_snapshots.append(snapshot)
                self._write_memory_csv(snapshot)

                if snapshot.gpu_memory_gb > self.memory_alert_gb:
                    self.logger.warning(
                        f"HIGH MEMORY ALERT: GPU using {snapshot.gpu_memory_gb:.1f}GB "
                        f"(>{self.memory_alert_gb}GB threshold, "
                        f"{snapshot.gpu_memory_percent:.1f}% of total)"
                    )

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(interval)

    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        gpu_mem, gpu_pct = get_gpu_memory()
        cpu_mem, cpu_pct = get_cpu_memory()

        return MemorySnapshot(
            timestamp=time.time(),
            gpu_memory_gb=gpu_mem,
            gpu_memory_percent=gpu_pct,
            cpu_memory_gb=cpu_mem,
            cpu_memory_percent=cpu_pct,
        )

    def _write_memory_csv(self, snapshot: MemorySnapshot) -> None:
        """Write memory snapshot to CSV file."""
        with open(self.memory_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(snapshot.timestamp).isoformat(),
                f"{snapshot.gpu_memory_gb:.2f}",
                f"{snapshot.gpu_memory_percent:.1f}",
                f"{snapshot.cpu_memory_gb:.2f}",
                f"{snapshot.cpu_memory_percent:.1f}",
            ])

    def get_benchmark_results(self, target_ms: float = 500.0) -> BenchmarkResults | None:
        """
        Calculate benchmark results from collected metrics.

        Args:
            target_ms: Latency target in milliseconds

        Returns:
            BenchmarkResults or None if no metrics collected
        """
        if not self._metrics:
            return None

        latencies = [m.total_pipeline_ms for m in self._metrics]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return latencies_sorted[min(idx, n - 1)]

        successful = sum(1 for m in self._metrics if m.total_pipeline_ms > 0)
        failed = len(self._metrics) - successful

        tokens_per_sec = [m.llm_tokens_per_sec for m in self._metrics if m.llm_tokens_per_sec > 0]
        tts_first = [m.tts_first_chunk_ms for m in self._metrics if m.tts_first_chunk_ms > 0]

        gpu_peak = max((m.gpu_memory_gb for m in self._metrics), default=0.0)
        cpu_peak = max((m.cpu_memory_gb for m in self._metrics), default=0.0)

        if self._memory_snapshots:
            gpu_peak = max(gpu_peak, max(s.gpu_memory_gb for s in self._memory_snapshots))
            cpu_peak = max(cpu_peak, max(s.cpu_memory_gb for s in self._memory_snapshots))

        passes = sum(1 for m in self._metrics if m.passes_latency_target(target_ms))
        passes_target = passes == len(self._metrics)

        first_ts = self._metrics[0].timestamp
        last_ts = self._metrics[-1].timestamp

        return BenchmarkResults(
            total_messages=len(self._metrics),
            successful_messages=successful,
            failed_messages=failed,
            latency_p50_ms=percentile(50),
            latency_p95_ms=percentile(95),
            latency_p99_ms=percentile(99),
            latency_mean_ms=sum(latencies) / n,
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            llm_tokens_per_sec_mean=sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0.0,
            tts_first_chunk_mean_ms=sum(tts_first) / len(tts_first) if tts_first else 0.0,
            gpu_memory_peak_gb=gpu_peak,
            cpu_memory_peak_gb=cpu_peak,
            passes_target=passes_target,
            target_ms=target_ms,
            duration_seconds=last_ts - first_ts,
            timestamp=datetime.now().isoformat(),
        )

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._memory_snapshots.clear()
        self._message_count = 0


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

    def mark(self) -> float:
        """Mark current elapsed time without stopping timer."""
        return (time.perf_counter() - self.start_time) * 1000


class StreamingTimer:
    """Timer for streaming operations that tracks first chunk and total time."""

    def __init__(self):
        self.start_time: float = 0.0
        self.first_chunk_time: float | None = None
        self.end_time: float = 0.0
        self.chunk_count: int = 0

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.first_chunk_time = None
        self.chunk_count = 0

    def mark_chunk(self) -> None:
        """Mark a chunk received."""
        if self.first_chunk_time is None:
            self.first_chunk_time = time.perf_counter()
        self.chunk_count += 1

    def stop(self) -> None:
        """Stop the timer."""
        self.end_time = time.perf_counter()

    @property
    def first_chunk_ms(self) -> float:
        """Time to first chunk in milliseconds."""
        if self.first_chunk_time is None:
            return 0.0
        return (self.first_chunk_time - self.start_time) * 1000

    @property
    def total_ms(self) -> float:
        """Total time in milliseconds."""
        return (self.end_time - self.start_time) * 1000
