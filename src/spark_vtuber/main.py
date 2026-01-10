"""
Main entry point for Spark VTuber.

Provides CLI interface for running the AI VTuber streaming system.
"""

import asyncio
import signal
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spark_vtuber.config.settings import Settings, get_settings
from spark_vtuber.utils.logging import setup_logging, get_logger

app = typer.Typer(
    name="spark-vtuber",
    help="AI-powered VTuber streaming system for NVIDIA DGX Spark",
)
console = Console()
logger = get_logger(__name__)


@app.command()
def run(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
    no_chat: bool = typer.Option(
        False,
        "--no-chat",
        help="Disable chat integration",
    ),
    no_avatar: bool = typer.Option(
        False,
        "--no-avatar",
        help="Disable avatar control",
    ),
    no_game: bool = typer.Option(
        False,
        "--no-game",
        help="Disable game integration",
    ),
    dual_avatar: bool = typer.Option(
        False,
        "--dual-avatar",
        help="Enable dual avatar mode (requires VNet Multiplayer Collab)",
    ),
) -> None:
    """Run the Spark VTuber streaming system."""
    settings = get_settings()

    if debug:
        settings.debug = True
        settings.log_level = "DEBUG"

    if dual_avatar:
        settings.avatar.dual_avatar_enabled = True

    setup_logging(level=settings.log_level)

    console.print(Panel.fit(
        "[bold blue]Spark VTuber[/bold blue]\n"
        "AI-powered VTuber streaming system",
        border_style="blue",
    ))

    asyncio.run(_run_pipeline(settings, no_chat, no_avatar, no_game))


async def _run_pipeline(
    settings: Settings,
    no_chat: bool,
    no_avatar: bool,
    no_game: bool,
) -> None:
    """Run the streaming pipeline."""
    from spark_vtuber.llm.llama import QwenLLM
    from spark_vtuber.tts.fish_speech import FishSpeechTTS
    from spark_vtuber.tts.styletts2 import StyleTTS2
    from spark_vtuber.memory.chroma import ChromaMemory
    from spark_vtuber.avatar.vtube_studio import VTubeStudioAvatar
    from spark_vtuber.avatar.dual_vtube_studio import DualVTubeStudioAvatar
    from spark_vtuber.chat.twitch import TwitchChat
    from spark_vtuber.pipeline import PipelineBuilder

    console.print("[yellow]Initializing components...[/yellow]")

    llm = QwenLLM(
        model_name=settings.llm.model_name,
        quantization=settings.llm.quantization,
        gpu_memory_utilization=settings.llm.gpu_memory_utilization,
        max_model_len=settings.llm.context_length,
    )

    if settings.tts.engine == "fish_speech":
        tts = FishSpeechTTS(
            sample_rate=settings.tts.sample_rate,
            use_api=settings.tts.use_api,
            api_key=settings.tts.api_key,
            reference_id=settings.tts.voice_id,
            model=settings.tts.model_name,
        )
    else:
        tts = StyleTTS2(
            sample_rate=settings.tts.sample_rate,
        )

    memory = ChromaMemory(
        persist_dir=settings.memory.chroma_persist_dir,
        embedding_model=settings.memory.embedding_model,
    )

    builder = PipelineBuilder(settings)
    builder.with_llm(llm).with_tts(tts).with_memory(memory)

    if not no_avatar:
        if settings.avatar.dual_avatar_enabled:
            console.print("[cyan]Initializing dual avatar mode (VNet)[/cyan]")
            avatar = DualVTubeStudioAvatar(
                primary_host=settings.avatar.vtube_studio_host,
                primary_port=settings.avatar.primary_avatar_port,
                secondary_host=settings.avatar.vtube_studio_host,
                secondary_port=settings.avatar.secondary_avatar_port,
            )
        else:
            avatar = VTubeStudioAvatar(
                host=settings.avatar.vtube_studio_host,
                port=settings.avatar.vtube_studio_port,
                plugin_name=settings.avatar.plugin_name,
            )
        builder.with_avatar(avatar)

    if not no_chat and settings.chat.twitch_enabled:
        chat = TwitchChat(
            channel=settings.chat.twitch_channel,
            oauth_token=settings.chat.twitch_oauth_token,
        )
        builder.with_chat(chat)

    pipeline = builder.build()

    shutdown_event = asyncio.Event()

    def signal_handler():
        console.print("\n[yellow]Shutting down...[/yellow]")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        console.print("[green]Starting pipeline...[/green]")
        await pipeline.initialize()

        console.print("[bold green]Pipeline running![/bold green]")
        console.print("Press Ctrl+C to stop")

        pipeline_task = asyncio.create_task(pipeline.start())

        await shutdown_event.wait()

        await pipeline.stop()
        pipeline_task.cancel()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Pipeline error")

    finally:
        await pipeline.shutdown()
        console.print("[green]Shutdown complete[/green]")


@app.command()
def status() -> None:
    """Show system status and configuration."""
    settings = get_settings()

    table = Table(title="Spark VTuber Configuration")
    table.add_column("Component", style="cyan")
    table.add_column("Setting", style="green")
    table.add_column("Value", style="yellow")

    table.add_row("LLM", "Model", settings.llm.model_name)
    table.add_row("LLM", "Quantization", settings.llm.quantization)
    table.add_row("LLM", "Context Length", str(settings.llm.context_length))

    table.add_row("TTS", "Engine", settings.tts.engine)
    table.add_row("TTS", "Sample Rate", str(settings.tts.sample_rate))

    table.add_row("STT", "Model Size", settings.stt.model_size)
    table.add_row("STT", "Device", settings.stt.device)

    table.add_row("Memory", "Embedding Model", settings.memory.embedding_model)
    table.add_row("Memory", "Max Memories", str(settings.memory.max_memories))

    table.add_row("Avatar", "VTube Studio Host", settings.avatar.vtube_studio_host)
    table.add_row("Avatar", "VTube Studio Port", str(settings.avatar.vtube_studio_port))
    table.add_row("Avatar", "Dual Avatar Mode", str(settings.avatar.dual_avatar_enabled))
    if settings.avatar.dual_avatar_enabled:
        table.add_row("Avatar", "Primary Port", str(settings.avatar.primary_avatar_port))
        table.add_row("Avatar", "Secondary Port", str(settings.avatar.secondary_avatar_port))

    table.add_row("Personality", "Primary", settings.personality.primary_name)
    table.add_row("Personality", "Secondary", settings.personality.secondary_name)

    table.add_row("Chat", "Twitch Enabled", str(settings.chat.twitch_enabled))
    table.add_row("Chat", "YouTube Enabled", str(settings.chat.youtube_enabled))

    table.add_row("Game", "Minecraft Enabled", str(settings.game.minecraft_enabled))
    table.add_row("Game", "Watch Mode", str(settings.game.watch_mode_enabled))

    console.print(table)


@app.command()
def test_tts(
    text: str = typer.Argument(..., help="Text to synthesize"),
    output: Path = typer.Option(
        Path("output.wav"),
        "--output",
        "-o",
        help="Output file path",
    ),
) -> None:
    """Test TTS synthesis."""
    asyncio.run(_test_tts(text, output))


async def _test_tts(text: str, output: Path) -> None:
    """Run TTS test."""
    from spark_vtuber.tts.fish_speech import FishSpeechTTS
    from spark_vtuber.tts.styletts2 import StyleTTS2
    import soundfile as sf

    settings = get_settings()

    console.print(f"[yellow]Synthesizing: {text}[/yellow]")
    console.print(f"[cyan]Using TTS engine: {settings.tts.engine}[/cyan]")

    if settings.tts.engine == "fish_speech":
        tts = FishSpeechTTS(
            sample_rate=settings.tts.sample_rate,
            use_api=settings.tts.use_api,
            api_key=settings.tts.api_key,
            reference_id=settings.tts.voice_id,
            model=settings.tts.model_name,
        )
    else:
        tts = StyleTTS2(
            sample_rate=settings.tts.sample_rate,
        )

    await tts.load()

    result = await tts.synthesize(text)

    sf.write(str(output), result.audio, result.sample_rate)

    console.print(f"[green]Saved to {output}[/green]")
    console.print(f"Duration: {result.duration_seconds:.2f}s")
    console.print(f"Latency: {result.latency_ms:.0f}ms")

    await tts.unload()


@app.command()
def test_llm(
    prompt: str = typer.Argument(..., help="Prompt to generate from"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m"),
) -> None:
    """Test LLM generation."""
    asyncio.run(_test_llm(prompt, max_tokens))


async def _test_llm(prompt: str, max_tokens: int) -> None:
    """Run LLM test."""
    from spark_vtuber.llm.llama import QwenLLM

    settings = get_settings()

    console.print(f"[yellow]Generating response for: {prompt}[/yellow]")

    llm = QwenLLM(
        model_name=settings.llm.model_name,
        quantization=settings.llm.quantization,
    )

    await llm.load()

    console.print("[cyan]Response:[/cyan]")
    async for token in llm.generate_stream(prompt, max_tokens=max_tokens):
        console.print(token, end="")

    console.print()

    await llm.unload()


@app.command()
def benchmark(
    messages: int = typer.Option(
        20,
        "--messages",
        "-m",
        help="Number of test messages to run",
    ),
    target: float = typer.Option(
        500.0,
        "--target",
        "-t",
        help="Latency target in milliseconds",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file path",
    ),
    real: bool = typer.Option(
        False,
        "--real",
        help="Use real pipeline components (requires GPU and models)",
    ),
    llm_delay: float = typer.Option(
        5.0,
        "--llm-delay",
        help="Simulated LLM delay per token in ms (mock only)",
    ),
    tts_delay: float = typer.Option(
        10.0,
        "--tts-delay",
        help="Simulated TTS delay per chunk in ms (mock only)",
    ),
) -> None:
    """Run performance benchmark on the pipeline."""
    asyncio.run(_run_benchmark(messages, target, output, real, llm_delay, tts_delay))


async def _run_benchmark(
    messages: int,
    target: float,
    output: Path | None,
    real: bool,
    llm_delay: float,
    tts_delay: float,
) -> None:
    """Run benchmark."""
    from spark_vtuber.metrics import (
        MetricsCollector,
        PipelineMetrics,
    )
    import time

    console.print(Panel.fit(
        "[bold blue]Spark VTuber Benchmark[/bold blue]\n"
        f"Running {messages} test messages",
        border_style="blue",
    ))

    console.print(f"[yellow]Target latency: <{target}ms[/yellow]")

    if real:
        console.print("[cyan]Using real pipeline components[/cyan]")
    else:
        console.print("[cyan]Using mock components (no GPU required)[/cyan]")

    test_messages = [
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

    collector = MetricsCollector(
        log_dir=Path("logs"),
        summary_interval=messages + 1,
    )

    await collector.start_memory_monitor(interval_seconds=1.0)

    msgs = (test_messages * ((messages // len(test_messages)) + 1))[:messages]

    if not real:
        for i, msg in enumerate(msgs, 1):
            console.print(f"\rProcessing message {i}/{messages}...", end="")

            metrics = PipelineMetrics()
            pipeline_start = time.perf_counter()

            await asyncio.sleep(0.005)
            metrics.memory_retrieval_ms = 5.0

            llm_start = time.perf_counter()
            llm_first_token = None
            token_count = 50

            for _ in range(token_count):
                await asyncio.sleep(llm_delay / 1000)
                if llm_first_token is None:
                    llm_first_token = time.perf_counter()

            llm_end = time.perf_counter()

            tts_start = time.perf_counter()
            tts_first_chunk = None

            for _ in range(5):
                await asyncio.sleep(tts_delay / 1000)
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

    else:
        from spark_vtuber.llm.llama import QwenLLM
        from spark_vtuber.tts.coqui import CoquiTTS
        from spark_vtuber.memory.chroma import ChromaMemory
        from spark_vtuber.pipeline import StreamingPipeline
        from spark_vtuber.chat.base import ChatMessage

        settings = get_settings()
        llm = QwenLLM(
            model_name=settings.llm.model_name,
            quantization=settings.llm.quantization,
        )
        tts = CoquiTTS(
            model_name=settings.tts.model_name,
            sample_rate=settings.tts.sample_rate,
        )
        memory = ChromaMemory(
            persist_dir=settings.memory.chroma_persist_dir,
            embedding_model=settings.memory.embedding_model,
        )

        pipeline = StreamingPipeline(
            llm=llm,
            tts=tts,
            memory=memory,
            settings=settings,
            enable_metrics=True,
        )

        await pipeline.initialize()

        for i, msg in enumerate(msgs, 1):
            console.print(f"\rProcessing message {i}/{messages}...", end="")

            chat_msg = ChatMessage(
                username="benchmark_user",
                content=msg,
                timestamp=time.time(),
            )
            await pipeline.process_message(chat_msg)

        await pipeline.shutdown()

        if pipeline.metrics_collector:
            collector = pipeline.metrics_collector

    console.print("\n")

    await collector.stop_memory_monitor()

    results = collector.get_benchmark_results(target_ms=target)

    if results:
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Messages", f"{results.total_messages}")
        table.add_row("Latency p50", f"{results.latency_p50_ms:.1f}ms")
        table.add_row("Latency p95", f"{results.latency_p95_ms:.1f}ms")
        table.add_row("Latency p99", f"{results.latency_p99_ms:.1f}ms")
        table.add_row("Latency mean", f"{results.latency_mean_ms:.1f}ms")
        table.add_row("LLM tokens/sec", f"{results.llm_tokens_per_sec_mean:.1f}")
        table.add_row("TTS first chunk", f"{results.tts_first_chunk_mean_ms:.1f}ms")
        table.add_row("GPU memory peak", f"{results.gpu_memory_peak_gb:.1f} GB")
        table.add_row("CPU memory peak", f"{results.cpu_memory_peak_gb:.1f} GB")
        table.add_row("Duration", f"{results.duration_seconds:.1f}s")

        console.print(table)

        status = "[green]PASS[/green]" if results.passes_target else "[red]FAIL[/red]"
        console.print(f"\nTarget (<{target}ms): {status}")

        if output:
            results.to_json(output)
            console.print(f"\n[green]Results written to {output}[/green]")

    else:
        console.print("[red]No results collected![/red]")


@app.command()
def version() -> None:
    """Show version information."""
    from spark_vtuber import __version__

    console.print(f"Spark VTuber v{__version__}")


if __name__ == "__main__":
    app()
