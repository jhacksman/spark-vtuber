"""
LLMVoX streaming TTS implementation for Spark VTuber.

LLMVoX is a lightweight 30M-parameter, LLM-agnostic, autoregressive streaming
Text-to-Speech system that provides true streaming synthesis with ~300ms latency.

Based on: https://arxiv.org/abs/2503.04724 (ACL 2025 Findings)
Code: https://github.com/mbzuai-oryx/LLMVoX

Key features:
- True streaming: generates audio token-by-token as text arrives
- Low latency: ~300ms end-to-end latency
- Lightweight: only 30M parameters (~500MB VRAM)
- LLM-agnostic: works with any LLM token stream
"""

import asyncio
import os
import time
from pathlib import Path
from queue import Empty, Queue
from typing import AsyncIterator

import numpy as np

from spark_vtuber.tts.base import BaseTTS, TTSResult


class LLMVoXTTS(BaseTTS):
    """
    LLMVoX streaming TTS implementation.

    Provides true streaming TTS that generates audio as text tokens arrive,
    unlike traditional TTS that waits for complete sentences.

    Memory usage: ~500MB VRAM (30M parameters)
    Latency: ~300ms first chunk (true streaming mode)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        model_path: str | None = None,
        wavtokenizer_config_path: str | None = None,
        wavtokenizer_model_path: str | None = None,
        encoder_model: str = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100",
        tokenizer_model: str = "google/byt5-small",
        device: str = "cuda",
        initial_chunk_size: int = 10,
        max_chunk_size: int = 1280,
        **kwargs,
    ):
        """
        Initialize LLMVoX TTS.

        Args:
            sample_rate: Output sample rate (24000 for LLMVoX)
            model_path: Path to LLMVoX checkpoint (auto-downloads if not specified)
            wavtokenizer_config_path: Path to WavTokenizer config
            wavtokenizer_model_path: Path to WavTokenizer model
            encoder_model: HuggingFace model for text encoding
            tokenizer_model: HuggingFace tokenizer model
            device: Device to run inference on (cuda, cpu)
            initial_chunk_size: Initial audio chunk size for streaming
            max_chunk_size: Maximum audio chunk size
            **kwargs: Additional arguments
        """
        super().__init__(sample_rate, **kwargs)
        self.model_path = model_path
        self.wavtokenizer_config_path = wavtokenizer_config_path
        self.wavtokenizer_model_path = wavtokenizer_model_path
        self.encoder_model = encoder_model
        self.tokenizer_model = tokenizer_model
        self.device = device
        self.initial_chunk_size = initial_chunk_size
        self.max_chunk_size = max_chunk_size

        # Model components (loaded lazily)
        self._wavtokenizer = None
        self._tokenizer = None
        self._llm_encoder = None
        self._gpt_model = None
        self._first_chunk_latency_ms: float = 0.0

    async def load(self) -> None:
        """Load the LLMVoX model components."""
        if self._loaded:
            self.logger.warning("LLMVoX TTS already loaded")
            return

        self.logger.info("Loading LLMVoX TTS...")
        start_time = time.time()

        loop = asyncio.get_event_loop()

        def _load_sync():
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            device = torch.device(self.device)

            # Download checkpoints if not specified
            model_dir = self._ensure_checkpoints()

            # Load WavTokenizer
            self.logger.info("Loading WavTokenizer...")
            wavtokenizer = self._load_wavtokenizer(model_dir, device)

            # Load text encoder (T5-based G2P)
            self.logger.info(f"Loading text encoder from {self.encoder_model}...")
            llm_model = T5ForConditionalGeneration.from_pretrained(self.encoder_model)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)

            # Add special tokens
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.add_special_tokens({"pad_token": "EOS"})
            llm_model.resize_token_embeddings(len(tokenizer))

            # Use only the encoder's embedding layer
            llm_encoder = llm_model.encoder.embed_tokens.to(device)

            # Load GPT model for speech generation
            self.logger.info("Loading LLMVoX GPT model...")
            gpt_model = self._load_gpt_model(model_dir, device)

            return {
                "wavtokenizer": wavtokenizer,
                "tokenizer": tokenizer,
                "llm_encoder": llm_encoder,
                "gpt_model": gpt_model,
                "device": device,
            }

        models = await loop.run_in_executor(None, _load_sync)

        self._wavtokenizer = models["wavtokenizer"]
        self._tokenizer = models["tokenizer"]
        self._llm_encoder = models["llm_encoder"]
        self._gpt_model = models["gpt_model"]
        self._device = models["device"]

        self._loaded = True
        load_time = time.time() - start_time
        self.logger.info(f"LLMVoX TTS loaded in {load_time:.2f}s")

    def _ensure_checkpoints(self) -> Path:
        """Ensure model checkpoints are downloaded."""
        from huggingface_hub import hf_hub_download, snapshot_download

        # Default model directory
        model_dir = Path(self.model_path) if self.model_path else Path("./models/llmvox")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Download LLMVoX checkpoint if not present
        llmvox_ckpt = model_dir / "ckpt_english_tiny.pt"
        if not llmvox_ckpt.exists():
            self.logger.info("Downloading LLMVoX checkpoint from HuggingFace...")
            hf_hub_download(
                repo_id="MBZUAI/LLMVoX",
                filename="ckpt_english_tiny.pt",
                local_dir=str(model_dir),
            )

        # Download WavTokenizer checkpoint if not present
        wavtokenizer_ckpt = model_dir / "wavtokenizer_large_speech_320_24k.ckpt"
        if not wavtokenizer_ckpt.exists():
            self.logger.info("Downloading WavTokenizer checkpoint from HuggingFace...")
            hf_hub_download(
                repo_id="MBZUAI/LLMVoX",
                filename="wavtokenizer_large_speech_320_24k.ckpt",
                local_dir=str(model_dir),
            )

        # Download WavTokenizer config if not present
        wavtokenizer_config = model_dir / "wavtokenizer_config.yaml"
        if not wavtokenizer_config.exists():
            self.logger.info("Downloading WavTokenizer config...")
            # Try to download from LLMVoX repo
            try:
                hf_hub_download(
                    repo_id="MBZUAI/LLMVoX",
                    filename="wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                    local_dir=str(model_dir),
                )
            except Exception:
                # Create a minimal config if download fails
                self._create_wavtokenizer_config(wavtokenizer_config)

        return model_dir

    def _create_wavtokenizer_config(self, config_path: Path) -> None:
        """Create a minimal WavTokenizer config."""
        config = """
# WavTokenizer config for LLMVoX
model:
  type: wavtokenizer
  sample_rate: 24000
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  n_mels: 80
  codebook_size: 4096
  codebook_dim: 512
"""
        config_path.write_text(config)

    def _load_wavtokenizer(self, model_dir: Path, device):
        """Load WavTokenizer model."""
        import torch

        # Try to import WavTokenizer
        try:
            from WavTokenizer.decoder.pretrained import WavTokenizer
        except ImportError:
            # WavTokenizer not installed, use a simple fallback
            self.logger.warning(
                "WavTokenizer not installed. Install from: "
                "https://github.com/jishengpeng/WavTokenizer"
            )
            return None

        config_path = self.wavtokenizer_config_path or str(
            model_dir / "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        )
        model_path = self.wavtokenizer_model_path or str(
            model_dir / "wavtokenizer_large_speech_320_24k.ckpt"
        )

        wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        return wavtokenizer.to(device)

    def _load_gpt_model(self, model_dir: Path, device):
        """Load LLMVoX GPT model for speech generation."""
        import math
        from dataclasses import dataclass

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        @dataclass
        class GPTConfig:
            block_size: int = 1024
            vocab_size: int = 50304
            n_layer: int = 12
            n_head: int = 12
            n_embd: int = 768
            dropout: float = 0.0
            bias: bool = True
            is_train: bool = False

        class LayerNorm(nn.Module):
            def __init__(self, ndim, bias):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(ndim))
                self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

            def forward(self, x):
                return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

        class CausalSelfAttention(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
                self.attn_dropout = nn.Dropout(config.dropout)
                self.resid_dropout = nn.Dropout(config.dropout)
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.dropout = config.dropout
                self.is_train = config.is_train
                self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

            def forward(self, x, kvcache=None):
                B, T, C = x.size()
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

                if kvcache:
                    prev_k, prev_v = kvcache
                    k = torch.cat([prev_k, k], dim=1)
                    v = torch.cat([prev_v, v], dim=1)

                new_kvcache = [k, v]
                curr_T = k.shape[1]

                k = k.view(B, curr_T, self.n_head, C // self.n_head).transpose(1, 2)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                v = v.view(B, curr_T, self.n_head, C // self.n_head).transpose(1, 2)

                if self.flash:
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_train
                    )
                else:
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    att = F.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v

                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self.resid_dropout(self.c_proj(y))
                return y, new_kvcache

        class MLP(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
                self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, x):
                x = self.c_fc(x)
                x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
                x = self.c_proj(x)
                x = self.dropout(x)
                return x

        class Block(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
                self.attn = CausalSelfAttention(config)
                self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
                self.mlp = MLP(config)

            def forward(self, x, kvcache=None):
                attn_out, cache_ele = self.attn(self.ln_1(x), kvcache)
                x = x + attn_out
                x = x + self.mlp(self.ln_2(x))
                return x, cache_ele

        class GPT(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.transformer = nn.ModuleDict(
                    dict(
                        wpe=nn.Embedding(config.block_size, config.n_embd),
                        drop=nn.Dropout(config.dropout),
                        h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                        ln_f=LayerNorm(config.n_embd, bias=config.bias),
                    )
                )
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            def forward(self, emb, kvcache=None):
                b, t, _ = emb.size()
                pos = torch.arange(0, t, dtype=torch.long, device=emb.device).unsqueeze(0)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(emb + pos_emb)

                if not kvcache:
                    kvcache = [None] * self.config.n_layer
                else:
                    x = x[:, [-1], :]

                new_kvcache = []
                for block, kvcache_block in zip(self.transformer.h, kvcache):
                    x, cache_ele = block(x, kvcache=kvcache_block)
                    new_kvcache.append(cache_ele)

                x = self.transformer.ln_f(x)
                logits = self.lm_head(x[:, [-1], :])
                return logits, None, new_kvcache

        # Load checkpoint
        ckpt_path = model_dir / "ckpt_english_tiny.pt"
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        model_args = {
            k: checkpoint_model_args[k]
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]
        }
        model_args["is_train"] = False

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        self.logger.info(f"LLMVoX GPT model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        return model

    async def unload(self) -> None:
        """Unload the LLMVoX model."""
        if not self._loaded:
            return

        self.logger.info("Unloading LLMVoX TTS")

        self._wavtokenizer = None
        self._tokenizer = None
        self._llm_encoder = None
        self._gpt_model = None

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_id: Voice reference ID (not used in LLMVoX)
            speed: Speech speed multiplier
            emotion: Emotion tag (not used in LLMVoX)

        Returns:
            TTSResult with audio data
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()

        # Collect all audio chunks from streaming
        audio_chunks = []
        async for chunk in self.synthesize_stream(text, voice_id, speed, emotion):
            audio_chunks.append(chunk)

        audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)

        latency_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            latency_ms=latency_ms,
            metadata={
                "engine": "llmvox",
                "first_chunk_latency_ms": self._first_chunk_latency_ms,
            },
        )

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize speech with TRUE streaming output.

        LLMVoX generates audio token-by-token as text is processed,
        providing true streaming with ~300ms first-chunk latency.

        Args:
            text: Text to synthesize
            voice_id: Voice reference ID (not used)
            speed: Speech speed multiplier
            emotion: Emotion tag (not used)
            chunk_size: Audio chunk size in samples

        Yields:
            Audio chunks as numpy arrays
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()
        first_chunk = True

        loop = asyncio.get_event_loop()

        # Generate audio using LLMVoX streaming
        audio_queue: Queue = Queue()

        def _generate_audio():
            """Generate audio tokens in a separate thread."""
            import torch
            import torch.nn.functional as F

            device = self._device
            pad_token_id = 384
            eoa_token_id = 453
            dump_size = self.initial_chunk_size
            max_dump = self.max_chunk_size
            max_audio_len = 8000

            # Clean text
            text_clean = self._clean_text(text)

            # Tokenize text
            text_tokens = self._tokenizer(text_clean)["input_ids"]
            text_tokens = text_tokens + [385]  # Add EOS token
            text_tokens = torch.tensor(text_tokens).unsqueeze(0).to(device)
            text_embeddings = self._llm_encoder(text_tokens)

            # Initialize generation state
            speech_gen_index = 0
            current_speech_token = None
            speech_outputs = []
            kvcache = None
            bandwidth_id = torch.tensor([0]).to(device)

            with torch.inference_mode():
                for i in range(text_embeddings.shape[1]):
                    # Get speech embedding
                    if speech_gen_index == 0:
                        speech_embed = torch.zeros((1, 1, 512), device=device)
                    else:
                        speech_token = torch.tensor([[current_speech_token]]).to(device)
                        if self._wavtokenizer is not None:
                            speech_embed = self._wavtokenizer.codes_to_features(speech_token).permute(0, 2, 1).to(device)
                        else:
                            speech_embed = torch.zeros((1, 1, 512), device=device)

                    # Combine text and speech embeddings
                    text_embed = text_embeddings[:, i, :].unsqueeze(1)
                    speech_decoder_input = torch.cat([text_embed, speech_embed], dim=2)
                    speech_decoder_input = F.normalize(speech_decoder_input, p=2, dim=2, eps=1e-8)

                    # Add previous context
                    if speech_gen_index > 0:
                        speech_decoder_input = torch.cat([speech_decoder_input_prev, speech_decoder_input], dim=1)

                    # Generate next speech token
                    speech_decoder_output, _, kvcache = self._gpt_model(speech_decoder_input, kvcache=kvcache)
                    logits = speech_decoder_output[:, -1, :]
                    probs = F.softmax(logits, dim=-1)

                    current_speech_token = probs.argmax(dim=-1).item()
                    speech_outputs.append(current_speech_token)

                    speech_decoder_input_prev = speech_decoder_input
                    speech_gen_index += 1

                    # Check if we have enough tokens to dump audio
                    if len(speech_outputs) >= dump_size:
                        token_batch = speech_outputs[:dump_size]
                        speech_outputs = speech_outputs[dump_size:]

                        # Convert tokens to audio
                        if self._wavtokenizer is not None:
                            predicted_tokens = torch.tensor([token_batch]).to(device)
                            features = self._wavtokenizer.codes_to_features(predicted_tokens)
                            audio_out = self._wavtokenizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)
                            audio_np = audio_out.cpu().numpy().astype("float32")
                            audio_queue.put(audio_np)

                        # Increase dump size for faster streaming
                        if dump_size < max_dump:
                            dump_size = min(dump_size * 3, max_dump)

                    # Check for end-of-audio token
                    if current_speech_token == eoa_token_id or len(speech_outputs) > max_audio_len:
                        break

                # Dump remaining tokens
                if speech_outputs and self._wavtokenizer is not None:
                    predicted_tokens = torch.tensor([speech_outputs]).to(device)
                    features = self._wavtokenizer.codes_to_features(predicted_tokens)
                    audio_out = self._wavtokenizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)
                    audio_np = audio_out.cpu().numpy().astype("float32")
                    audio_queue.put(audio_np)

            audio_queue.put(None)  # Signal end of generation

        # Start generation in background thread
        import threading

        gen_thread = threading.Thread(target=_generate_audio, daemon=True)
        gen_thread.start()

        # Yield audio chunks as they become available
        while True:
            try:
                audio_chunk = await loop.run_in_executor(None, lambda: audio_queue.get(timeout=30.0))
                if audio_chunk is None:
                    break

                if first_chunk:
                    self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                    first_chunk = False

                # Apply speed adjustment if needed
                if speed != 1.0:
                    import scipy.signal

                    new_length = int(len(audio_chunk) / speed)
                    audio_chunk = scipy.signal.resample(audio_chunk, new_length).astype(np.float32)

                yield audio_chunk

            except Empty:
                self.logger.warning("Audio generation timeout")
                break

        gen_thread.join(timeout=5.0)

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS processing."""
        import re

        text = text.strip()
        text = text.replace("**", "")
        text = text.replace("-", " ")
        text = re.sub(r"(\d)\.(?=\s|$)", r"\1", text)
        text = re.sub(r"\*", "", text)
        text = re.sub(r"#", " number ", text)
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"@", " at ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\.{3,}", " pause ", text)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        text = re.sub(r"\/+", " slash ", text)
        text = re.sub(r"\\+", " backslash ", text)
        return text

    async def synthesize_from_token_stream(
        self,
        token_stream: AsyncIterator[str],
        voice_id: str | None = None,
        speed: float = 1.0,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize speech from a streaming token source (e.g., vLLM).

        This is the key method for true end-to-end streaming:
        vLLM tokens -> LLMVoX -> audio chunks

        Args:
            token_stream: Async iterator of text tokens from LLM
            voice_id: Voice reference ID (not used)
            speed: Speech speed multiplier

        Yields:
            Audio chunks as numpy arrays
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()
        first_chunk = True

        # Accumulate tokens until we have a sentence boundary
        text_buffer = ""
        sentence_endings = {".", "!", "?", ";", ":"}

        async for token in token_stream:
            text_buffer += token

            # Check for sentence boundary
            if any(text_buffer.rstrip().endswith(end) for end in sentence_endings):
                # Synthesize the accumulated text
                async for audio_chunk in self.synthesize_stream(text_buffer.strip(), voice_id, speed):
                    if first_chunk:
                        self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                        first_chunk = False
                    yield audio_chunk
                text_buffer = ""

        # Synthesize any remaining text
        if text_buffer.strip():
            async for audio_chunk in self.synthesize_stream(text_buffer.strip(), voice_id, speed):
                if first_chunk:
                    self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                    first_chunk = False
                yield audio_chunk

    async def clone_voice(
        self,
        reference_audio: np.ndarray,
        voice_id: str,
    ) -> None:
        """
        Clone a voice from reference audio.

        Note: LLMVoX does not support voice cloning in the base model.
        This is a no-op placeholder for API compatibility.

        Args:
            reference_audio: Reference audio samples
            voice_id: Identifier for the cloned voice
        """
        self.logger.warning("LLMVoX does not support voice cloning. Using default voice.")

    def get_available_voices(self) -> list[str]:
        """Get list of available voice IDs."""
        return ["default"]

    def get_first_chunk_latency(self) -> float:
        """Get the latency of the first audio chunk in milliseconds."""
        return self._first_chunk_latency_ms
