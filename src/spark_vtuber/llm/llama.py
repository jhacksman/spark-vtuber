"""
Llama LLM implementation for Spark VTuber.

Supports Llama 3.1 and compatible models with vLLM backend.
"""

import asyncio
import time
from typing import AsyncIterator

from spark_vtuber.llm.base import BaseLLM, LLMResponse


class LlamaLLM(BaseLLM):
    """
    Llama model implementation using vLLM for efficient inference.

    Supports:
    - Llama 3.1 70B and smaller variants
    - 4-bit quantization (AWQ/GPTQ)
    - LoRA adapter loading
    - Streaming generation
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        quantization: str = "awq",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        **kwargs,
    ):
        """
        Initialize Llama LLM.

        Args:
            model_name: HuggingFace model name or local path
            quantization: Quantization method (awq, gptq, none)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            **kwargs: Additional vLLM arguments
        """
        super().__init__(model_name, **kwargs)
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._engine = None
        self._tokenizer = None
        self._adapters: dict[str, str] = {}
        self._active_adapter: str | None = None

    async def load(self) -> None:
        """Load the model using vLLM."""
        if self._loaded:
            self.logger.warning("Model already loaded")
            return

        self.logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs

            engine_args = AsyncEngineArgs(
                model=self.model_name,
                quantization=self.quantization if self.quantization != "none" else None,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                enable_lora=True,
                max_lora_rank=64,
            )

            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._loaded = True

            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f}s")

        except ImportError:
            self.logger.warning("vLLM not available, falling back to transformers")
            await self._load_transformers()

    async def _load_transformers(self) -> None:
        """Fallback loading using transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        self.logger.info("Loading with transformers backend")

        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        self._loaded = True
        self._use_transformers = True

    async def unload(self) -> None:
        """Unload the model from memory."""
        if not self._loaded:
            return

        self.logger.info("Unloading model")

        if self._engine:
            del self._engine
            self._engine = None

        if hasattr(self, "_model"):
            del self._model
            self._model = None

        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None

        import torch
        torch.cuda.empty_cache()

        self._loaded = False
        self.logger.info("Model unloaded")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a complete response."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        tokens = []

        async for token in self.generate_stream(
            prompt, max_tokens, temperature, top_p, stop_sequences
        ):
            tokens.append(token)

        text = "".join(tokens)
        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            text=text,
            tokens_generated=len(tokens),
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if self._engine:
            async for token in self._generate_vllm(
                prompt, max_tokens, temperature, top_p, stop_sequences
            ):
                yield token
        else:
            async for token in self._generate_transformers(
                prompt, max_tokens, temperature, top_p, stop_sequences
            ):
                yield token

    async def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str] | None,
    ) -> AsyncIterator[str]:
        """Generate using vLLM engine."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )

        request_id = f"req_{time.time()}"

        lora_request = None
        if self._active_adapter and self._active_adapter in self._adapters:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(
                self._active_adapter,
                1,
                self._adapters[self._active_adapter],
            )

        prev_text = ""
        async for output in self._engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        ):
            if output.outputs:
                current_text = output.outputs[0].text
                # vLLM returns cumulative text, extract only new tokens
                new_text = current_text[len(prev_text):]
                prev_text = current_text
                if new_text:
                    yield new_text

    async def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str] | None,
    ) -> AsyncIterator[str]:
        """Generate using transformers backend."""
        from transformers import TextIteratorStreamer
        import torch
        from threading import Thread

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "streamer": streamer,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            if stop_sequences:
                for stop in stop_sequences:
                    if stop in text:
                        text = text.split(stop)[0]
                        yield text
                        return
            yield text

        thread.join()

    async def load_lora_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load a LoRA adapter."""
        self.logger.info(f"Loading LoRA adapter: {adapter_name} from {adapter_path}")
        self._adapters[adapter_name] = adapter_path

        if hasattr(self, "_model") and self._model:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_path,
                adapter_name=adapter_name,
            )

    async def set_active_adapter(self, adapter_name: str | None) -> None:
        """Set the active LoRA adapter."""
        if adapter_name and adapter_name not in self._adapters:
            raise ValueError(f"Unknown adapter: {adapter_name}")

        self._active_adapter = adapter_name
        self.logger.info(f"Active adapter: {adapter_name or 'base model'}")

        if hasattr(self, "_model") and self._model and hasattr(self._model, "set_adapter"):
            if adapter_name:
                self._model.set_adapter(adapter_name)
            else:
                self._model.disable_adapter_layers()

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage."""
        import torch

        if not torch.cuda.is_available():
            return {"gpu_allocated_gb": 0, "gpu_reserved_gb": 0}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        return {
            "gpu_allocated_gb": round(allocated, 2),
            "gpu_reserved_gb": round(reserved, 2),
        }
