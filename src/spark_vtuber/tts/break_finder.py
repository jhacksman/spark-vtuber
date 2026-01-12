"""
Break Finder Agent for intelligent TTS chunking.

Finds natural break points in LLM responses for pipeline buffering streaming.
The first chunk is sent to TTS immediately while remaining content generates
in the background during playback.

This agent is model-agnostic - configure any model via settings.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass

import httpx


@dataclass
class BreakPoint:
    """Result from break point detection."""

    position: int  # Character index where to split
    first_chunk: str  # Text before the break
    remainder: str  # Text after the break
    confidence: float  # 0.0-1.0 confidence in this break point
    method: str  # How the break was found (llm, heuristic, fallback)
    latency_ms: float  # Time taken to find the break


class BreakFinder:
    """
    Intelligent break point finder for TTS streaming.

    Uses an LLM to find natural break points in text, with fast heuristic
    fallbacks when the LLM is unavailable or too slow.

    The break finder identifies:
    - Interjections: "Oh!", "Well,", "Hmm,", "Ah,"
    - Short complete thoughts: "I see.", "Got it.", "Right,"
    - Questions: "You know what?", "Guess what?"
    - Emotional expressions: "Wow!", "Oh my!"
    - Natural pause points in longer text
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8001/v1",
        model_name: str = "Qwen/Qwen3-0.5B-Instruct",
        timeout_ms: int = 100,
        min_first_chunk_chars: int = 5,
        max_first_chunk_chars: int = 150,
        enabled: bool = True,
    ):
        """
        Initialize break finder.

        Args:
            api_base: OpenAI-compatible API base URL for break finder model
            model_name: Model to use for break finding
            timeout_ms: Maximum time to wait for LLM response
            min_first_chunk_chars: Minimum characters for first chunk
            max_first_chunk_chars: Maximum characters for first chunk
            enabled: Whether to use LLM-based break finding (falls back to heuristics if False)
        """
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.timeout_ms = timeout_ms
        self.min_first_chunk_chars = min_first_chunk_chars
        self.max_first_chunk_chars = max_first_chunk_chars
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_ms / 1000)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def find_break(self, text: str) -> BreakPoint:
        """
        Find the optimal break point in text for TTS streaming.

        Tries LLM-based detection first, falls back to heuristics if:
        - LLM is disabled
        - LLM times out
        - LLM returns invalid response

        Args:
            text: Full text to find break point in

        Returns:
            BreakPoint with position, chunks, and metadata
        """
        start_time = time.time()

        # Short text doesn't need breaking
        if len(text) <= self.min_first_chunk_chars:
            return BreakPoint(
                position=len(text),
                first_chunk=text,
                remainder="",
                confidence=1.0,
                method="short_text",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Try LLM-based detection if enabled
        if self.enabled:
            try:
                result = await asyncio.wait_for(
                    self._find_break_llm(text),
                    timeout=self.timeout_ms / 1000,
                )
                if result:
                    result.latency_ms = (time.time() - start_time) * 1000
                    return result
            except asyncio.TimeoutError:
                self.logger.debug("LLM break finder timed out, using heuristics")
            except Exception as e:
                self.logger.debug(f"LLM break finder failed: {e}, using heuristics")

        # Fall back to heuristics
        result = self._find_break_heuristic(text)
        result.latency_ms = (time.time() - start_time) * 1000
        return result

    async def _find_break_llm(self, text: str) -> BreakPoint | None:
        """
        Use LLM to find natural break point.

        The LLM is prompted to identify the best position to split the text
        for natural-sounding TTS output.
        """
        client = await self._get_client()

        # Truncate text if too long (we only need to analyze the beginning)
        analysis_text = text[: self.max_first_chunk_chars * 2]

        prompt = f"""Find the best position to split this text for text-to-speech.
The first chunk should be a natural-sounding phrase that can stand alone.
Look for: interjections, short complete thoughts, questions, or natural pauses.

Text: "{analysis_text}"

Reply with ONLY a number (the character position to split at, between {self.min_first_chunk_chars} and {self.max_first_chunk_chars}).
If no good break point exists, reply with {self.max_first_chunk_chars}."""

        try:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract position from response
            content = data["choices"][0]["message"]["content"].strip()
            position = int(re.search(r"\d+", content).group())

            # Validate position
            position = max(self.min_first_chunk_chars, min(position, len(text)))

            # Adjust to nearest word boundary
            position = self._adjust_to_word_boundary(text, position)

            return BreakPoint(
                position=position,
                first_chunk=text[:position].strip(),
                remainder=text[position:].strip(),
                confidence=0.9,
                method="llm",
                latency_ms=0,  # Will be set by caller
            )

        except Exception as e:
            self.logger.debug(f"LLM break finding failed: {e}")
            return None

    def _find_break_heuristic(self, text: str) -> BreakPoint:
        """
        Find break point using fast heuristics.

        Prioritizes:
        1. Interjections at the start
        2. Short complete sentences
        3. Comma-separated phrases
        4. Word boundaries near target length
        """
        # Common interjections and short phrases that make good first chunks
        interjection_patterns = [
            r"^(Oh[,!]?\s)",
            r"^(Ah[,!]?\s)",
            r"^(Well[,!]?\s)",
            r"^(Hmm[,!]?\s)",
            r"^(Wow[,!]?\s)",
            r"^(Hey[,!]?\s)",
            r"^(So[,!]?\s)",
            r"^(Yeah[,!]?\s)",
            r"^(Yes[,!]?\s)",
            r"^(No[,!]?\s)",
            r"^(Okay[,!]?\s)",
            r"^(Alright[,!]?\s)",
            r"^(Sure[,!]?\s)",
            r"^(Right[,!]?\s)",
            r"^(Look[,!]?\s)",
            r"^(Listen[,!]?\s)",
            r"^(See[,!]?\s)",
            r"^(You know[,!]?\s)",
            r"^(I mean[,!]?\s)",
            r"^(Actually[,!]?\s)",
            r"^(Honestly[,!]?\s)",
            r"^(Basically[,!]?\s)",
        ]

        # Check for interjections
        for pattern in interjection_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                # Include a bit more after the interjection if possible
                pos = match.end()
                extended = self._extend_to_natural_break(text, pos)
                if extended <= self.max_first_chunk_chars:
                    return BreakPoint(
                        position=extended,
                        first_chunk=text[:extended].strip(),
                        remainder=text[extended:].strip(),
                        confidence=0.8,
                        method="interjection",
                        latency_ms=0,
                    )

        # Look for short complete sentences
        sentence_end = re.search(
            r"^.{" + str(self.min_first_chunk_chars) + r"," + str(self.max_first_chunk_chars) + r"}?[.!?](?:\s|$)",
            text,
        )
        if sentence_end:
            pos = sentence_end.end()
            return BreakPoint(
                position=pos,
                first_chunk=text[:pos].strip(),
                remainder=text[pos:].strip(),
                confidence=0.85,
                method="sentence",
                latency_ms=0,
            )

        # Look for comma breaks
        comma_match = re.search(
            r"^.{" + str(self.min_first_chunk_chars) + r"," + str(self.max_first_chunk_chars) + r"}?,\s",
            text,
        )
        if comma_match:
            pos = comma_match.end()
            return BreakPoint(
                position=pos,
                first_chunk=text[:pos].strip(),
                remainder=text[pos:].strip(),
                confidence=0.7,
                method="comma",
                latency_ms=0,
            )

        # Fall back to word boundary near target length
        target = min(self.max_first_chunk_chars, len(text))
        pos = self._adjust_to_word_boundary(text, target)

        return BreakPoint(
            position=pos,
            first_chunk=text[:pos].strip(),
            remainder=text[pos:].strip(),
            confidence=0.5,
            method="fallback",
            latency_ms=0,
        )

    def _extend_to_natural_break(self, text: str, start_pos: int) -> int:
        """Extend position to next natural break point."""
        # Look for punctuation or comma within reasonable distance
        search_text = text[start_pos : start_pos + 50]

        # Find nearest break
        for i, char in enumerate(search_text):
            if char in ".!?,;:":
                return start_pos + i + 1

        # No punctuation found, find word boundary
        space_pos = search_text.find(" ")
        if space_pos > 0:
            return start_pos + space_pos

        return start_pos

    def _adjust_to_word_boundary(self, text: str, position: int) -> int:
        """Adjust position to nearest word boundary."""
        if position >= len(text):
            return len(text)

        # If we're at a space, we're good
        if text[position] == " ":
            return position

        # Look backwards for space
        for i in range(position, max(0, position - 20), -1):
            if text[i] == " ":
                return i + 1

        # Look forwards for space
        for i in range(position, min(len(text), position + 20)):
            if text[i] == " ":
                return i

        return position
