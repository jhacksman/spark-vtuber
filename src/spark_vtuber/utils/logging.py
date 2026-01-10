"""
Logging configuration for Spark VTuber.

Uses loguru for structured logging with rich formatting.
"""

import sys
from pathlib import Path
from typing import Literal

from loguru import logger


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level
        log_file: Optional file path for log output
        json_format: Whether to use JSON format for logs
    """
    # Remove default handler
    logger.remove()

    # Console format
    if json_format:
        console_format = "{message}"
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        serialize=json_format,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="100 MB",
            retention="7 days",
            compression="gz",
        )

    logger.info(f"Logging configured at level {level}")


def get_logger(name: str) -> "logger":
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance bound to the given name
    """
    return logger.bind(name=name)


class LoggerMixin:
    """Mixin class that provides a logger attribute."""

    @property
    def logger(self) -> "logger":
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
