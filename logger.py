#!/usr/bin/env python3
"""
Centralized logging system for noise detector.

This module provides structured logging with levels, file output, and
debugging capabilities. Use this instead of print() statements for
better debuggability in production.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("System started")
    log.error("Failed to open device", exc_info=True)
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Color-coded log formatter for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    debug: bool = False
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, only console logging)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        debug: If True, enable DEBUG level and verbose output
    
    Returns:
        Root logger
    """
    if debug:
        level = "DEBUG"
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger):
    """Log system information for debugging."""
    import platform
    import sys
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info("=" * 60)

