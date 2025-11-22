"""
Simple logging configuration for dspy_judge package.
"""

import logging
import os
from typing import Optional


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Name for the logger (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_from_env() -> None:
    """
    Configure logging from environment variables.
    
    Environment variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_FORMAT: Custom log format string
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    format_string = os.getenv("LOG_FORMAT")
    setup_logging(level, format_string)


# Set up basic logging when module is imported
setup_logging()