import logging
import os
import sys
from typing import Optional

def setup_logger(
    name: str = "amc",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Initializes and configures the logger.

    Args:
        name: Name of the logger.
        level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: File path to save logs to. If None, logs to console only.
        log_format: Custom format string for logging.

    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Define format
    if not log_format:
        log_format = "%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s"
    
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create directory if it does not exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
