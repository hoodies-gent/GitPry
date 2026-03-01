import logging
import os
from rich.logging import RichHandler

def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Configure the global logger with a defensive logging strategy.
    Prioritizes structured output via rich.
    """
    logger = logging.getLogger("gitpry")
    
    # Avoid adding handlers multiple times if setup is called again
    if not logger.handlers:
        level = logging.DEBUG if debug else logging.INFO
        logger.setLevel(level)

        handler = RichHandler(
            show_time=False,       # Clean output for CLI
            show_path=False,       # Hide internal file paths
            markup=True,           # Support rich markup in log messages
            rich_tracebacks=True   # Beautiful stack traces if exceptions do occur
        )
        
        # We define a very simple format since rich handles the rest
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Expose a default logger instance
# For CLI behavior, we'll configure it dynamically based on the --verbose flag or env var
is_debug_env = os.environ.get("GITPRY_DEBUG", "false").lower() == "true"
logger = setup_logger(debug=is_debug_env)
