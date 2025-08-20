import logging
import os
from pathlib import Path
from tqdm import tqdm

# Silence adalflow logging
logging.getLogger("adalflow").setLevel(logging.CRITICAL)

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    grey = "\x1b[90m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    reset = "\x1b[0m"
    format = "[RAGalyze][%(levelname_title)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset,
    }

    def format(self, record):
        record.levelname_title = record.levelname.title()
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(format: str = None, log_name: str = None):
    """
    Configure logging for the application.
    Reads LOG_LEVEL and LOG_FILE_PATH from environment (defaults: INFO, logs/application.log).
    Ensures log directory exists, and configures both file and console handlers.
    """
    # Determine log directory and default file path
    base_dir = Path(__file__).parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = log_dir / "application.log"

    # Get log level and file path from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file_path = Path(os.environ.get("LOG_FILE_PATH", str(default_log_file)))

    # ensure log_file_path is within the project's logs directory to prevent path traversal
    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(
            f"logger/logging_config.py:LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'"
        )
    # Ensure parent dirs exist for the log file
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # File log (plain format)
    file_handler = logging.FileHandler(resolved_path)
    file_handler.setFormatter(
        logging.Formatter(
            format
            or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
        )
    )

    # Console log (plain format, for root logger)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            format or "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Set all core loggers to use TqdmCompatibleLogger by default
    logging.setLoggerClass(TqdmCompatibleLogger)

    # Configure core loggers to prevent propagation and avoid duplicate output
    for logger_name in ["core"]:
        project_logger = logging.getLogger(logger_name)
        project_logger.handlers.clear()  # Prevent duplicate handlers
        project_logger.propagate = (
            False  # Prevent propagation to root logger to avoid duplicate output
        )

    # Reset logger class to default for other loggers
    logging.setLoggerClass(logging.Logger)

    # Initial debug message to confirm configuration
    if log_name:
        logger = logging.getLogger(log_name)
    else:
        logger = logging.getLogger(__name__)
    logger.debug(f"Log level set to {log_level_str}, log file: {resolved_path}")


class TqdmCompatibleLogger(logging.Logger):
    """
    Custom Logger class that handles tqdm progress bar conflicts.
    Overrides info, warning, and error methods to detect and properly format
    progress bar lines vs regular log messages.
    """

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._current_line = ""

    def _handle_tqdm_output(self, message, level_name, color_code="", icon=""):
        """
        Handle output that might contain tqdm progress bars.
        """
        try:
            stripped_message = str(message).strip()

            if not stripped_message:
                return

            # Get the module name from the logger
            module_path = self.name if self.name != "__main__" else "main"
            # Ensure consistent alignment by using fixed width for all icons
            # Each icon is padded to exactly 4 characters for visual alignment
            if icon == "⚠️":
                padded_icon = "⚠️  "  # Warning icon + 2 spaces
            elif icon == "ℹ️":
                padded_icon = "ℹ️  "  # Info icon + 2 spaces
            else:  # "❌" for error
                padded_icon = "❌ "  # Error icon + 2 spaces
            formatted_message = (
                padded_icon + f"{color_code}[{module_path}] {stripped_message}\033[0m"
            )

            # Try to use tqdm.write() if tqdm is available and active
            try:
                # Use tqdm.write() to avoid interfering with progress bars
                tqdm.write(formatted_message)
            except (ImportError, AttributeError):
                # Fallback to regular print if tqdm is not available
                # Check if this is a tqdm progress bar line (contains % and |)
                if stripped_message and (
                    "|" in stripped_message and "%" in stripped_message
                ):
                    # This looks like a progress bar - print with carriage return
                    print(f"\r{formatted_message}", end="", flush=True)
                    self._current_line = stripped_message
                else:
                    # Regular log line - print normally with newline
                    if (
                        self._current_line
                    ):  # If we had a progress bar, add newline first
                        print()
                        self._current_line = ""
                    print(formatted_message)
                    self._current_line = ""

        except Exception as e:
            # Fallback to standard logging if something goes wrong
            super().error(f"Error in tqdm-compatible logging: {e}")
            super().log(getattr(logging, level_name.upper()), message)

    def info(self, message, *args, **kwargs):
        """
        Override info method to handle tqdm progress bars.
        """
        if args or kwargs:
            message = message % args if args else message
        self._handle_tqdm_output(message, "Info", "\x1b[92m", "ℹ️")

    def warning(self, message, *args, **kwargs):
        """
        Override warning method to handle tqdm progress bars.
        """
        if args or kwargs:
            message = message % args if args else message
        self._handle_tqdm_output(message, "Warning", "\x1b[93m", "⚠️")

    def error(self, message, *args, **kwargs):
        """
        Override error method to handle tqdm progress bars.
        """
        if args or kwargs:
            message = message % args if args else message
        self._handle_tqdm_output(message, "Error", "\x1b[91m", "❌")


def get_tqdm_compatible_logger(name, setup_logging_config=True):
    """
    Get a tqdm-compatible logger instance with optional logging setup.

    Args:
        name: Logger name
        setup_logging_config: Whether to run setup_logging configuration (default: True)
    """
    # Set up logging configuration if requested
    if setup_logging_config:
        # Check if logging has already been configured to avoid duplicate setup
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            setup_logging(log_name=name)

    # Check if logger already exists and is the right type
    existing_logger = logging.getLogger(name)
    if isinstance(existing_logger, TqdmCompatibleLogger):
        return existing_logger

    # Create new TqdmCompatibleLogger instance
    logging.setLoggerClass(TqdmCompatibleLogger)
    # Force creation of new logger by clearing the existing one
    if name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[name]
    logger = logging.getLogger(name)
    logging.setLoggerClass(logging.Logger)  # Reset to default
    return logger
