import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import colorlog

LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def config_logger(
    log_name: str,
    logger_name: str,
    default_folder: str,
    write_mode: str = "a",
    level: str = "debug",
    clear_log: bool = False,
) -> logging.Logger:
    """
    Configures and returns a logger with specified settings.

    Args:
        log_name (str): The name of the log file.
        logger_name (str): The name of the logger.
        default_folder (str): The root folder where logs are stored.
        write_mode (str): The mode in which the log file is opened. Defaults to 'a' (append).
        level (str): The logging level. Defaults to 'debug'.
        clear_log (bool): If True, existing log file will be cleared. Defaults to False.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL[level.lower()])

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

        # if not logger.hasHandlers():
    basic_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)",
        "%y-%m-%d %H:%M:%S",
    )
    formatter = colorlog.ColoredFormatter(
        "%(asctime)s %(log_color)s [%(levelname)-.4s] %(reset)s %(message)s %(purple)s (%(filename)s:%(lineno)s)",
        "%y-%m-%d %H:%M:%S",
    )

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(LOG_LEVEL[level.lower()])
    logger.addHandler(stream_handler)

    create_day = datetime.now().strftime("%y-%m-%d")
    log_root_path = os.path.join(default_folder, create_day)
    if not os.path.isdir(log_root_path):
        os.makedirs(log_root_path)

    # File handler
    if log_name:
        log_path = os.path.join(log_root_path, log_name)
        if clear_log and os.path.exists(log_path):
            logging.warning("Clearing existing log files")
            os.remove(log_path)
        file_handler = RotatingFileHandler(
            filename=log_path,
            mode=write_mode,
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        file_handler.setFormatter(basic_formatter)
        file_handler.setLevel(LOG_LEVEL[level.lower()])
        logger.addHandler(file_handler)

    logger.info(f"Create logger.({logger.name})")
    logger.info(
        "Enabled stream {}".format(
            f"and file mode.({log_name})" if log_name else "mode"
        )
    )
    return logger
