#!/usr/bin/env python3

"""Logger that handles two outputs (stdout and file)."""

import logging
import pathlib

from datetime import datetime as dt
from pytz import timezone as tz

tz0 = tz("Europe/Berlin")

# LOGGING_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


class Log0:
    """
    A logger that handles two outputs (stdout and file).

    This class provides a convenient way to set up logging with both console and file output.
    It allows for different logging levels for console and file outputs.

    Parameters
    ----------
    dir0 : str, optional
        Directory where log files will be stored. Default is "logs".
    fn0 : str, optional
        Name of the log file. If None, a timestamp-based name will be used.
    write : bool, optional
        Whether to write logs to a file. Default is False.
    stream_lvl : str, optional
        Logging level for console output. Default is "INFO".
    file_lvl : str, optional
        Logging level for file output. Default is "DEBUG".

    Attributes
    ----------
    handler0 : logging.StreamHandler
        Handler for console output.
    handler1 : logging.FileHandler
        Handler for file output (only if write=True).
    logger : logging.Logger
        The logger object.
    of0 : pathlib.Path or None
        Path to the log file (None if write=False).

    Methods
    -------
    __init__(self, dir0="logs", fn0=None, write=False, stream_lvl="INFO", file_lvl="DEBUG")
        Initialize the Log0 instance.

    Examples
    --------
    Without writing to log file:

    >>> import mvn
    >>> logZ = mvn.Log0(
    ...     write=False,
    ...     stream_lvl="INFO",
    ...     file_lvl="DEBUG",
    ... )
    >>> log0 = logZ.logger
    >>> log0.info(f"{logZ.of0 = }")

    Simple usage:

    >>> import mvn
    >>> import pathlib
    >>> logZ = mvn.Log0()
    >>> log0 = logZ.logger
    >>> log0.info(f"{pathlib.Path.cwd() = }")

    With writing to log file:

    >>> logZ = mvn.Log0(
    ...     write=True,
    ...     stream_lvl="INFO",
    ...     file_lvl="DEBUG",
    ... )
    >>> log0 = logZ.logger
    >>> log0.info(f"{logZ.of0 = }")

    Changing logging levels:

    >>> log0.info(f"handler0: {logZ.logging.getLevelName(logZ.handler0.level)}")
    >>> log0.info(f"handler1: {logZ.logging.getLevelName(logZ.handler1.level)}")
    >>> log0.info(f"logger: {logZ.logging.getLevelName(log0.level)}")
    >>> logZ.handler0.setLevel("DEBUG")
    >>> log0.info(f"handler0: {logZ.logging.getLevelName(logZ.handler0.level)}")
    >>> log0.info(f"handler1: {logZ.logging.getLevelName(logZ.handler1.level)}")
    >>> log0.info(f"logger: {logZ.logging.getLevelName(log0.level)}")
    >>> log0.setLevel("CRITICAL")
    >>> log0.info(f"handler0: {logZ.logging.getLevelName(logZ.handler0.level)}")
    >>> log0.info(f"handler1: {logZ.logging.getLevelName(logZ.handler1.level)}")
    >>> log0.info(f"logger: {logZ.logging.getLevelName(log0.level)}")
    >>> # no output expected from log0.info after setting "CRITICAL" logging level

    Notes
    -----
    Logging levels and their corresponding numeric values:

    - CRITICAL: 50
    - ERROR: 40
    - WARNING: 30
    - INFO: 20
    - DEBUG: 10
    - NOTSET: 0

    The overall logging level is set to the minimum of console and file levels.
    When changing logging levels after initialization, remember to update both
    handler levels and the overall logger level if necessary.

    The logger uses different formats for console and file outputs:
    - Console: "%(levelname).1s: %(message)s"
    - File: "%(asctime)s %(levelname).1s: %(funcName)-16s %(message)s"

    The log file is created with mode 0o600 (read, and write for owner only).
    """

    def __init__(
        self,
        dir0="logs",
        fn0=None,
        write=False,
        stream_lvl="INFO",
        file_lvl="DEBUG",
    ):
        """Initialize Log0 class."""
        # Setup logging stream handler
        self.handler0 = logging.StreamHandler()
        self.handler0.setFormatter(
            logging.Formatter(
                " ".join(
                    [
                        # "%(asctime)s",
                        # "%(name)s",
                        "%(levelname).1s:",
                        # "%(module)s",
                        # "%(funcName)-16s ",
                        "%(message)s",
                    ]
                ),
                datefmt="%Y%m%dT%H%M%S",
            )
        )
        self.file_lvl = file_lvl
        self.stream_lvl = stream_lvl
        self.logging = logging  # module accessible from instance
        self.logger = logging.getLogger(__name__)
        self.handler0.setLevel(self.stream_lvl)
        self.logger.setLevel(self.handler0.level)

        # Detach any old handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Attach new handle
        self.logger.addHandler(self.handler0)

        if not write:
            self.of0 = None
        else:
            self.dir0 = pathlib.Path(dir0)
            self.fn0 = (
                str(fn0)
                if fn0 is not None
                else f"{dt.now(tz0).strftime('%Y%m%dT%H%M%S')}.log"
            )
            self.of0 = self.dir0 / self.fn0
            self.dir0.mkdir(mode=0o600, parents=True, exist_ok=True)
            # Setup logging file handler
            self.handler1 = logging.FileHandler(self.of0)
            self.handler1.setFormatter(
                logging.Formatter(
                    " ".join(
                        [
                            "%(asctime)s",
                            # "%(name)s",
                            "%(levelname).1s:",
                            # "%(module)s",
                            "%(funcName)-16s ",
                            "%(message)s",
                        ]
                    ),
                    datefmt="%Y%m%dT%H%M%S",
                )
            )

            # Set logging levels
            self.handler1.setLevel(self.file_lvl)
            self.logger.setLevel(min(self.handler0.level, self.handler1.level))
            # Attach new handle
            self.logger.addHandler(self.handler1)
