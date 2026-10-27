# coding=utf-8

import logging
from logging import handlers

__all__ = ["Logger"]


class Logger:
    """Logger object"""

    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(
        self,
        filename,
        level="info",
        when="D",
        backCount=3,
        screen_fmt="%(levelname)s: %(message)s",
        file_fmt="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    ):
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations[level])

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(screen_fmt))
        sh.setLevel(self.level_relations[level])

        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding="utf-8"
        )
        th.setFormatter(logging.Formatter(file_fmt))
        th.setLevel(self.level_relations[level])

        self.logger.addHandler(sh)
        self.logger.addHandler(th)
