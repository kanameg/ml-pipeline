import logging
from logging import StreamHandler, FileHandler, Formatter
from logging import INFO, DEBUG, NOTSET
import os
import yaml

# -------------------------------------------------------------------
# read configuration YAML file
# -------------------------------------------------------------------
CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.safe_load(file)

LOG_DIR = yml['SETTING']['LOG_DIR']
MODEL_DIR = yml['SETTING']['MODEL_DIR']
FEATURE_DIR = yml['SETTING']['FEATURE_DIR']
RAW_DATA_DIR = yml['SETTING']['RAW_DATA_DIR']


class Logger():
    """
    Logger class
    """
    def __init__(self, run_name) -> None:
        # -------------------------------------------------------------------
        # configure root logger
        # -------------------------------------------------------------------
        log_dir = LOG_DIR + run_name + '/'
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)

        # setteing log stream handler
        stream_handler = StreamHandler()
        stream_handler.setLevel(DEBUG)
        stream_handler.setFormatter(Formatter("%(message)s"))
        # settting log file handler
        file_handler = FileHandler(
            log_dir + run_name + ".log"
        )
        file_handler.setLevel(DEBUG)
        file_handler.setFormatter(
            Formatter("%(asctime)s [%(levelname)7s] (%(name)s) %(message)s")
        )
        # setting root logger
        logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])

        self.logger = logging.getLogger(run_name)

    def info(self, message):
        """
        Infomation
        """
        self.logger.info(message)

    def debug(self, message):
        """
        Debug
        """
        self.logger.debug(message)

    def error(self, message):
        """
        Error
        """
        self.logger.error(message)

    