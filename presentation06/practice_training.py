import logging
from logging import getLogger, Logger
import wandb

PROJECT_NAME = "fine-tuned gemma"


def logger() -> Logger:
    return getLogger(PROJECT_NAME)


def debug(message: str):
    logger().debug(message)


def info(message: str):
    logger().info(message)


def warn(message: str):
    logger().warn(message)


def error(message: str):
    logger().error(message)


def init_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("practice_training.log"),
        ],
    )


init_logger()

wandb.init(project=PROJECT_NAME)
