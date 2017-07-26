import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

def main(config):
	prepare_dirs_and_logger(config)

if __name__ == "__main__":
	config, unparsed = get_config()
	main(config)